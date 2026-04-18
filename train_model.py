"""
train_model.py — Entrenamiento offline del clasificador ML para el bot.

Lee ml_dataset.csv, calcula labels (future return), entrena modelo y guarda ai_model.json.
Se ejecuta manualmente o via dashboard. No requiere pandas/sklearn — todo manual.

Uso:
    python train_model.py                         # Usa defaults
    python train_model.py --horizon 5 --threshold 0.001
    python train_model.py --auto                  # Re-entrena cada 30 min (daemon)
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ML_DATASET_FILE = "ml_dataset.csv"
AI_MODEL_FILE = "ai_model.json"
TRAIN_LOG_FILE = "train_model.log"

# ─────────────────────────────────────────────
#  CONFIGURACION DE ENTRENAMIENTO
# ─────────────────────────────────────────────

DEFAULT_HORIZON = 5          # Mirar N filas adelante para calcular retorno
DEFAULT_THRESHOLD = 0.0005   # 0.05% minimo para label = 1 (positivo)
LEARNING_RATE = 0.05
EPOCHS = 500
REGULARIZATION = 0.0001       # L2 regularization
MIN_SAMPLES = 100            # Minimo de muestras para entrenar
TEST_SPLIT = 0.2             # 20% para test

# Features a usar del dataset
FEATURE_NAMES = [
    "ema_diff_pct",
    "rsi_norm",
    "macd",
    "macd_signal",
    "macd_hist",
]

# Features extendidas calculadas en runtime
EXTENDED_FEATURES = [
    "ema_diff_pct",
    "rsi_norm",
    "macd",
    "macd_signal",
    "macd_hist",
    "signal_buy",
    "in_position",
    "update_mode_tick",
    "atr_norm",
    "bb_pct",
    "volume_ratio",
    "hour_sin",
    "hour_cos",
]


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(TRAIN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────
#  LECTURA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────

def read_dataset(filepath: str = ML_DATASET_FILE) -> List[Dict]:
    """Lee el CSV del dataset."""
    if not os.path.exists(filepath):
        _log(f"Dataset no encontrado: {filepath}")
        return []
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        _log(f"Error leyendo dataset: {e}")
    return rows


def compute_labels(rows: List[Dict], horizon: int, threshold: float) -> List[Dict]:
    """
    Calcula labels: mira N filas adelante para el mismo symbol.
    Label = 1 si future_return >= threshold, else 0.
    """
    # Agrupar por symbol para calcular retorno intra-symbol
    by_symbol: Dict[str, List] = {}
    for i, row in enumerate(rows):
        sym = row.get("symbol", "")
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append((i, row))

    labeled = []
    for sym, items in by_symbol.items():
        for idx, (orig_i, row) in enumerate(items):
            current_price = _safe_float(row.get("price", 0))
            if current_price <= 0:
                continue

            # Buscar precio futuro (N filas adelante del mismo symbol)
            future_price = None
            if idx + horizon < len(items):
                future_price = _safe_float(items[idx + horizon][1].get("price", 0))

            if future_price is None or future_price <= 0:
                continue

            future_return = (future_price - current_price) / current_price
            label = 1 if future_return >= threshold else 0

            # Construir features
            ema_fast = _safe_float(row.get("ema_fast", 0))
            ema_slow = _safe_float(row.get("ema_slow", 0))
            rsi = _safe_float(row.get("rsi", 50))
            macd = _safe_float(row.get("macd", 0))
            macd_signal = _safe_float(row.get("macd_signal", 0))
            macd_hist = _safe_float(row.get("macd_hist", 0))
            signal_regla = row.get("signal_regla", "HOLD")
            in_position = row.get("in_position", "False")
            update_mode = row.get("update_mode", "candle")

            features = {
                "ema_diff_pct": (ema_fast - ema_slow) / max(current_price, 1e-9),
                "rsi_norm": (rsi - 50.0) / 50.0,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "signal_buy": 1.0 if signal_regla == "BUY" else 0.0,
                "in_position": 1.0 if str(in_position).lower() in ("true", "1") else 0.0,
                "update_mode_tick": 1.0 if str(update_mode).lower() == "tick" else 0.0,
                # Features V3 (pueden faltar en datasets antiguos)
                "atr_norm": _safe_float(row.get("atr_norm", 0)),
                "bb_pct": _safe_float(row.get("bb_pct", 0.5)),
                "volume_ratio": _safe_float(row.get("volume_ratio", 1.0)),
                "hour_sin": _safe_float(row.get("hour_sin", 0)),
                "hour_cos": _safe_float(row.get("hour_cos", 0)),
            }

            labeled.append({
                "features": features,
                "label": label,
                "future_return": future_return,
                "symbol": sym,
            })

    return labeled


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────
#  MODELO: REGRESION LOGISTICA MANUAL
# ─────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalize_features(
    data: List[Dict], feature_names: List[str]
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Calcula media y std, retorna (features_norm, means, stds)."""
    n = len(data)
    num_features = len(feature_names)

    # Calcular media
    means = [0.0] * num_features
    for row in data:
        for i, name in enumerate(feature_names):
            means[i] += row["features"].get(name, 0.0)
    means = [m / n for m in means]

    # Calcular std
    stds = [0.0] * num_features
    for row in data:
        for i, name in enumerate(feature_names):
            diff = row["features"].get(name, 0.0) - means[i]
            stds[i] += diff * diff
    stds = [math.sqrt(s / max(n - 1, 1)) for s in stds]
    # Evitar division por 0
    stds = [s if s > 1e-9 else 1.0 for s in stds]

    return means, stds


def train_logistic_regression(
    train_data: List[Dict],
    feature_names: List[str],
    means: List[float],
    stds: List[float],
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    reg: float = REGULARIZATION,
) -> Tuple[List[float], float]:
    """Entrena regresion logistica con SGD + class weights. Retorna (weights, bias)."""
    n_features = len(feature_names)
    weights = [0.0] * n_features
    bias = 0.0
    n = len(train_data)

    # Class weights para compensar desbalance
    n_pos = sum(1 for r in train_data if r["label"] == 1)
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        w_pos = n / (2 * n_pos)
        w_neg = n / (2 * n_neg)
    else:
        w_pos = 1.0
        w_neg = 1.0
    _log(f"  Class weights: pos={w_pos:.2f} ({n_pos}), neg={w_neg:.2f} ({n_neg})")

    for epoch in range(epochs):
        total_loss = 0.0
        # Mini-batch gradient descent (full batch para datasets chicos)
        grad_w = [0.0] * n_features
        grad_b = 0.0

        for row in train_data:
            # Forward
            z = bias
            x_norm = []
            for i, name in enumerate(feature_names):
                raw = row["features"].get(name, 0.0)
                normalized = (raw - means[i]) / stds[i]
                x_norm.append(normalized)
                z += weights[i] * normalized

            pred = _sigmoid(z)
            label = row["label"]

            # Loss (binary cross entropy)
            eps = 1e-7
            loss = -(label * math.log(pred + eps) + (1 - label) * math.log(1 - pred + eps))
            total_loss += loss

            # Gradientes (con class weighting)
            sample_weight = w_pos if label == 1 else w_neg
            error = (pred - label) * sample_weight
            for i in range(n_features):
                grad_w[i] += error * x_norm[i]
            grad_b += error

        # Update
        for i in range(n_features):
            weights[i] -= lr * (grad_w[i] / n + reg * weights[i])
        bias -= lr * grad_b / n

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / n
            _log(f"  Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    return weights, bias


def evaluate_model(
    test_data: List[Dict],
    feature_names: List[str],
    weights: List[float],
    bias: float,
    means: List[float],
    stds: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evalua el modelo en datos de test."""
    tp = fp = tn = fn = 0

    for row in test_data:
        z = bias
        for i, name in enumerate(feature_names):
            raw = row["features"].get(name, 0.0)
            normalized = (raw - means[i]) / stds[i]
            z += weights[i] * normalized

        pred_prob = _sigmoid(z)
        pred_label = 1 if pred_prob >= threshold else 0
        actual = row["label"]

        if pred_label == 1 and actual == 1:
            tp += 1
        elif pred_label == 1 and actual == 0:
            fp += 1
        elif pred_label == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "positive_rate": round((tp + fn) / total * 100, 1) if total > 0 else 0,
    }


# ─────────────────────────────────────────────
#  MODELO: GRADIENT BOOSTED STUMPS (ENSEMBLE)
# ─────────────────────────────────────────────

N_STUMPS = 100          # Número de stumps (árboles de profundidad 1)
ENSEMBLE_LR = 0.1       # Learning rate del ensemble
N_THRESHOLDS = 20       # Candidatos de corte por feature


def _find_best_stump(
    data: List[Dict],
    feature_names: List[str],
    residuals: List[float],
) -> Dict:
    """Encuentra el mejor stump (feature + threshold) que minimiza MSE sobre residuales."""
    best_loss = float("inf")
    best_stump = {}

    for feat_name in feature_names:
        vals = [row["features"].get(feat_name, 0.0) for row in data]
        # Generar thresholds candidatos (percentiles)
        sorted_vals = sorted(set(vals))
        if len(sorted_vals) <= 1:
            continue
        step = max(1, len(sorted_vals) // N_THRESHOLDS)
        thresholds = sorted_vals[::step]

        for thr in thresholds:
            left_sum = 0.0
            left_count = 0
            right_sum = 0.0
            right_count = 0
            for i, row in enumerate(data):
                v = row["features"].get(feat_name, 0.0)
                if v <= thr:
                    left_sum += residuals[i]
                    left_count += 1
                else:
                    right_sum += residuals[i]
                    right_count += 1

            if left_count == 0 or right_count == 0:
                continue

            left_pred = left_sum / left_count
            right_pred = right_sum / right_count

            # MSE sobre residuales
            loss = 0.0
            for i, row in enumerate(data):
                v = row["features"].get(feat_name, 0.0)
                pred = left_pred if v <= thr else right_pred
                loss += (residuals[i] - pred) ** 2

            if loss < best_loss:
                best_loss = loss
                best_stump = {
                    "feature": feat_name,
                    "threshold": thr,
                    "left_value": left_pred,
                    "right_value": right_pred,
                }

    return best_stump


def train_gradient_boosted_stumps(
    train_data: List[Dict],
    feature_names: List[str],
    n_stumps: int = N_STUMPS,
    lr: float = ENSEMBLE_LR,
) -> Tuple[List[Dict], float]:
    """Entrena ensemble de gradient boosted decision stumps. Retorna (stumps, base_score)."""
    n = len(train_data)
    n_pos = sum(1 for r in train_data if r["label"] == 1)
    base_score = max(min(n_pos / n, 0.99), 0.01)
    _log(f"  Base score: {base_score:.3f} ({n_pos}/{n} positivos)")

    # Log-odds iniciales
    raw_predictions = [math.log(base_score / (1 - base_score))] * n
    stumps = []

    for i in range(n_stumps):
        # Residuales (gradientes de log-loss): label - predicted_prob
        residuals = []
        for j in range(n):
            prob = 1.0 / (1.0 + math.exp(-raw_predictions[j]))
            residuals.append(train_data[j]["label"] - prob)

        stump = _find_best_stump(train_data, feature_names, residuals)
        if not stump:
            _log(f"  Stump {i+1}: sin split válido, deteniendo")
            break

        stumps.append(stump)

        # Actualizar predicciones
        for j, row in enumerate(train_data):
            v = row["features"].get(stump["feature"], 0.0)
            if v <= stump["threshold"]:
                raw_predictions[j] += lr * stump["left_value"]
            else:
                raw_predictions[j] += lr * stump["right_value"]

        if (i + 1) % 25 == 0:
            # Log-loss
            total_loss = 0.0
            for j in range(n):
                p = 1.0 / (1.0 + math.exp(-raw_predictions[j]))
                p = max(min(p, 1 - 1e-7), 1e-7)
                total_loss -= train_data[j]["label"] * math.log(p) + (1 - train_data[j]["label"]) * math.log(1 - p)
            _log(f"  Stump {i+1}/{n_stumps} — LogLoss: {total_loss / n:.4f}")

    _log(f"  Entrenados {len(stumps)} stumps")
    return stumps, base_score


def evaluate_ensemble(
    test_data: List[Dict],
    feature_names: List[str],
    stumps: List[Dict],
    base_score: float,
    lr: float = ENSEMBLE_LR,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evalúa ensemble en datos de test."""
    tp = fp = tn = fn = 0

    for row in test_data:
        raw = math.log(base_score / max(1 - base_score, 1e-9))
        for stump in stumps:
            v = row["features"].get(stump["feature"], 0.0)
            if v <= stump["threshold"]:
                raw += lr * stump["left_value"]
            else:
                raw += lr * stump["right_value"]

        prob = 1.0 / (1.0 + math.exp(-raw))
        pred_label = 1 if prob >= threshold else 0
        actual = row["label"]

        if pred_label == 1 and actual == 1:
            tp += 1
        elif pred_label == 1 and actual == 0:
            fp += 1
        elif pred_label == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "positive_rate": round((tp + fn) / total * 100, 1) if total > 0 else 0,
    }


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def train_and_save(
    horizon: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
    feature_set: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Pipeline completo: leer, procesar, entrenar, evaluar, guardar."""

    _log("=" * 50)
    _log("ENTRENAMIENTO DE MODELO ML")
    _log(f"  Horizon: {horizon} | Threshold: {threshold*100:.3f}%")
    _log("=" * 50)

    # 1. Leer dataset
    rows = read_dataset()
    if len(rows) < MIN_SAMPLES:
        msg = f"Dataset insuficiente: {len(rows)} filas (min {MIN_SAMPLES})"
        _log(msg)
        return {"success": False, "error": msg, "rows": len(rows)}

    _log(f"Dataset: {len(rows)} filas")

    # 2. Calcular labels
    labeled = compute_labels(rows, horizon, threshold)
    if len(labeled) < MIN_SAMPLES:
        msg = f"Datos etiquetados insuficientes: {len(labeled)} (min {MIN_SAMPLES})"
        _log(msg)
        return {"success": False, "error": msg, "labeled": len(labeled)}

    positives = sum(1 for d in labeled if d["label"] == 1)
    negatives = len(labeled) - positives
    _log(f"Etiquetados: {len(labeled)} | Positivos: {positives} ({positives/len(labeled)*100:.1f}%) | Negativos: {negatives}")

    if positives < 10 or negatives < 10:
        msg = f"Clases desbalanceadas extremas: {positives} pos, {negatives} neg"
        _log(msg)
        return {"success": False, "error": msg}

    # 3. Feature set
    if feature_set is None:
        feature_set = list(EXTENDED_FEATURES)
    _log(f"Features: {feature_set}")

    # 4. Split train/test
    split_idx = int(len(labeled) * (1 - TEST_SPLIT))
    train_data = labeled[:split_idx]
    test_data = labeled[split_idx:]
    _log(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # 5. Normalizar (para logistic regression)
    means, stds = normalize_features(train_data, feature_set)

    # 6a. Entrenar logistic regression
    _log("Entrenando modelo logístico...")
    weights, bias = train_logistic_regression(train_data, feature_set, means, stds)
    lr_train = evaluate_model(train_data, feature_set, weights, bias, means, stds)
    lr_test = evaluate_model(test_data, feature_set, weights, bias, means, stds)
    _log(f"  Logistic — Test Acc: {lr_test['accuracy']:.3f} | F1: {lr_test['f1']:.3f}")

    # 6b. Entrenar ensemble (gradient boosted stumps)
    _log("Entrenando ensemble (gradient boosted stumps)...")
    stumps, base_score = train_gradient_boosted_stumps(train_data, feature_set)
    ens_train = evaluate_ensemble(train_data, feature_set, stumps, base_score)
    ens_test = evaluate_ensemble(test_data, feature_set, stumps, base_score)
    _log(f"  Ensemble — Test Acc: {ens_test['accuracy']:.3f} | F1: {ens_test['f1']:.3f}")

    # 7. Elegir el mejor por F1 en test
    use_ensemble = ens_test["f1"] >= lr_test["f1"]
    chosen = "ensemble" if use_ensemble else "logistic"
    train_metrics = ens_train if use_ensemble else lr_train
    test_metrics = ens_test if use_ensemble else lr_test
    _log(f"Modelo elegido: {chosen} (F1 test: {test_metrics['f1']:.3f})")

    # 8. Feature importance
    if use_ensemble:
        feat_counts: Dict[str, int] = {}
        for s in stumps:
            feat_counts[s["feature"]] = feat_counts.get(s["feature"], 0) + 1
        _log("Feature importance (stumps):")
        for name in sorted(feat_counts, key=feat_counts.get, reverse=True):
            _log(f"  {name:20s}: {feat_counts[name]} splits")
    else:
        _log("Feature weights (logistic):")
        for i, name in enumerate(feature_set):
            _log(f"  {name:20s}: {weights[i]:+.4f} (mean={means[i]:.6f}, std={stds[i]:.6f})")
        _log(f"  {'bias':20s}: {bias:+.4f}")

    # 9. Guardar modelo
    if use_ensemble:
        model = {
            "model_type": "ensemble",
            "feature_names": feature_set,
            "stumps": [{k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()} for s in stumps],
            "base_score": round(base_score, 6),
            "learning_rate": ENSEMBLE_LR,
            "metadata": {
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset_rows": len(rows),
                "labeled_samples": len(labeled),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "horizon": horizon,
                "threshold": threshold,
                "positive_rate": round(positives / len(labeled) * 100, 1),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "n_stumps": len(stumps),
                "competitor": {"logistic_f1": lr_test["f1"]},
            },
        }
    else:
        model = {
            "model_type": "logistic",
            "feature_names": feature_set,
            "weights": [round(w, 6) for w in weights],
            "means": [round(m, 6) for m in means],
            "stds": [round(s, 6) for s in stds],
            "bias": round(bias, 6),
            "metadata": {
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset_rows": len(rows),
                "labeled_samples": len(labeled),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "horizon": horizon,
                "threshold": threshold,
                "positive_rate": round(positives / len(labeled) * 100, 1),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "regularization": REGULARIZATION,
                "competitor": {"ensemble_f1": ens_test["f1"]},
            },
        }

    with open(AI_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    _log(f"Modelo guardado en {AI_MODEL_FILE}")
    _log(f"Accuracy test: {test_metrics['accuracy']:.1%} | F1: {test_metrics['f1']:.1%}")
    _log("=" * 50)

    return {
        "success": True,
        "model_file": AI_MODEL_FILE,
        "model_type": chosen,
        "dataset_rows": len(rows),
        "labeled_samples": len(labeled),
        "positive_rate": round(positives / len(labeled) * 100, 1),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def get_model_status() -> Dict[str, Any]:
    """Retorna estado del modelo actual (si existe)."""
    if not os.path.exists(AI_MODEL_FILE):
        return {"exists": False}
    try:
        with open(AI_MODEL_FILE, "r", encoding="utf-8") as f:
            model = json.load(f)
        meta = model.get("metadata", {})
        return {
            "exists": True,
            "model_type": model.get("model_type", "logistic"),
            "trained_at": meta.get("trained_at", "?"),
            "dataset_rows": meta.get("dataset_rows", 0),
            "labeled_samples": meta.get("labeled_samples", 0),
            "test_accuracy": meta.get("test_metrics", {}).get("accuracy", 0),
            "test_f1": meta.get("test_metrics", {}).get("f1", 0),
            "test_precision": meta.get("test_metrics", {}).get("precision", 0),
            "features": model.get("feature_names", []),
            "horizon": meta.get("horizon", "?"),
            "threshold": meta.get("threshold", "?"),
        }
    except Exception:
        return {"exists": True, "error": "No se pudo leer"}


# ─────────────────────────────────────────────
#  AUTO-RETRAIN (daemon mode)
# ─────────────────────────────────────────────

_retrain_thread = None
_retrain_stop = None


def start_auto_retrain(interval_minutes: int = 30, horizon: int = DEFAULT_HORIZON, threshold: float = DEFAULT_THRESHOLD):
    """Inicia reentrenamiento automatico cada N minutos."""
    import threading
    global _retrain_thread, _retrain_stop

    if _retrain_thread and _retrain_thread.is_alive():
        return

    _retrain_stop = threading.Event()

    def _loop():
        _retrain_stop.wait(60)  # Esperar 1 min inicial
        while not _retrain_stop.is_set():
            try:
                _log("[AUTO-RETRAIN] Iniciando reentrenamiento...")
                result = train_and_save(horizon=horizon, threshold=threshold)
                if result.get("success"):
                    _log(f"[AUTO-RETRAIN] Completado. Test acc={result['test_metrics']['accuracy']:.3f}")
                else:
                    _log(f"[AUTO-RETRAIN] No se pudo entrenar: {result.get('error', '?')}")
            except Exception as e:
                _log(f"[AUTO-RETRAIN] Error: {e}")
            _retrain_stop.wait(interval_minutes * 60)

    _retrain_thread = threading.Thread(target=_loop, daemon=True, name="MLAutoRetrain")
    _retrain_thread.start()
    _log(f"[AUTO-RETRAIN] Iniciado. Intervalo: {interval_minutes} min")


def stop_auto_retrain():
    global _retrain_stop
    if _retrain_stop:
        _retrain_stop.set()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo ML para el bot")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Filas adelante para calcular retorno")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Retorno minimo para label positivo")
    parser.add_argument("--auto", action="store_true", help="Modo daemon: re-entrena cada 30 min")
    parser.add_argument("--interval", type=int, default=30, help="Intervalo auto-retrain en minutos")

    args = parser.parse_args()

    if args.auto:
        start_auto_retrain(interval_minutes=args.interval, horizon=args.horizon, threshold=args.threshold)
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            stop_auto_retrain()
            _log("Auto-retrain detenido.")
    else:
        result = train_and_save(horizon=args.horizon, threshold=args.threshold)
        if result.get("success"):
            print(f"\nModelo listo. Test accuracy: {result['test_metrics']['accuracy']:.1%}")
        else:
            print(f"\nError: {result.get('error', 'desconocido')}")
            sys.exit(1)
