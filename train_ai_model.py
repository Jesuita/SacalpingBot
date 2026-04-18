import argparse
import csv
import datetime
import json
import math
from typing import Dict, List, Tuple

DATASET_FILE = "ml_dataset.csv"
MODEL_FILE = "ai_model.json"

FEATURE_NAMES = [
    "ema_diff_pct",
    "rsi_norm",
    "macd",
    "macd_signal",
    "macd_hist",
    "signal_buy",
    "in_position",
    "update_mode_tick",
]


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def build_feature(row: Dict[str, str]) -> List[float]:
    price = max(safe_float(row.get("price", 0.0), 0.0), 1e-9)
    ema_fast = safe_float(row.get("ema_fast", 0.0), 0.0)
    ema_slow = safe_float(row.get("ema_slow", 0.0), 0.0)
    rsi = safe_float(row.get("rsi", 50.0), 50.0)
    macd = safe_float(row.get("macd", 0.0), 0.0)
    macd_signal = safe_float(row.get("macd_signal", 0.0), 0.0)
    macd_hist = safe_float(row.get("macd_hist", 0.0), 0.0)
    signal = (row.get("signal_regla", "") or "").strip().upper()
    in_position = 1.0 if str(row.get("in_position", "")).strip().lower() in {"1", "true", "yes"} else 0.0
    update_mode = (row.get("update_mode", "") or "").strip().lower()

    ema_diff_pct = (ema_fast - ema_slow) / price
    rsi_norm = (rsi - 50.0) / 50.0

    return [
        ema_diff_pct,
        rsi_norm,
        macd,
        macd_signal,
        macd_hist,
        1.0 if signal == "BUY" else 0.0,
        in_position,
        1.0 if update_mode == "tick" else 0.0,
    ]


def build_labeled_dataset(rows: List[Dict[str, str]], horizon_rows: int, min_future_return: float) -> Tuple[List[List[float]], List[int]]:
    by_symbol: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        symbol = (r.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        by_symbol.setdefault(symbol, []).append(r)

    x_all: List[List[float]] = []
    y_all: List[int] = []

    for symbol_rows in by_symbol.values():
        n = len(symbol_rows)
        for i in range(0, max(0, n - horizon_rows)):
            now = symbol_rows[i]
            fut = symbol_rows[i + horizon_rows]
            price_now = max(safe_float(now.get("price", 0.0), 0.0), 1e-9)
            price_fut = safe_float(fut.get("price", 0.0), 0.0)
            future_ret = (price_fut - price_now) / price_now

            x_all.append(build_feature(now))
            y_all.append(1 if future_ret >= min_future_return else 0)

    return x_all, y_all


def train_test_split_time(x: List[List[float]], y: List[int], train_ratio: float = 0.8):
    split = int(len(x) * train_ratio)
    split = min(max(split, 1), len(x) - 1)
    return x[:split], y[:split], x[split:], y[split:]


def compute_norm_stats(x: List[List[float]]) -> Tuple[List[float], List[float]]:
    cols = len(x[0])
    means = []
    stds = []
    for j in range(cols):
        col = [row[j] for row in x]
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / len(col)
        std = math.sqrt(var)
        if std < 1e-9:
            std = 1.0
        means.append(mean)
        stds.append(std)
    return means, stds


def normalize(x: List[List[float]], means: List[float], stds: List[float]) -> List[List[float]]:
    out = []
    for row in x:
        out.append([(row[j] - means[j]) / stds[j] for j in range(len(row))])
    return out


def train_logreg_sgd(x: List[List[float]], y: List[int], lr: float = 0.05, epochs: int = 60, l2: float = 1e-4, pos_weight: float = 1.0):
    n_features = len(x[0])
    w = [0.0] * n_features
    b = 0.0

    for _ in range(epochs):
        for xi, yi in zip(x, y):
            z = b
            for j in range(n_features):
                z += w[j] * xi[j]
            p = sigmoid(z)
            class_w = pos_weight if yi == 1 else 1.0
            err = (p - yi) * class_w

            for j in range(n_features):
                grad = err * xi[j] + l2 * w[j]
                w[j] -= lr * grad
            b -= lr * err
    return w, b


def predict_proba(x: List[List[float]], w: List[float], b: float) -> List[float]:
    out = []
    for xi in x:
        z = b
        for j in range(len(w)):
            z += w[j] * xi[j]
        out.append(sigmoid(z))
    return out


def classification_metrics(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_prob):
        pred = 1 if yp >= threshold else 0
        if pred == 1 and yt == 1:
            tp += 1
        elif pred == 1 and yt == 0:
            fp += 1
        elif pred == 0 and yt == 0:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_best_threshold(y_true: List[int], y_prob: List[float]) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for i in range(20, 81):
        t = i / 100.0
        m = classification_metrics(y_true, y_prob, threshold=t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
    return best_t


def main():
    parser = argparse.ArgumentParser(description="Entrena modelo logistico para filtro AI del bot")
    parser.add_argument("--dataset", default=DATASET_FILE)
    parser.add_argument("--out", default=MODEL_FILE)
    parser.add_argument("--horizon-rows", type=int, default=5)
    parser.add_argument("--min-future-return", type=float, default=0.0005)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=0.0, help="0 = auto balance")
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    if len(rows) < 200:
        raise RuntimeError("Dataset insuficiente para entrenar (minimo recomendado: 200 filas)")

    x_all, y_all = build_labeled_dataset(rows, args.horizon_rows, args.min_future_return)
    if len(x_all) < 200:
        raise RuntimeError("No hay suficientes muestras etiquetadas para entrenar")

    x_train, y_train, x_test, y_test = train_test_split_time(x_all, y_all, train_ratio=0.8)

    means, stds = compute_norm_stats(x_train)
    x_train_n = normalize(x_train, means, stds)
    x_test_n = normalize(x_test, means, stds)

    positives = sum(1 for v in y_train if v == 1)
    negatives = max(1, len(y_train) - positives)
    auto_pos_weight = negatives / max(positives, 1)
    pos_weight = args.pos_weight if args.pos_weight > 0 else auto_pos_weight

    w, b = train_logreg_sgd(x_train_n, y_train, lr=args.lr, epochs=args.epochs, pos_weight=pos_weight)

    train_prob = predict_proba(x_train_n, w, b)
    test_prob = predict_proba(x_test_n, w, b)
    best_threshold = find_best_threshold(y_train, train_prob)
    train_metrics = classification_metrics(y_train, train_prob, threshold=best_threshold)
    test_metrics = classification_metrics(y_test, test_prob, threshold=best_threshold)

    artifact = {
        "type": "logistic_regression_sgd",
        "feature_names": FEATURE_NAMES,
        "weights": w,
        "bias": b,
        "means": means,
        "stds": stds,
        "meta": {
            "trained_at": datetime.datetime.now().isoformat(),
            "dataset": args.dataset,
            "samples_total": len(x_all),
            "samples_train": len(x_train),
            "samples_test": len(x_test),
            "horizon_rows": args.horizon_rows,
            "min_future_return": args.min_future_return,
            "threshold": args.threshold,
            "recommended_threshold": best_threshold,
            "epochs": args.epochs,
            "lr": args.lr,
            "pos_weight": pos_weight,
            "metrics_train": train_metrics,
            "metrics_test": test_metrics,
        },
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=2)

    print("Modelo entrenado y guardado en", args.out)
    print("Train:", train_metrics)
    print("Test :", test_metrics)


if __name__ == "__main__":
    main()
