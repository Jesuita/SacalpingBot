"""Tests de train_model: pipeline de entrenamiento ML."""
import json
import os
import tempfile
from unittest.mock import patch
import train_model as tm


def _make_rows(n=200, positive_pct=0.5):
    """Genera filas sintéticas para ml_dataset.csv."""
    rows = []
    for i in range(n):
        label = 1 if i < int(n * positive_pct) else 0
        rows.append({
            "timestamp": f"2025-01-01T00:{i:02d}:00",
            "symbol": "BTCUSDT",
            "price": 50000 + i * 10 * (1 if label else -1),
            "ema_fast": 50050 + i,
            "ema_slow": 50000 + i * 0.5,
            "rsi": 55.0 + (i % 20) - 10,
            "macd": 5.0 + i * 0.1 * (1 if label else -1),
            "macd_signal": 4.0,
            "macd_hist": 1.0 + i * 0.05 * (1 if label else -1),
            "signal": "BUY" if label else "HOLD",
            "in_position": 0,
            "balance": 100.0,
            "update_mode": "candle",
        })
    return rows


class TestTrainAndSaveInsufficientData:
    def test_too_few_rows(self):
        with patch.object(tm, "read_dataset", return_value=[{"price": 100}] * 10):
            result = tm.train_and_save()
        assert result["success"] is False
        assert "insuficiente" in result["error"].lower() or "min" in result["error"].lower()


class TestTrainAndSavePipeline:
    def test_full_pipeline(self):
        """Pipeline completo con datos sintéticos."""
        rows = _make_rows(200, positive_pct=0.4)

        # compute_labels espera estructura {"features": {...}, "label": int, ...}
        labeled = []
        for i in range(len(rows) - 5):
            r = rows[i]
            future_price = rows[i + 5]["price"]
            ret = (future_price - r["price"]) / r["price"]
            label = 1 if i % 3 == 0 else 0  # balance forzado
            ema_f = r["ema_fast"]
            ema_s = r["ema_slow"]
            labeled.append({
                "features": {
                    "ema_diff_pct": (ema_f - ema_s) / max(r["price"], 1e-9),
                    "rsi_norm": (r["rsi"] - 50) / 50,
                    "macd": r["macd"],
                    "macd_signal": r["macd_signal"],
                    "macd_hist": r["macd_hist"],
                    "signal_buy": 1.0 if r["signal"] == "BUY" else 0.0,
                    "in_position": float(r["in_position"]),
                    "update_mode_tick": 0.0,
                },
                "label": label,
                "future_return": ret,
                "symbol": r["symbol"],
            })

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_model = f.name

        try:
            with patch.object(tm, "read_dataset", return_value=rows), \
                 patch.object(tm, "compute_labels", return_value=labeled), \
                 patch.object(tm, "AI_MODEL_FILE", tmp_model):
                result = tm.train_and_save()

            assert result["success"] is True
            assert "train_metrics" in result
            assert "test_metrics" in result
            assert result["train_metrics"]["accuracy"] >= 0

            # Verificar que se guardó el modelo
            with open(tmp_model, "r") as f:
                model = json.load(f)
            assert "feature_names" in model
            assert "model_type" in model
            if model["model_type"] == "logistic":
                assert "weights" in model
                assert "bias" in model
            else:
                assert "stumps" in model
                assert "base_score" in model
        finally:
            if os.path.exists(tmp_model):
                os.unlink(tmp_model)


class TestTrainConstants:
    def test_min_samples(self):
        assert tm.MIN_SAMPLES >= 50

    def test_feature_names_exist(self):
        assert len(tm.FEATURE_NAMES) > 0
        assert len(tm.EXTENDED_FEATURES) >= len(tm.FEATURE_NAMES)

    def test_test_split_valid(self):
        assert 0 < tm.TEST_SPLIT < 1
