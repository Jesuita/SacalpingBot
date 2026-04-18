"""Tests para los endpoints del dashboard."""
import json
import os
import sys
import tempfile

import pytest

# Parchear archivos de estado antes de importar dashboard
_tmpdir = tempfile.mkdtemp()
os.environ.setdefault("TRADING_MODE", "paper")

# Crear archivos de estado temporales para evitar side-effects
_state_file = os.path.join(_tmpdir, "bot_state.json")
_config_file = os.path.join(_tmpdir, "runtime_config.json")

# Parchear paths de dashboard si es posible
import dashboard
dashboard.STATE_FILE = _state_file
dashboard.CONFIG_FILE = _config_file

# Escribir state y config iniciales
with open(_state_file, "w") as f:
    json.dump(dashboard.default_state(), f)
with open(_config_file, "w") as f:
    json.dump(dashboard.DEFAULT_CONFIG.copy(), f)

from fastapi.testclient import TestClient

client = TestClient(dashboard.app)


class TestGetIndex:
    def test_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<!doctype html>" in resp.text.lower()

    def test_no_cache_header(self):
        resp = client.get("/")
        assert "no-store" in resp.headers.get("cache-control", "")


class TestGetConfig:
    def test_returns_json(self):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "symbols" in data
        assert "trade_pct" in data

    def test_default_values(self):
        resp = client.get("/config")
        data = resp.json()
        assert isinstance(data["symbols"], list)
        assert data["trade_pct"] > 0
        assert data["take_profit_pct"] > 0


class TestPostConfig:
    def test_update_trade_pct(self):
        resp = client.post("/config", json={"trade_pct": 0.80})
        assert resp.status_code == 200
        data = resp.json()
        assert abs(data["trade_pct"] - 0.80) < 0.01

    def test_clamp_trade_pct_high(self):
        resp = client.post("/config", json={"trade_pct": 5.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["trade_pct"] <= 1.0

    def test_clamp_trade_pct_low(self):
        resp = client.post("/config", json={"trade_pct": 0.001})
        assert resp.status_code == 200
        data = resp.json()
        assert data["trade_pct"] >= 0.01

    def test_preserves_other_fields(self):
        resp = client.post("/config", json={"poll_seconds": 30})
        assert resp.status_code == 200
        data = resp.json()
        assert data["poll_seconds"] == 30
        assert "symbols" in data

    def test_symbols_validation(self):
        resp = client.post("/config", json={"symbols": "BTCUSDT,INVALID"})
        assert resp.status_code == 200
        data = resp.json()
        assert "BTCUSDT" in data["symbols"]
        assert "INVALID" not in data["symbols"]

    def test_ai_adaptive_sizing_fields(self):
        resp = client.post("/config", json={
            "ai_adaptive_sizing": True,
            "ai_high_confidence": 0.80,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ai_adaptive_sizing"] is True
        assert abs(data["ai_high_confidence"] - 0.80) < 0.01


class TestGetState:
    def test_returns_json(self):
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "pairs" in data or "mode" in data


class TestGetTradesAnalysis:
    def test_returns_json(self):
        resp = client.get("/trades-analysis")
        assert resp.status_code == 200


class TestGetEvents:
    def test_returns_json(self):
        resp = client.get("/events")
        assert resp.status_code == 200


class TestGetDatasetSummary:
    def test_returns_json(self):
        resp = client.get("/dataset-summary")
        assert resp.status_code == 200


class TestGetRegimeParams:
    def test_returns_json(self):
        resp = client.get("/regime-params")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_contains_default_regimes(self):
        resp = client.get("/regime-params")
        data = resp.json()
        for regime in ("trending_up", "trending_down", "ranging", "volatile"):
            assert regime in data
            assert "take_profit_pct" in data[regime]


class TestPostRegimeParams:
    def test_save_valid_params(self):
        payload = {
            "trending_up": {"take_profit_pct": 0.015, "stop_loss_pct": 0.007}
        }
        resp = client.post("/regime-params", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "trending_up" in data["saved"]

    def test_rejects_empty_body(self):
        resp = client.post("/regime-params", json={})
        assert resp.status_code == 400

    def test_ignores_unknown_regimes(self):
        payload = {"unknown_regime": {"take_profit_pct": 0.01}}
        resp = client.post("/regime-params", json=payload)
        assert resp.status_code == 400

    def test_clamps_values(self):
        payload = {"volatile": {"take_profit_pct": 0.9, "trade_pct_mult": 5.0}}
        resp = client.post("/regime-params", json=payload)
        data = resp.json()
        assert data["saved"]["volatile"]["take_profit_pct"] <= 0.05
        assert data["saved"]["volatile"]["trade_pct_mult"] <= 2.0


class TestUpdateRegimeFromGrid:
    def test_no_grid_results(self):
        resp = client.post("/regime-params/update-from-grid", json={"regime": "ranging"})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_invalid_regime(self):
        resp = client.post("/regime-params/update-from-grid", json={"regime": "invalid"})
        assert resp.status_code == 400


class TestGridSearchState:
    def test_returns_json(self):
        resp = client.get("/grid-search/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
