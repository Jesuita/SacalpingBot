"""Tests para regime params y grid search functions."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock

import auto_optimizer
from auto_optimizer import (
    load_regime_params,
    save_regime_params,
    get_params_for_regime,
    update_regime_from_grid_search,
    _DEFAULT_REGIME_PARAMS,
    REGIME_PARAMS_FILE,
)


@pytest.fixture(autouse=True)
def cleanup_regime_file():
    yield
    if os.path.exists(REGIME_PARAMS_FILE):
        os.remove(REGIME_PARAMS_FILE)


class TestLoadRegimeParams:
    def test_returns_defaults_when_no_file(self):
        if os.path.exists(REGIME_PARAMS_FILE):
            os.remove(REGIME_PARAMS_FILE)
        params = load_regime_params()
        assert params == _DEFAULT_REGIME_PARAMS

    def test_loads_from_file(self):
        custom = {"trending_up": {"take_profit_pct": 0.02}}
        with open(REGIME_PARAMS_FILE, "w") as f:
            json.dump(custom, f)
        assert load_regime_params() == custom

    def test_returns_defaults_on_corrupt_file(self):
        with open(REGIME_PARAMS_FILE, "w") as f:
            f.write("NOT JSON")
        assert load_regime_params() == _DEFAULT_REGIME_PARAMS


class TestSaveRegimeParams:
    def test_creates_file(self):
        data = {"volatile": {"stop_loss_pct": 0.01}}
        save_regime_params(data)
        with open(REGIME_PARAMS_FILE) as f:
            assert json.load(f) == data


class TestGetParamsForRegime:
    def test_known_regime(self):
        p = get_params_for_regime("trending_up")
        assert p["take_profit_pct"] == _DEFAULT_REGIME_PARAMS["trending_up"]["take_profit_pct"]

    def test_unknown_regime_fallback(self):
        p = get_params_for_regime("unknown_regime")
        assert "take_profit_pct" in p


class TestUpdateRegimeFromGridSearch:
    def test_no_grid_search_results(self):
        with patch.object(auto_optimizer, "_gs_state", {"best": None}):
            result = update_regime_from_grid_search("trending_up")
        assert "error" in result

    def test_updates_regime_with_best(self):
        best = {
            "take_profit_pct": 0.015,
            "stop_loss_pct": 0.008,
            "trailing_stop_pct": 0.007,
        }
        with patch.object(auto_optimizer, "_gs_state", {"best": best}):
            with patch("auto_optimizer._log_action"):
                result = update_regime_from_grid_search("volatile")
        assert result["ok"] is True
        assert result["regime"] == "volatile"
        assert result["params"]["take_profit_pct"] == 0.015
        # Verify file was saved
        saved = load_regime_params()
        assert saved["volatile"]["take_profit_pct"] == 0.015
