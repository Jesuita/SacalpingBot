"""Tests de Intelligence Engine: evaluate_entry()."""
import time
from unittest.mock import patch, MagicMock
import intelligence_engine as ie


def _base_kwargs():
    """Parámetros base para evaluate_entry."""
    return dict(
        symbol="BTCUSDT",
        price=50000.0,
        ema_fast=50100.0,
        ema_slow=50000.0,
        rsi=55.0,
        macd_hist=10.0,
        candles=[
            {"open": 50000, "high": 50200, "low": 49800, "close": 50100, "volume": 100}
            for _ in range(30)
        ],
        raw_signal="BUY",
    )


class TestEvaluateEntryDisabled:
    def test_intel_disabled_approves(self):
        """Con intel deshabilitado, siempre aprueba."""
        with patch.object(ie, "_get_intel_config", return_value={"intel_enabled": False}):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["approved"] is True
        assert result["score"] == 50
        assert result["blocks"] == []

    def test_intel_disabled_details(self):
        with patch.object(ie, "_get_intel_config", return_value={"intel_enabled": False}):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["details"].get("intel_disabled") is True


class TestEvaluateEntryLayers:
    def _config_all_off(self):
        return {
            "intel_enabled": True,
            "intel_hours_enabled": False,
            "intel_regime_enabled": False,
            "intel_volume_enabled": False,
            "intel_score_enabled": False,
            "intel_correlation_enabled": False,
            "intel_adaptive_trailing_enabled": False,
        }

    def test_all_layers_off_approves(self):
        """Con todas las capas deshabilitadas, aprueba."""
        with patch.object(ie, "_get_intel_config", return_value=self._config_all_off()):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["approved"] is True

    def test_hours_block(self):
        """Hora tóxica bloquea entrada."""
        cfg = self._config_all_off()
        cfg["intel_hours_enabled"] = True
        current_hour = int(time.strftime("%H"))

        orig_state = ie._state.copy()
        try:
            ie._state["blocked_hours"] = [current_hour]
            ie._state["hourly_profile"] = {current_hour: {"pf": 0.3, "pnl": -5.0}}
            with patch.object(ie, "_get_intel_config", return_value=cfg):
                result = ie.evaluate_entry(**_base_kwargs())
            assert result["approved"] is False
            assert any("hora_toxica" in b for b in result["blocks"])
        finally:
            ie._state.update(orig_state)

    def test_volume_spike_blocks(self):
        """Spike de volumen bloquea."""
        cfg = self._config_all_off()
        cfg["intel_volume_enabled"] = True
        cfg["intel_volume_spike_mult"] = 3.0
        cfg["intel_volume_drought_mult"] = 0.3

        with patch.object(ie, "_get_intel_config", return_value=cfg), \
             patch.object(ie, "analyze_volume", return_value={"status": "spike", "ratio": 5.0}):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["approved"] is False
        assert any("volumen_spike" in b for b in result["blocks"])

    def test_volume_drought_blocks(self):
        """Sequía de volumen bloquea."""
        cfg = self._config_all_off()
        cfg["intel_volume_enabled"] = True
        cfg["intel_volume_spike_mult"] = 3.0
        cfg["intel_volume_drought_mult"] = 0.3

        with patch.object(ie, "_get_intel_config", return_value=cfg), \
             patch.object(ie, "analyze_volume", return_value={"status": "drought", "ratio": 0.1}):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["approved"] is False
        assert any("volumen_sequia" in b for b in result["blocks"])

    def test_regime_ranging_blocks(self):
        """Régimen ranging bloquea si configurado."""
        cfg = self._config_all_off()
        cfg["intel_regime_enabled"] = True
        cfg["intel_regime_block_ranging"] = True

        with patch.object(ie, "_get_intel_config", return_value=cfg), \
             patch.object(ie, "detect_regime", return_value={
                 "regime": "ranging", "choppiness": 70, "slope_pct": 0.01, "atr_pct": 0.5
             }):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["approved"] is False
        assert any("regimen_ranging" in b for b in result["blocks"])

    def test_low_score_blocks(self):
        """Score bajo bloquea."""
        cfg = self._config_all_off()
        cfg["intel_score_enabled"] = True
        cfg["intel_score_min_buy"] = 60

        with patch.object(ie, "_get_intel_config", return_value=cfg), \
             patch.object(ie, "calculate_confidence_score", return_value={
                 "score": 30, "grade": "D", "components": {}
             }):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["approved"] is False
        assert any("score_bajo" in b for b in result["blocks"])

    def test_correlation_blocks(self):
        """Máximo de posiciones correlacionadas bloquea."""
        cfg = self._config_all_off()
        cfg["intel_correlation_enabled"] = True
        cfg["intel_correlation_max_entries"] = 2

        with patch.object(ie, "_get_intel_config", return_value=cfg), \
             patch.object(ie, "is_correlation_blocked", return_value=True), \
             patch.object(ie, "get_open_position_count", return_value=3):
            result = ie.evaluate_entry(**_base_kwargs())
        assert result["approved"] is False
        assert any("correlacion_max" in b for b in result["blocks"])


class TestEvaluateEntryOutput:
    def test_result_structure(self):
        """El resultado tiene todas las claves esperadas."""
        with patch.object(ie, "_get_intel_config", return_value={"intel_enabled": False}):
            result = ie.evaluate_entry(**_base_kwargs())
        assert "approved" in result
        assert "score" in result
        assert "blocks" in result
        assert "adjustments" in result
        assert "details" in result
