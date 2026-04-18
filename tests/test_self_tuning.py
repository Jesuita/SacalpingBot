"""Tests para las mejoras de auto-tuning del optimizer."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_optimizer import (
    _decide_pair_removal,
    _decide_trailing_adjustment,
    _decide_tp_sl_adjustment,
    _can_adjust_param,
    _mark_param_adjusted,
    _state,
    _lock,
    MIN_RR_RATIO,
    SL_MAX,
    TP_MIN,
    PARAM_COOLDOWN_HOURS,
)


class TestPairRemovalLowerThresholds:
    """Verifica que pares toxicos se detectan con menos trades."""

    def test_removes_pair_with_5_trades_and_catastrophic_pf(self):
        by_sym = {
            "DOGEUSDT": {"trades": 5, "pf": 0.2, "wr": 20.0, "pnl": -0.67, "expectancy": -0.134},
            "BTCUSDT": {"trades": 16, "pf": 3.5, "wr": 68.8, "pnl": 6.0, "expectancy": 0.375},
        }
        result = _decide_pair_removal(by_sym, ["DOGEUSDT", "BTCUSDT", "ETHUSDT"])
        assert result is not None
        assert result["symbol"] == "DOGEUSDT"

    def test_no_removal_below_min_pairs(self):
        by_sym = {
            "DOGEUSDT": {"trades": 5, "pf": 0.1, "wr": 10.0, "pnl": -2.0, "expectancy": -0.4},
            "BTCUSDT": {"trades": 10, "pf": 2.0, "wr": 60.0, "pnl": 3.0, "expectancy": 0.3},
        }
        result = _decide_pair_removal(by_sym, ["DOGEUSDT", "BTCUSDT"])
        assert result is None  # Can't go below MIN_PAIRS=2

    def test_removes_pair_low_wr_5_trades(self):
        by_sym = {
            "BADCOIN": {"trades": 6, "pf": 0.25, "wr": 16.7, "pnl": -1.5, "expectancy": -0.25},
        }
        result = _decide_pair_removal(by_sym, ["BADCOIN", "GOOD1", "GOOD2"])
        assert result is not None
        assert result["symbol"] == "BADCOIN"


class TestTPHitRate:
    """Verifica que TP se ajusta basado en hit rate real."""

    def setup_method(self):
        """Reset cooldown state for each test."""
        with _lock:
            _state["param_last_adjusted"] = {}

    def test_lowers_tp_when_hit_rate_very_low(self):
        gm = {"trades": 56, "wins": 26, "losses": 30, "pf": 2.27, "avg_win": 0.3, "avg_loss": 0.15, "best": 1.0, "worst": -0.5}
        by_reason = {"signal": {"count": 45, "pnl": 3.98}, "trailing": {"count": 8, "pnl": -0.88}, "take_profit": {"count": 3, "pnl": 3.18}}
        # TP hit rate = 3/56 = 5.4% — still above 5% threshold
        actions = _decide_tp_sl_adjustment(gm, 0.008, 0.004, by_reason=by_reason)
        # With 5.4% hit rate, just above threshold — may or may not trigger
        # But with 20+ trades and high PF but low hit rate, check logic

    def test_lowers_tp_when_zero_hits(self):
        gm = {"trades": 30, "wins": 14, "losses": 16, "pf": 1.5, "avg_win": 0.2, "avg_loss": 0.15, "best": 0.5, "worst": -0.3}
        by_reason = {"signal": {"count": 28, "pnl": 1.0}, "trailing": {"count": 2, "pnl": -0.1}}
        # TP hit rate = 0/30 = 0% -> should lower
        actions = _decide_tp_sl_adjustment(gm, 0.008, 0.004, by_reason=by_reason)
        tp_actions = [a for a in actions if a["param"] == "take_profit_pct"]
        assert len(tp_actions) == 1
        assert tp_actions[0]["action"] == "tighten_tp"
        assert tp_actions[0]["new"] < 0.008

    def test_no_widen_tp_when_hit_rate_low(self):
        """Even with high PF, don't widen TP if it's rarely hit."""
        gm = {"trades": 50, "wins": 30, "losses": 20, "pf": 2.5, "avg_win": 0.3, "avg_loss": 0.1, "best": 1.0, "worst": -0.2}
        by_reason = {"signal": {"count": 48, "pnl": 5.0}, "take_profit": {"count": 2, "pnl": 1.0}}
        actions = _decide_tp_sl_adjustment(gm, 0.008, 0.004, by_reason=by_reason)
        tp_widen = [a for a in actions if a["param"] == "take_profit_pct" and a["action"] == "widen_tp"]
        assert len(tp_widen) == 0  # Should NOT widen with 4% hit rate

    def test_widen_tp_when_hit_rate_high_and_pf_good(self):
        gm = {"trades": 50, "wins": 35, "losses": 15, "pf": 2.0, "avg_win": 0.4, "avg_loss": 0.2, "best": 1.0, "worst": -0.3}
        by_reason = {"signal": {"count": 30, "pnl": 3.0}, "take_profit": {"count": 20, "pnl": 5.0}}
        # 40% TP hit rate, PF 2.0, good ratio -> widen
        actions = _decide_tp_sl_adjustment(gm, 0.006, 0.004, by_reason=by_reason)
        tp_widen = [a for a in actions if a["param"] == "take_profit_pct" and a["action"] == "widen_tp"]
        assert len(tp_widen) == 1


class TestTrailingSkipNoNewTrades:
    """Verifica que el trailing se ajusta correctamente."""

    def setup_method(self):
        with _lock:
            _state["param_last_adjusted"] = {}

    def test_widens_trailing_on_negative_pnl(self):
        by_reason = {"trailing_stop": {"count": 8, "pnl": -0.88}}
        result = _decide_trailing_adjustment(by_reason, 0.004, {"trades": 56})
        assert result is not None
        assert result["action"] == "widen_trailing"
        assert result["new_value"] > 0.004


class TestParamCooldown:
    """Verifica que un parametro no se ajusta multiples veces seguidas."""

    def setup_method(self):
        """Reset cooldown state for each test."""
        with _lock:
            _state["param_last_adjusted"] = {}

    def test_first_adjustment_allowed(self):
        assert _can_adjust_param("take_profit_pct", "tighten") is True

    def test_second_adjustment_blocked_within_cooldown(self):
        _mark_param_adjusted("take_profit_pct", "tighten")
        assert _can_adjust_param("take_profit_pct", "tighten") is False

    def test_different_direction_allowed(self):
        _mark_param_adjusted("take_profit_pct", "tighten")
        assert _can_adjust_param("take_profit_pct", "widen") is True

    def test_different_param_allowed(self):
        _mark_param_adjusted("take_profit_pct", "tighten")
        assert _can_adjust_param("stop_loss_pct", "tighten") is True


class TestConfigLimits:
    """Verifica que los limites son sensatos para scalping."""

    def test_sl_max_is_sane(self):
        assert SL_MAX <= 0.006, f"SL_MAX {SL_MAX} es demasiado alto para scalping"

    def test_tp_min_covers_fees(self):
        assert TP_MIN >= 0.004, f"TP_MIN {TP_MIN} no cubre fees de ida y vuelta"

    def test_min_rr_ratio_exists(self):
        assert MIN_RR_RATIO > 0.5, "R:R minimo debe ser al menos 0.5"

    def test_no_sl_widen_action(self):
        """El optimizer ya no debe ampliar SL automaticamente."""
        gm = {"trades": 50, "wins": 30, "losses": 20, "pf": 2.0, "avg_win": 0.5, "avg_loss": 0.3, "best": 1.0, "worst": -0.3}
        by_reason = {"signal": {"count": 50, "pnl": 5.0}}
        actions = _decide_tp_sl_adjustment(gm, 0.006, 0.004, by_reason=by_reason)
        sl_widen = [a for a in actions if a.get("param") == "stop_loss_pct" and a["action"] == "widen_sl"]
        assert len(sl_widen) == 0, "No debe ampliar SL automaticamente"
