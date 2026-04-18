"""Tests para RealWallet con APIs mockeadas."""
import math
from unittest.mock import patch, MagicMock, PropertyMock
import pytest


# Mock antes de importar para evitar llamadas reales
@patch("scalping_bot.log", MagicMock())
@patch("scalping_bot.log_trade", MagicMock())
@patch("scalping_bot.EVENT_LOGGER", MagicMock())
class TestRealWallet:
    """Tests de RealWallet con Binance mockeado."""

    def _make_wallet(self):
        """Crea un RealWallet con API mockeada."""
        from scalping_bot import RealWallet

        with patch.object(RealWallet, "refresh_balances") as mock_rb, \
             patch.object(RealWallet, "_reconcile_position"):
            wallet = RealWallet("BTCUSDT", 100.0, "test_key", "test_secret", lambda: {
                "trade_pct": 0.95,
                "take_profit_pct": 0.008,
                "stop_loss_pct": 0.005,
                "trailing_stop_pct": 0.004,
            })
            wallet.usdt = 100.0
            wallet.asset = 0.0
        return wallet


    def test_init_creates_session(self):
        w = self._make_wallet()
        assert w.api_key == "test_key"
        assert w.session is not None
        assert w.asset_symbol == "BTC"

    def test_get_symbol_filters(self):
        w = self._make_wallet()
        exchange_resp = MagicMock()
        exchange_resp.status_code = 200
        exchange_resp.json.return_value = {
            "symbols": [{
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001", "maxQty": "9000"},
                    {"filterType": "NOTIONAL", "minNotional": "10.0"},
                ]
            }]
        }
        with patch("scalping_bot.requests.get", return_value=exchange_resp):
            filters = w._get_symbol_filters("BTCUSDT")
        assert filters["step_size"] == 0.00001
        assert filters["min_qty"] == 0.00001
        assert filters["min_notional"] == 10.0

    def test_adjust_quantity(self):
        w = self._make_wallet()
        assert w._adjust_quantity(0.123456, 0.001) == 0.123
        assert w._adjust_quantity(0.999, 0.01) == 0.99
        assert w._adjust_quantity(5.0, 1.0) == 5.0

    def test_buy_rejects_below_min_notional(self):
        w = self._make_wallet()
        w.usdt = 3.0  # Below min_notional
        with patch.object(w, "refresh_balances"), \
             patch.object(w, "_get_symbol_filters", return_value={"min_notional": 10.0}):
            result = w.buy(50000.0)
        assert result is None

    def test_buy_adjusts_step_size(self):
        w = self._make_wallet()
        w.usdt = 100.0
        mock_order = {
            "orderId": 12345, "status": "FILLED", "executedQty": "0.00190",
            "fills": [{"price": "50000.0", "qty": "0.00190"}],
        }
        with patch.object(w, "refresh_balances"), \
             patch.object(w, "_get_symbol_filters", return_value={
                 "min_notional": 10.0, "step_size": 0.0001, "min_qty": 0.0001
             }), \
             patch.object(w, "execute_order", return_value=mock_order), \
             patch.object(w, "_record_trade"):
            result = w.buy(50000.0)
        assert result is not None
        assert result["type"] == "BUY"

    def test_sell_adjusts_step_size(self):
        w = self._make_wallet()
        w.asset = 0.001999
        w.entry_price = 50000.0
        mock_order = {
            "orderId": 12346, "status": "FILLED", "executedQty": "0.00190",
            "fills": [{"price": "50500.0", "qty": "0.00190"}],
        }
        with patch.object(w, "refresh_balances"), \
             patch.object(w, "_get_symbol_filters", return_value={
                 "step_size": 0.0001
             }), \
             patch.object(w, "execute_order", return_value=mock_order), \
             patch.object(w, "_record_trade"), \
             patch.object(w, "_reset_position"):
            result = w.sell(50500.0, "take_profit")
        assert result is not None
        assert result["pnl"] > 0

    def test_sell_rejects_zero_qty(self):
        w = self._make_wallet()
        w.asset = 0.000001  # Will round to 0 with step_size 0.001
        with patch.object(w, "refresh_balances"), \
             patch.object(w, "_get_symbol_filters", return_value={
                 "step_size": 0.001
             }):
            result = w.sell(50000.0, "test")
        assert result is None

    def test_execute_order_logs_details(self):
        w = self._make_wallet()
        mock_resp = {"orderId": 99, "status": "FILLED", "executedQty": "0.001"}
        with patch.object(w, "signed_request", return_value=mock_resp):
            result = w.execute_order("BUY", "BTCUSDT", 0.001)
        assert result["orderId"] == 99

    def test_verify_order_status(self):
        w = self._make_wallet()
        mock_resp = {"orderId": 99, "status": "FILLED"}
        with patch.object(w, "signed_request", return_value=mock_resp):
            result = w._verify_order_status("BTCUSDT", 99)
        assert result["status"] == "FILLED"

    def test_total_value(self):
        w = self._make_wallet()
        w.usdt = 50.0
        w.asset = 0.001
        with patch.object(w, "refresh_balances"):
            val = w.total_value(50000.0)
        assert abs(val - 100.0) < 0.01
