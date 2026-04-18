"""Tests de PaperWallet."""
from unittest.mock import patch, MagicMock
from scalping_bot import PaperWallet


def _make_config(trade_pct=0.95, trailing_stop_pct=0.006, fee_pct=0.0):
    return lambda: {"trade_pct": trade_pct, "trailing_stop_pct": trailing_stop_pct, "fee_pct": fee_pct}


@patch("scalping_bot.log", MagicMock())
@patch("scalping_bot.log_trade", MagicMock())
class TestPaperWalletBuy:
    def test_buy_reduces_usdt(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        w.buy(50000.0)
        assert w.usdt < 100.0
        assert w.asset > 0

    def test_buy_amount_matches_trade_pct(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trade_pct=0.5))
        w.buy(50000.0)
        # Gastó 50% → $50 en BTC
        assert abs(w.usdt - 50.0) < 0.01
        assert abs(w.asset - 50.0 / 50000.0) < 1e-10

    def test_buy_sets_entry_price(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        w.buy(42000.0)
        assert w.entry_price == 42000.0

    def test_buy_records_trade(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        trade = w.buy(50000.0)
        assert trade is not None
        assert trade["type"] == "BUY"
        assert len(w.trades) == 1

    def test_buy_with_zero_balance(self):
        w = PaperWallet("BTCUSDT", 0.0, _make_config())
        result = w.buy(50000.0)
        assert result is None

    def test_in_position_after_buy(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        assert not w.in_position
        w.buy(50000.0)
        assert w.in_position


@patch("scalping_bot.log", MagicMock())
@patch("scalping_bot.log_trade", MagicMock())
class TestPaperWalletSell:
    def test_sell_returns_usdt(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        w.buy(50000.0)
        w.sell(50000.0, "test")
        assert w.usdt > 0
        assert w.asset == 0.0

    def test_sell_with_profit(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trade_pct=1.0))
        w.buy(50000.0)
        trade = w.sell(51000.0, "tp")
        assert trade["pnl"] > 0

    def test_sell_with_loss(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trade_pct=1.0))
        w.buy(50000.0)
        trade = w.sell(49000.0, "sl")
        assert trade["pnl"] < 0

    def test_sell_without_position(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        result = w.sell(50000.0, "test")
        assert result is None

    def test_double_sell(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        w.buy(50000.0)
        w.sell(50000.0, "test")
        result = w.sell(50000.0, "test2")
        assert result is None

    def test_not_in_position_after_sell(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        w.buy(50000.0)
        w.sell(50000.0, "test")
        assert not w.in_position

    def test_sell_clears_entry_price(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        w.buy(50000.0)
        w.sell(51000.0, "test")
        assert w.entry_price is None
        assert w.peak_price is None


@patch("scalping_bot.log", MagicMock())
@patch("scalping_bot.log_trade", MagicMock())
class TestPaperWalletTrailing:
    def test_trailing_updates_on_new_high(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trailing_stop_pct=0.01))
        w.buy(50000.0)
        w.update_trailing_stop(51000.0)
        assert w.peak_price == 51000.0
        assert w.trailing_stop_price is not None
        assert w.trailing_stop_price < 51000.0

    def test_trailing_no_update_on_lower_price(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trailing_stop_pct=0.01))
        w.buy(50000.0)
        w.update_trailing_stop(51000.0)
        stop1 = w.trailing_stop_price
        w.update_trailing_stop(50500.0)
        # Peak y stop no deben bajar
        assert w.peak_price == 51000.0
        assert w.trailing_stop_price == stop1

    def test_trailing_stop_value(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trailing_stop_pct=0.01))
        w.buy(50000.0)
        # Precio debe superar peak_price (entry_price) para generar trailing
        w.update_trailing_stop(50100.0)
        # Stop = 50100 * (1 - 0.01) = 49599
        assert w.trailing_stop_price is not None
        assert abs(w.trailing_stop_price - 50100.0 * 0.99) < 0.01

    def test_no_trailing_without_position(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        w.update_trailing_stop(50000.0)
        assert w.trailing_stop_price is None


@patch("scalping_bot.log", MagicMock())
@patch("scalping_bot.log_trade", MagicMock())
class TestPaperWalletTotalValue:
    def test_initial_value(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config())
        assert w.total_value(50000.0) == 100.0

    def test_value_in_position(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trade_pct=1.0))
        w.buy(50000.0)
        # Precio sube 10%
        val = w.total_value(55000.0)
        assert val > 100.0

    def test_round_trip_preserves_value(self):
        w = PaperWallet("BTCUSDT", 100.0, _make_config(trade_pct=1.0))
        w.buy(50000.0)
        w.sell(50000.0, "test")
        assert abs(w.total_value(50000.0) - 100.0) < 0.01
