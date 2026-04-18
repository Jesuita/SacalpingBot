"""Tests para backtest.py."""
from backtest import run_backtest, walk_forward, BacktestResult


def _make_candles(n=100, start_price=50000, trend=0.5):
    """Genera velas sintéticas con tendencia."""
    candles = []
    price = start_price
    for i in range(n):
        o = price
        h = price + abs(trend) * 2
        l = price - abs(trend)
        price += trend + (i % 3 - 1) * 0.3
        c = price
        candles.append({
            "timestamp": 1700000000000 + i * 60000,
            "open": o, "high": max(o, h, c), "low": min(o, l, c),
            "close": c, "volume": 100.0,
        })
    return candles


class TestRunBacktest:
    def test_returns_result(self):
        candles = _make_candles(200)
        r = run_backtest(candles)
        assert isinstance(r, BacktestResult)

    def test_no_trades_on_short_data(self):
        candles = _make_candles(10)
        r = run_backtest(candles)
        assert r.total_trades == 0

    def test_balance_positive(self):
        candles = _make_candles(200)
        r = run_backtest(candles)
        assert r.final_balance > 0

    def test_metrics_consistent(self):
        candles = _make_candles(500)
        r = run_backtest(candles)
        assert r.wins + r.losses == r.total_trades
        if r.total_trades > 0:
            assert 0 <= r.win_rate <= 100
            assert r.max_drawdown_pct >= 0

    def test_pnl_matches_balance(self):
        candles = _make_candles(300)
        r = run_backtest(candles, initial_balance=100.0)
        assert abs(r.total_pnl - (r.final_balance - 100.0)) < 0.01

    def test_custom_params(self):
        candles = _make_candles(200)
        r = run_backtest(candles, trailing_stop_pct=0.01, take_profit_pct=0.02)
        assert isinstance(r, BacktestResult)


class TestWalkForward:
    def test_returns_list(self):
        candles = _make_candles(600)
        results = walk_forward(candles, window_size=200, step=100)
        assert isinstance(results, list)
        assert len(results) >= 2

    def test_each_is_result(self):
        candles = _make_candles(600)
        results = walk_forward(candles, window_size=200, step=100)
        for r in results:
            assert isinstance(r, BacktestResult)

    def test_short_data_empty(self):
        candles = _make_candles(30)
        results = walk_forward(candles, window_size=500, step=250)
        assert len(results) == 0
