"""Tests de indicadores técnicos: EMA, RSI, MACD."""
from scalping_bot import calc_ema, calc_rsi, calc_macd


# ─── EMA ────────────────────────────────────────────

class TestEMA:
    def test_empty_list(self):
        assert calc_ema([], 9) == 0.0

    def test_single_value(self):
        assert calc_ema([100.0], 9) == 100.0

    def test_fewer_than_period(self):
        prices = [100.0, 102.0, 101.0]
        result = calc_ema(prices, 9)
        # Con menos de period puntos, devuelve SMA
        assert abs(result - sum(prices) / len(prices)) < 1e-9

    def test_exact_period(self):
        prices = [float(i) for i in range(1, 10)]  # 1..9
        result = calc_ema(prices, 9)
        assert result > 0

    def test_constant_prices(self):
        prices = [50.0] * 30
        result = calc_ema(prices, 9)
        assert abs(result - 50.0) < 1e-6

    def test_rising_prices(self):
        prices = [float(i) for i in range(1, 51)]  # 1..50
        ema9 = calc_ema(prices, 9)
        ema21 = calc_ema(prices, 21)
        # En tendencia alcista, EMA rápida > EMA lenta
        assert ema9 > ema21

    def test_fast_reacts_more(self):
        prices = [100.0] * 30 + [200.0]  # salto repentino
        ema9 = calc_ema(prices, 9)
        ema21 = calc_ema(prices, 21)
        assert ema9 > ema21  # EMA9 reacciona más al salto


# ─── RSI ────────────────────────────────────────────

class TestRSI:
    def test_insufficient_data(self):
        assert calc_rsi([100.0, 101.0], 14) == 50.0

    def test_all_gains(self):
        prices = [float(i) for i in range(100, 120)]  # solo subidas
        rsi = calc_rsi(prices, 14)
        assert rsi == 100.0

    def test_all_losses(self):
        prices = [float(i) for i in range(120, 100, -1)]  # solo bajadas
        rsi = calc_rsi(prices, 14)
        assert rsi == 0.0

    def test_constant_prices(self):
        prices = [100.0] * 20
        rsi = calc_rsi(prices, 14)
        # Sin pérdidas → avg_loss=0 → RSI=100
        assert rsi == 100.0

    def test_mixed_movement(self):
        prices = [100 + (i % 3) for i in range(30)]
        rsi = calc_rsi(prices, 14)
        assert 0 <= rsi <= 100

    def test_oversold_region(self):
        # Precios cayendo fuerte
        prices = [100.0 - i * 2 for i in range(20)]
        rsi = calc_rsi(prices, 14)
        assert rsi < 30

    def test_overbought_region(self):
        # Precios subiendo fuerte
        prices = [100.0 + i * 2 for i in range(20)]
        rsi = calc_rsi(prices, 14)
        assert rsi > 70


# ─── MACD ────────────────────────────────────────────

class TestMACD:
    def test_insufficient_data(self):
        macd, signal, hist = calc_macd([100.0] * 10)
        assert macd == 0.0 and signal == 0.0 and hist == 0.0

    def test_constant_prices(self):
        prices = [100.0] * 50
        macd, signal, hist = calc_macd(prices)
        assert abs(macd) < 0.01
        assert abs(signal) < 0.01
        assert abs(hist) < 0.01

    def test_rising_trend(self):
        prices = [100.0 + i * 0.5 for i in range(50)]
        macd, signal, hist = calc_macd(prices)
        assert macd > 0  # EMA rápida > EMA lenta en tendencia alcista

    def test_falling_trend(self):
        prices = [100.0 - i * 0.5 for i in range(50)]
        macd, signal, hist = calc_macd(prices)
        assert macd < 0

    def test_returns_three_values(self):
        prices = [100.0 + i for i in range(50)]
        result = calc_macd(prices)
        assert len(result) == 3

    def test_histogram_sign(self):
        # Subida fuerte: MACD debe estar por encima de signal → hist positivo
        prices = [100.0] * 30 + [100.0 + i * 3 for i in range(20)]
        _, _, hist = calc_macd(prices)
        assert hist > 0
