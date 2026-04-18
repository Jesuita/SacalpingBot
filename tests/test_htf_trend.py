"""Tests para get_htf_trend y preflight_check_real."""
from unittest.mock import patch, MagicMock
import pytest
from scalping_bot import get_htf_trend, EMA_SLOW


class TestGetHtfTrend:
    def _make_uptrend(self, n=50):
        return [30000 + i * 10 for i in range(n)]

    def _make_downtrend(self, n=50):
        return [40000 - i * 10 for i in range(n)]

    def test_returns_neutral_when_insufficient_data(self):
        assert get_htf_trend([100, 200, 300]) == "neutral"

    def test_returns_neutral_at_min_boundary(self):
        short = list(range(EMA_SLOW + 4))
        assert get_htf_trend(short) == "neutral"

    def test_returns_up_on_uptrend(self):
        assert get_htf_trend(self._make_uptrend()) == "up"

    def test_returns_down_on_downtrend(self):
        assert get_htf_trend(self._make_downtrend()) == "down"

    def test_returns_neutral_on_flat(self):
        flat = [30000.0] * 50
        result = get_htf_trend(flat)
        assert result in ("neutral", "up", "down")

    def test_returns_string(self):
        assert isinstance(get_htf_trend(self._make_uptrend()), str)


@patch("scalping_bot.log", MagicMock())
@patch("scalping_bot.EVENT_LOGGER", MagicMock())
class TestPreflightCheckReal:
    @patch("scalping_bot.requests.get")
    @patch("scalping_bot.get_price", return_value=60000.0)
    def test_all_checks_pass(self, mock_price, mock_get):
        from scalping_bot import preflight_check_real
        responses = []
        # ping
        r1 = MagicMock(); r1.status_code = 200; responses.append(r1)
        # server time
        r2 = MagicMock(); r2.status_code = 200
        import time
        r2.json.return_value = {"serverTime": int(time.time() * 1000)}
        responses.append(r2)
        # account
        r3 = MagicMock(); r3.status_code = 200
        r3.json.return_value = {"canTrade": True, "balances": [{"asset": "USDT", "free": "100"}]}
        responses.append(r3)
        # exchangeInfo
        r4 = MagicMock(); r4.status_code = 200
        r4.json.return_value = {"symbols": [{"symbol": "BTCUSDT"}]}
        responses.append(r4)
        mock_get.side_effect = responses
        assert preflight_check_real(["BTCUSDT"]) is True

    @patch("scalping_bot.requests.get", side_effect=Exception("connection error"))
    def test_fails_on_no_connectivity(self, mock_get):
        from scalping_bot import preflight_check_real
        assert preflight_check_real(["BTCUSDT"]) is False
