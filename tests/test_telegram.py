"""Tests para telegram_notifier — resumenes y comandos."""

import time
import unittest
from unittest.mock import patch, MagicMock

import telegram_notifier


class FakeWallet:
    def __init__(self, usdt=100, in_pos=False, trades=None):
        self.usdt_balance = usdt
        self._in_position = in_pos
        self.trades = trades or []

    @property
    def in_position(self):
        return self._in_position


class FakeBot:
    def __init__(self, wallets=None):
        self.wallets = wallets or {}
        self.paused_symbols = set()


class TestSendMessage(unittest.TestCase):
    @patch("telegram_notifier._get_config", return_value={"enabled": False, "token": "", "chat_id": ""})
    def test_disabled_returns_false(self, _):
        self.assertFalse(telegram_notifier.send_message("hi"))

    @patch("telegram_notifier.requests.post")
    @patch("telegram_notifier._get_config", return_value={"enabled": True, "token": "T", "chat_id": "C"})
    def test_send_ok(self, _, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        telegram_notifier._last_sent.clear()
        self.assertTrue(telegram_notifier.send_message("test", msg_type="test_unique"))

    @patch("telegram_notifier.requests.post")
    @patch("telegram_notifier._get_config", return_value={"enabled": True, "token": "T", "chat_id": "C"})
    def test_rate_limit(self, _, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        telegram_notifier._last_sent.clear()
        telegram_notifier.send_message("a", msg_type="rl_test")
        result = telegram_notifier.send_message("b", msg_type="rl_test")
        self.assertFalse(result)


class TestDailySummary(unittest.TestCase):
    def test_no_bot_returns_none(self):
        telegram_notifier._bot_ref = None
        self.assertIsNone(telegram_notifier._build_daily_summary())

    def test_with_trades(self):
        trades = [{"timestamp": time.time(), "pnl": 0.5}, {"timestamp": time.time(), "pnl": -0.1}]
        bot = FakeBot({"BTCUSDT": FakeWallet(trades=trades)})
        telegram_notifier._bot_ref = bot
        msg = telegram_notifier._build_daily_summary()
        self.assertIn("RESUMEN DIARIO", msg)
        self.assertIn("BTCUSDT", msg)
        self.assertIn("2 trades", msg)

    def test_empty_wallets(self):
        telegram_notifier._bot_ref = FakeBot({})
        self.assertIsNone(telegram_notifier._build_daily_summary())


class TestWeeklySummary(unittest.TestCase):
    def test_with_trades(self):
        trades = [{"timestamp": time.time(), "pnl": 1.0}]
        bot = FakeBot({"ETHUSDT": FakeWallet(trades=trades)})
        telegram_notifier._bot_ref = bot
        msg = telegram_notifier._build_weekly_summary()
        self.assertIn("RESUMEN SEMANAL", msg)
        self.assertIn("ETHUSDT", msg)


class TestCommandHandlers(unittest.TestCase):
    def setUp(self):
        self.bot = FakeBot({
            "BTCUSDT": FakeWallet(usdt=95.5, in_pos=True),
            "ETHUSDT": FakeWallet(usdt=50.0),
        })
        telegram_notifier._bot_ref = self.bot

    def test_status_handler(self):
        result = telegram_notifier._default_status_handler()
        self.assertIn("BTCUSDT", result)
        self.assertIn("EN POSICION", result)

    def test_balance_handler(self):
        result = telegram_notifier._default_balance_handler()
        self.assertIn("95.50", result)
        self.assertIn("Total", result)

    def test_pause_handler(self):
        result = telegram_notifier._default_pause_handler("btcusdt")
        self.assertIn("pausado", result)
        self.assertIn("BTCUSDT", self.bot.paused_symbols)

    def test_resume_handler(self):
        self.bot.paused_symbols.add("BTCUSDT")
        result = telegram_notifier._default_resume_handler("BTCUSDT")
        self.assertIn("reanudado", result)
        self.assertNotIn("BTCUSDT", self.bot.paused_symbols)

    def test_pause_no_args(self):
        result = telegram_notifier._default_pause_handler("")
        self.assertIn("Uso:", result)

    def test_register_command(self):
        telegram_notifier.register_command("/test", lambda: "ok")
        self.assertIn("test", telegram_notifier._command_handlers)

    def test_no_bot_status(self):
        telegram_notifier._bot_ref = None
        result = telegram_notifier._default_status_handler()
        self.assertEqual(result, "Bot no disponible.")


if __name__ == "__main__":
    unittest.main()
