"""Tests para el modulo de persistencia SQLite."""
import json
import os
import pytest

# Usar DB en memoria para tests
import db as db_mod

# Override para usar temp file
_test_db = os.path.join(os.path.dirname(__file__), "_test_bot_data.db")
db_mod.DB_FILE = _test_db


@pytest.fixture(autouse=True)
def clean_db():
    """Reset DB antes de cada test."""
    # Cerrar conexiones previas
    if hasattr(db_mod._local, "conn") and db_mod._local.conn:
        db_mod._local.conn.close()
        db_mod._local.conn = None
    if os.path.exists(_test_db):
        os.remove(_test_db)
    db_mod.init_db()
    yield
    if hasattr(db_mod._local, "conn") and db_mod._local.conn:
        db_mod._local.conn.close()
        db_mod._local.conn = None
    if os.path.exists(_test_db):
        os.remove(_test_db)


class TestInsertAndGetTrades:
    def test_insert_and_retrieve(self):
        db_mod.insert_trade("BTCUSDT", "BUY", 50000.0, qty=0.001, reason="entrada")
        trades = db_mod.get_trades()
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTCUSDT"
        assert trades[0]["price"] == 50000.0

    def test_filter_by_symbol(self):
        db_mod.insert_trade("BTCUSDT", "BUY", 50000.0)
        db_mod.insert_trade("ETHUSDT", "BUY", 3000.0)
        btc = db_mod.get_trades(symbol="BTCUSDT")
        assert len(btc) == 1
        assert btc[0]["symbol"] == "BTCUSDT"

    def test_pnl_tracking(self):
        db_mod.insert_trade("BTCUSDT", "BUY", 50000.0, qty=0.001)
        db_mod.insert_trade("BTCUSDT", "SELL", 51000.0, qty=0.001, pnl=1.0, reason="take_profit")
        trades = db_mod.get_trades()
        sells = [t for t in trades if t["side"] == "SELL"]
        assert sells[0]["pnl"] == 1.0


class TestTradesSummary:
    def test_empty_summary(self):
        s = db_mod.get_trades_summary()
        assert s["total"] == 0

    def test_summary_with_trades(self):
        db_mod.insert_trade("BTCUSDT", "BUY", 50000.0)
        db_mod.insert_trade("BTCUSDT", "SELL", 51000.0, pnl=1.0)
        db_mod.insert_trade("BTCUSDT", "SELL", 49000.0, pnl=-0.5)
        s = db_mod.get_trades_summary()
        assert s["total"] == 3
        assert s["wins"] == 1
        assert s["losses"] == 1


class TestMlFeatures:
    def test_insert_and_count(self):
        db_mod.insert_ml_feature("BTCUSDT", 50000.0, ema_fast=50100, ema_slow=49900, rsi=55)
        assert db_mod.get_ml_features_count() == 1

    def test_summary(self):
        db_mod.insert_ml_feature("BTCUSDT", 50000.0)
        db_mod.insert_ml_feature("ETHUSDT", 3000.0)
        s = db_mod.get_ml_features_summary()
        assert s["total"] == 2
        assert s["symbols"] == 2


class TestEvents:
    def test_insert_and_retrieve(self):
        db_mod.insert_event("Bot iniciado", level="INFO", category="system")
        events = db_mod.get_events()
        assert len(events) == 1
        assert events[0]["message"] == "Bot iniciado"

    def test_filter_by_category(self):
        db_mod.insert_event("Trade", category="trade")
        db_mod.insert_event("Error", category="error")
        trades = db_mod.get_events(category="trade")
        assert len(trades) == 1


class TestOptimizerActions:
    def test_insert(self):
        db_mod.insert_optimizer_action("adjust_tp", params={"tp": 0.01}, result={"ok": True})
        with db_mod.get_db() as conn:
            row = conn.execute("SELECT * FROM optimizer_actions").fetchone()
        assert dict(row)["action"] == "adjust_tp"


class TestMigration:
    def test_migrate_nonexistent_file(self):
        count = db_mod.migrate_trades_log("nonexistent.log")
        assert count == 0

    def test_migrate_trades_log(self, tmp_path):
        log_file = str(tmp_path / "trades.log")
        with open(log_file, "w") as f:
            f.write("2026-04-17 10:00:00,BTCUSDT,BUY,50000.0,0.0,entrada\n")
            f.write("2026-04-17 10:05:00,BTCUSDT,SELL,51000.0,1.0,take_profit\n")
        count = db_mod.migrate_trades_log(log_file)
        assert count == 2
        trades = db_mod.get_trades()
        assert len(trades) == 2
