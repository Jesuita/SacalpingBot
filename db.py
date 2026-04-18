"""Modulo de persistencia SQLite para el bot de scalping.

Tablas:
 - trades: historial de trades ejecutados
 - ml_features: features por decision para entrenamiento IA
 - events: log de eventos del bot
 - optimizer_actions: acciones del optimizador

Compatibilidad: mantiene trades.log como backup.
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_data.db")

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Retorna conexion por thread (thread-safe)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_FILE, timeout=10)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


@contextmanager
def get_db():
    """Context manager para obtener conexion."""
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Crea tablas si no existen."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL DEFAULT 0,
                pnl REAL NOT NULL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                reason TEXT DEFAULT '',
                balance_after REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);

            CREATE TABLE IF NOT EXISTS ml_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                ema_fast REAL,
                ema_slow REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                signal_rule TEXT,
                in_position INTEGER DEFAULT 0,
                balance REAL DEFAULT 0,
                ai_score REAL,
                regime TEXT,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_ml_symbol ON ml_features(symbol);
            CREATE INDEX IF NOT EXISTS idx_ml_timestamp ON ml_features(timestamp);

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT DEFAULT 'INFO',
                category TEXT DEFAULT 'general',
                message TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);

            CREATE TABLE IF NOT EXISTS optimizer_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                params TEXT DEFAULT '{}',
                result TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_optimizer_timestamp ON optimizer_actions(timestamp);
        """)


# ─── Trades ───

def insert_trade(symbol: str, side: str, price: float, qty: float = 0,
                 pnl: float = 0, pnl_pct: float = 0, reason: str = "",
                 balance_after: float = 0, metadata: dict = None,
                 timestamp: str = None):
    """Inserta un trade en la DB."""
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    meta = json.dumps(metadata or {})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, price, qty, pnl, pnl_pct, reason, balance_after, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, symbol, side, price, qty, pnl, pnl_pct, reason, balance_after, meta),
        )


def get_trades(symbol: str = None, limit: int = 500, offset: int = 0) -> List[Dict]:
    """Obtiene trades, opcionalmente filtrados por symbol."""
    with get_db() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (symbol, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows]


def get_trades_summary() -> Dict[str, Any]:
    """Resumen global de trades para el dashboard."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN side='SELL' AND pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN side='SELL' AND pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN side='SELL' THEN pnl ELSE 0 END) as total_pnl,
                AVG(CASE WHEN side='SELL' AND pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN side='SELL' AND pnl <= 0 THEN pnl END) as avg_loss
            FROM trades
        """).fetchone()
    d = dict(row)
    total_sells = (d["wins"] or 0) + (d["losses"] or 0)
    d["win_rate"] = (d["wins"] or 0) / max(total_sells, 1) * 100
    d["profit_factor"] = abs((d["avg_win"] or 0) * (d["wins"] or 0)) / max(abs((d["avg_loss"] or 0) * (d["losses"] or 0)), 0.01)
    return d


def get_trades_by_day(days: int = 30) -> List[Dict]:
    """PnL agrupado por dia."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT DATE(timestamp) as day,
                   COUNT(*) as trades,
                   SUM(CASE WHEN side='SELL' THEN pnl ELSE 0 END) as pnl
            FROM trades
            WHERE timestamp >= DATE('now', ?)
            GROUP BY DATE(timestamp)
            ORDER BY day
        """, (f"-{days} days",)).fetchall()
    return [dict(r) for r in rows]


# ─── ML Features ───

def insert_ml_feature(symbol: str, price: float, ema_fast: float = None,
                      ema_slow: float = None, rsi: float = None,
                      macd: float = None, macd_signal: float = None,
                      macd_hist: float = None, signal_rule: str = None,
                      in_position: bool = False, balance: float = 0,
                      ai_score: float = None, regime: str = None,
                      metadata: dict = None, timestamp: str = None):
    """Inserta un registro de features ML."""
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    meta = json.dumps(metadata or {})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO ml_features (timestamp, symbol, price, ema_fast, ema_slow, rsi, "
            "macd, macd_signal, macd_hist, signal_rule, in_position, balance, ai_score, regime, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, symbol, price, ema_fast, ema_slow, rsi, macd, macd_signal,
             macd_hist, signal_rule, int(in_position), balance, ai_score, regime, meta),
        )


def get_ml_features_count() -> int:
    """Cuenta total de registros ML."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM ml_features").fetchone()
    return row["cnt"]


def get_ml_features_summary() -> Dict:
    """Resumen del dataset ML."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT symbol) as symbols,
                   MIN(timestamp) as first_record,
                   MAX(timestamp) as last_record
            FROM ml_features
        """).fetchone()
    return dict(row)


# ─── Events ───

def insert_event(message: str, level: str = "INFO", category: str = "general",
                 metadata: dict = None, timestamp: str = None):
    """Inserta un evento."""
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    meta = json.dumps(metadata or {})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO events (timestamp, level, category, message, metadata) VALUES (?, ?, ?, ?, ?)",
            (ts, level, category, message, meta),
        )


def get_events(limit: int = 200, category: str = None, level: str = None) -> List[Dict]:
    """Obtiene eventos con filtros opcionales."""
    clauses = []
    params = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if level:
        clauses.append("level = ?")
        params.append(level)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM events{where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Optimizer ───

def insert_optimizer_action(action: str, params: dict = None, result: dict = None,
                            timestamp: str = None):
    """Registra accion del optimizador."""
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO optimizer_actions (timestamp, action, params, result) VALUES (?, ?, ?, ?)",
            (ts, action, json.dumps(params or {}), json.dumps(result or {})),
        )


# ─── Migration ───

def migrate_trades_log(log_path: str = "trades.log") -> int:
    """Migra trades desde trades.log a SQLite. Retorna cantidad importada."""
    if not os.path.exists(log_path):
        return 0
    imported = 0
    with get_db() as conn:
        existing = conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()["cnt"]
        if existing > 0:
            return 0  # Ya migrado

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            import csv
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    ts = row[0].strip()
                    symbol = row[1].strip()
                    side = row[2].strip().upper()
                    price = float(row[3])
                    pnl = float(row[4]) if len(row) > 4 else 0.0
                    reason = row[5].strip() if len(row) > 5 else ""
                    conn.execute(
                        "INSERT INTO trades (timestamp, symbol, side, price, pnl, reason) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (ts, symbol, side, price, pnl, reason),
                    )
                    imported += 1
                except (ValueError, IndexError):
                    continue
    return imported


# Inicializar DB al importar
init_db()
