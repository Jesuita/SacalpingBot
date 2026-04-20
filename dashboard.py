import asyncio
import csv
import datetime
import json
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import requests
import uvicorn

STATE_FILE = "bot_state.json"
CONFIG_FILE = "runtime_config.json"
TRADES_LOG_FILE = "trades.log"
BOT_EVENTS_FILE = "bot_events.log"
ML_DATASET_FILE = "ml_dataset.csv"
MULTI_SOURCE_CONFIG_FILE = "multi_source_config.json"
AI_MODEL_FILE = "ai_model.json"
ENV_FILE = ".env"
PERIODIC_ANALYSIS_FILE = "periodic_analysis.json"

# Market Scanner
try:
    import market_scanner
    market_scanner.start_scanner()
except ImportError:
    market_scanner = None

# Auto Optimizer
try:
    import auto_optimizer
    auto_optimizer.start_optimizer()
except ImportError:
    auto_optimizer = None

# Intelligence Engine
try:
    import intelligence_engine
    intelligence_engine.start_intelligence()
except ImportError:
    intelligence_engine = None

# ML Training
try:
    import train_model
except ImportError:
    train_model = None

# Backtesting
try:
    import backtest as backtest_mod
except ImportError:
    backtest_mod = None

# Database
try:
    import db as db_mod
except ImportError:
    db_mod = None

# Portfolio
try:
    import portfolio as portfolio_mod
except ImportError:
    portfolio_mod = None

# RL Agent
try:
    import rl_agent as rl_mod
except ImportError:
    rl_mod = None

from config_defaults import (
    DEFAULT_RUNTIME_CONFIG as DEFAULT_CONFIG,
    DEFAULT_MULTI_SOURCE_CONFIG,
    normalize_runtime_config as normalize_config,
)

app = FastAPI(title="Scalping Bot Dashboard")

# Static files for PWA
from fastapi.staticfiles import StaticFiles
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def _load_html_page() -> str:
    path = os.path.join(_TEMPLATE_DIR, "index.html")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()

HTML_PAGE = _load_html_page()



def default_state() -> Dict[str, Any]:
    return {
        "updated_at": "",
        "mode": "paper",
        "pairs": {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "price": 0,
                "signal": "HOLD",
                "rsi": 50,
                "in_position": False,
                "balance": 0,
                "initial_balance": 0,
                "gain_pct": 0,
                "candles": [],
                "ema_fast": [],
                "ema_slow": [],
                "rsi_series": [],
                "trades": [],
                "event": "idle",
                "updated_at": "",
            },
            "ETHUSDT": {
                "symbol": "ETHUSDT",
                "price": 0,
                "signal": "HOLD",
                "rsi": 50,
                "in_position": False,
                "balance": 0,
                "initial_balance": 0,
                "gain_pct": 0,
                "candles": [],
                "ema_fast": [],
                "ema_slow": [],
                "rsi_series": [],
                "trades": [],
                "event": "idle",
                "updated_at": "",
            },
        },
    }


def load_state() -> Dict[str, Any]:
  base = default_state()
  if not os.path.exists(STATE_FILE):
    return base

  try:
    with open(STATE_FILE, "r", encoding="utf-8") as fh:
      data = json.load(fh)
      if "pairs" not in data or not data.get("pairs"):
        data["pairs"] = base["pairs"]
      return data
  except Exception:
    return base


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            return normalize_config(json.load(fh))
    except Exception:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()


def save_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_config(payload)
    with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    return cfg


def normalize_multi_source_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = DEFAULT_MULTI_SOURCE_CONFIG.copy()
    cfg.update(payload or {})
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["threshold_pct"] = min(max(float(cfg.get("threshold_pct", 0.15)), 0.01), 5.0)
    action = str(cfg.get("action", "alert")).strip().lower()
    cfg["action"] = action if action in {"alert", "reduce", "pause"} else "alert"
    cfg["refresh_seconds"] = min(max(int(float(cfg.get("refresh_seconds", 5))), 2), 30)
    return cfg


def load_multi_source_config() -> Dict[str, Any]:
    if not os.path.exists(MULTI_SOURCE_CONFIG_FILE):
        save_multi_source_config(DEFAULT_MULTI_SOURCE_CONFIG)
        return DEFAULT_MULTI_SOURCE_CONFIG.copy()
    try:
        with open(MULTI_SOURCE_CONFIG_FILE, "r", encoding="utf-8") as fh:
            return normalize_multi_source_config(json.load(fh))
    except Exception:
        save_multi_source_config(DEFAULT_MULTI_SOURCE_CONFIG)
        return DEFAULT_MULTI_SOURCE_CONFIG.copy()


def save_multi_source_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_multi_source_config(payload)
    with open(MULTI_SOURCE_CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    return cfg


def _fetch_price_binance(symbol: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    url = "https://api.binance.com/api/v3/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=3)
    resp.raise_for_status()
    latency = (time.perf_counter() - t0) * 1000
    return {"name": "Binance", "price": float(resp.json()["price"]), "latency_ms": latency, "ok": True}


def _fetch_price_coinbase(symbol: str) -> Dict[str, Any]:
    base = "BTC" if symbol.startswith("BTC") else "ETH"
    product = f"{base}-USD"
    t0 = time.perf_counter()
    url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()
    latency = (time.perf_counter() - t0) * 1000
    return {"name": "Coinbase", "price": float(resp.json()["price"]), "latency_ms": latency, "ok": True}


def _fetch_price_kraken(symbol: str) -> Dict[str, Any]:
    pair = "XXBTZUSD" if symbol.startswith("BTC") else "XETHZUSD"
    t0 = time.perf_counter()
    url = "https://api.kraken.com/0/public/Ticker"
    resp = requests.get(url, params={"pair": pair}, timeout=3)
    resp.raise_for_status()
    payload = resp.json()
    result = payload.get("result", {})
    key = next(iter(result.keys()))
    last = float(result[key]["c"][0])
    latency = (time.perf_counter() - t0) * 1000
    return {"name": "Kraken", "price": last, "latency_ms": latency, "ok": True}


def get_multi_source_snapshot(symbol: str) -> Dict[str, Any]:
    cfg = load_multi_source_config()
    sources = []
    fetchers = [_fetch_price_binance]
    if symbol.startswith("BTC") or symbol.startswith("ETH"):
        fetchers.extend([_fetch_price_coinbase, _fetch_price_kraken])

    for fn in fetchers:
        try:
            sources.append(fn(symbol))
        except Exception:
            name = fn.__name__.replace("_fetch_price_", "").capitalize()
            sources.append({"name": name, "price": None, "latency_ms": None, "ok": False})

    binance_price = None
    for src in sources:
        if src["name"] == "Binance" and src.get("price"):
            binance_price = src["price"]

    valid_prices = [s["price"] for s in sources if s.get("price")]
    reference_price = sum(valid_prices) / len(valid_prices) if valid_prices else 0.0
    max_dev = 0.0

    for src in sources:
        if src.get("price") is None or not binance_price:
            src["deviation_pct"] = None
            continue
        dev = abs(src["price"] - binance_price) / max(binance_price, 1e-9) * 100
        src["deviation_pct"] = dev
        max_dev = max(max_dev, dev)

    up_count = len([s for s in sources if s.get("ok")])
    quality = 100.0
    expected_sources = max(len(fetchers), 1)
    quality -= (expected_sources - up_count) * 25
    quality -= min(max_dev, 2.0) * 15
    quality = max(0.0, min(100.0, quality))

    recommendation = "ok"
    if cfg["enabled"] and max_dev >= cfg["threshold_pct"]:
        recommendation = cfg["action"]

    return {
        "symbol": symbol,
        "updated_at": time.strftime("%H:%M:%S"),
        "reference_price": reference_price,
        "max_deviation_pct": max_dev,
        "quality_score": quality,
        "recommendation": recommendation,
        "sources": sources,
        "config": cfg,
    }


def parse_trade_log_line(line: str) -> Dict[str, Any]:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 7:
        return {}

    row: Dict[str, Any] = {
        "date": parts[0],
        "time": parts[1],
    }

    for item in parts[2:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        row[key.strip()] = value.strip()

    try:
        row["price"] = float(row.get("precio", 0.0))
    except ValueError:
        row["price"] = 0.0

    try:
        row["pnl"] = float(row.get("pnl", 0.0))
    except ValueError:
        row["pnl"] = 0.0

    row["symbol"] = row.get("symbol", "")
    row["type"] = row.get("tipo", "")
    row["reason"] = row.get("razon", "")
    try:
        row["dt"] = datetime.datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        row["dt"] = None
    return row


def filter_trade_rows(
  rows: list,
  symbol: str = "",
  from_date: str = "",
  to_date: str = "",
  lookback_hours: int = 0,
) -> list:
    filtered = rows
    if symbol:
        wanted = symbol.strip().upper()
        filtered = [r for r in filtered if str(r.get("symbol", "")).upper() == wanted]

    date_from_obj = None
    date_to_obj = None
    try:
        if from_date:
            date_from_obj = datetime.datetime.strptime(from_date, "%Y-%m-%d").date()
    except Exception:
        date_from_obj = None
    try:
        if to_date:
            date_to_obj = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
    except Exception:
        date_to_obj = None

    if date_from_obj or date_to_obj:
        tmp = []
        for row in filtered:
            dt = row.get("dt")
            if not dt:
                continue
            d = dt.date()
            if date_from_obj and d < date_from_obj:
                continue
            if date_to_obj and d > date_to_obj:
                continue
            tmp.append(row)
        filtered = tmp

        if lookback_hours and lookback_hours > 0:
          cutoff = datetime.datetime.now() - datetime.timedelta(hours=int(lookback_hours))
          filtered = [r for r in filtered if r.get("dt") and r.get("dt") >= cutoff]
    return filtered


def pair_closed_trades(rows: list) -> list:
    queue_by_symbol: Dict[str, list] = {}
    pairs = []

    ordered = sorted(rows, key=lambda r: r.get("dt") or datetime.datetime.min)
    for row in ordered:
        symbol = row.get("symbol", "")
        side = row.get("type", "")
        dt = row.get("dt")
        if not symbol or not dt:
            continue
        if side == "BUY":
            queue_by_symbol.setdefault(symbol, []).append(row)
            continue
        if side != "SELL":
            continue

        queue = queue_by_symbol.get(symbol, [])
        if not queue:
            continue
        buy = queue.pop(0)
        dur_sec = int(max(0, (dt - buy.get("dt")).total_seconds())) if buy.get("dt") else 0
        pairs.append(
            {
                "symbol": symbol,
                "buy_date": buy.get("date", ""),
                "buy_time": buy.get("time", ""),
                "sell_date": row.get("date", ""),
                "sell_time": row.get("time", ""),
                "duration_sec": dur_sec,
                "pnl": float(row.get("pnl", 0.0)),
                "reason": row.get("reason", ""),
            }
        )

    return pairs


def build_trades_insights(rows: list, limit_rows: int = 200) -> Dict[str, Any]:
    base = empty_analysis()
    if not rows:
        return {
            "summary": {
                **base["global"],
                "buy_count": 0,
                "sell_count": 0,
                "open_positions": 0,
                "profit_factor": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
            },
            "durations": {
                "paired_trades": 0,
                "avg_sec_all": 0.0,
                "avg_sec_win": 0.0,
                "avg_sec_loss": 0.0,
                "min_sec": 0,
                "max_sec": 0,
            },
            "by_symbol": {},
            "by_day": [],
            "by_hour": [],
            "by_reason": [],
            "top_wins": [],
            "top_losses": [],
            "rows": [],
        }

    sells = [r for r in rows if r.get("type") == "SELL"]
    buys = [r for r in rows if r.get("type") == "BUY"]
    sell_pnls = [float(r.get("pnl", 0.0)) for r in sells]
    gross_profit = sum([p for p in sell_pnls if p > 0])
    gross_loss = sum([p for p in sell_pnls if p < 0])
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (gross_profit if gross_profit > 0 else 0.0)

    summary = compute_stats(rows)
    summary.update(
        {
            "buy_count": len(buys),
            "sell_count": len(sells),
            "open_positions": max(0, len(buys) - len(sells)),
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }
    )

    by_symbol_rows: Dict[str, list] = {}
    for row in rows:
      sym = row.get("symbol", "")
      if not sym:
        continue
      by_symbol_rows.setdefault(sym, []).append(row)
    by_symbol = {}
    for sym, items in by_symbol_rows.items():
        stats = compute_stats(items)
        sym_sells = [r for r in items if r.get("type") == "SELL"]
        sym_pnls = [float(r.get("pnl", 0.0)) for r in sym_sells]
        sym_gp = sum([p for p in sym_pnls if p > 0])
        sym_gl = sum([p for p in sym_pnls if p < 0])
        stats.update(
            {
                "buy_count": len([r for r in items if r.get("type") == "BUY"]),
                "sell_count": len(sym_sells),
                "profit_factor": sym_gp / abs(sym_gl) if sym_gl < 0 else (sym_gp if sym_gp > 0 else 0.0),
            }
        )
        by_symbol[sym] = stats

    by_day_map: Dict[str, list] = {}
    for row in rows:
      d = row.get("date", "")
      if d:
        by_day_map.setdefault(d, []).append(row)
    by_day = []
    for d, items in sorted(by_day_map.items(), key=lambda kv: kv[0]):
        st = compute_stats(items)
        by_day.append(
            {
                "date": d,
                "sell_count": st["closed_trades"],
                "pnl_total": st["pnl_total"],
                "win_rate": st["win_rate"],
            }
        )

    by_hour_map: Dict[str, list] = {}
    for row in sells:
      dt = row.get("dt")
      if not dt:
        continue
      h = dt.strftime("%H")
      by_hour_map.setdefault(h, []).append(row)
    by_hour = []
    for h, items in sorted(by_hour_map.items(), key=lambda kv: kv[0]):
        pnls = [float(r.get("pnl", 0.0)) for r in items]
        by_hour.append(
            {
                "hour": h,
                "sell_count": len(items),
                "pnl_total": sum(pnls),
                "wins": len([p for p in pnls if p > 0]),
                "losses": len([p for p in pnls if p < 0]),
            }
        )

    by_reason_map: Dict[str, Dict[str, Any]] = {}
    for row in sells:
        reason = row.get("reason", "") or "unknown"
        bucket = by_reason_map.setdefault(reason, {"reason": reason, "sell_count": 0, "pnl_total": 0.0})
        bucket["sell_count"] += 1
        bucket["pnl_total"] += float(row.get("pnl", 0.0))
    by_reason = sorted(by_reason_map.values(), key=lambda r: abs(float(r.get("pnl_total", 0.0))), reverse=True)

    top_wins = sorted(sells, key=lambda r: float(r.get("pnl", 0.0)), reverse=True)[:10]
    top_losses = sorted(sells, key=lambda r: float(r.get("pnl", 0.0)))[:10]

    pairs = pair_closed_trades(rows)
    pair_durations = [int(p.get("duration_sec", 0)) for p in pairs]
    win_durations = [int(p.get("duration_sec", 0)) for p in pairs if float(p.get("pnl", 0.0)) > 0]
    loss_durations = [int(p.get("duration_sec", 0)) for p in pairs if float(p.get("pnl", 0.0)) < 0]
    durations = {
        "paired_trades": len(pairs),
        "avg_sec_all": (sum(pair_durations) / len(pair_durations)) if pair_durations else 0.0,
        "avg_sec_win": (sum(win_durations) / len(win_durations)) if win_durations else 0.0,
        "avg_sec_loss": (sum(loss_durations) / len(loss_durations)) if loss_durations else 0.0,
        "min_sec": min(pair_durations) if pair_durations else 0,
        "max_sec": max(pair_durations) if pair_durations else 0,
    }

    slim = [
        {
            "date": r.get("date", ""),
            "time": r.get("time", ""),
            "symbol": r.get("symbol", ""),
            "type": r.get("type", ""),
            "price": float(r.get("price", 0.0)),
            "pnl": float(r.get("pnl", 0.0)),
            "reason": r.get("reason", ""),
        }
        for r in rows[-max(20, min(limit_rows, 2000)):]
    ]

    return {
        "summary": summary,
        "durations": durations,
        "by_symbol": by_symbol,
        "by_day": by_day,
        "by_hour": by_hour,
        "by_reason": by_reason,
        "top_wins": [
            {
                "date": r.get("date", ""),
                "time": r.get("time", ""),
                "symbol": r.get("symbol", ""),
                "pnl": float(r.get("pnl", 0.0)),
                "reason": r.get("reason", ""),
            }
            for r in top_wins
        ],
        "top_losses": [
            {
                "date": r.get("date", ""),
                "time": r.get("time", ""),
                "symbol": r.get("symbol", ""),
                "pnl": float(r.get("pnl", 0.0)),
                "reason": r.get("reason", ""),
            }
            for r in top_losses
        ],
        "rows": slim,
    }


def empty_analysis() -> Dict[str, Any]:
    return {
        "global": {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "avg_pnl": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        },
        "symbols": {},
        "last_entries": [],
    }


def compute_stats(rows: list) -> Dict[str, Any]:
    sells = [r for r in rows if r.get("type") == "SELL"]
    pnls = [float(r.get("pnl", 0.0)) for r in sells]
    closed = len(sells)
    wins = len([p for p in pnls if p > 0])
    losses = len([p for p in pnls if p < 0])
    win_values = [p for p in pnls if p > 0]
    loss_values = [abs(p) for p in pnls if p < 0]
    pnl_total = sum(pnls)
    avg = pnl_total / closed if closed else 0.0
    win_rate = (wins / closed * 100) if closed else 0.0
    avg_win = sum(win_values) / len(win_values) if win_values else 0.0
    avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
    p_win = wins / closed if closed else 0.0
    p_loss = losses / closed if closed else 0.0
    expectancy = p_win * avg_win - p_loss * avg_loss
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = peak - equity
        max_drawdown = max(max_drawdown, drawdown)
    best_trade = max(pnls) if pnls else 0.0
    worst_trade = min(pnls) if pnls else 0.0
    return {
        "closed_trades": closed,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl_total": pnl_total,
        "avg_pnl": avg,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }


def analyze_trades_log() -> Dict[str, Any]:
    if not os.path.exists(TRADES_LOG_FILE):
        return empty_analysis()

    rows = []
    try:
        with open(TRADES_LOG_FILE, "r", encoding="utf-8") as fh:
            for raw in fh:
                parsed = parse_trade_log_line(raw)
                if parsed:
                    rows.append(parsed)
    except Exception:
        return empty_analysis()

    analysis = empty_analysis()
    if not rows:
        return analysis

    analysis["global"] = compute_stats(rows)

    by_symbol: Dict[str, list] = {}
    for row in rows:
        sym = row.get("symbol", "")
        if not sym:
            continue
        by_symbol.setdefault(sym, []).append(row)

    analysis["symbols"] = {sym: compute_stats(items) for sym, items in by_symbol.items()}
    analysis["last_entries"] = [
      {
        "date": r.get("date", ""),
        "time": r.get("time", ""),
        "symbol": r.get("symbol", ""),
        "type": r.get("type", ""),
        "price": float(r.get("price", 0.0)),
        "pnl": float(r.get("pnl", 0.0)),
        "reason": r.get("reason", ""),
      }
      for r in rows[-20:]
    ]
    return analysis


def tail_events(limit: int = 120) -> Dict[str, Any]:
    limit = min(max(limit, 10), 500)
    if not os.path.exists(BOT_EVENTS_FILE):
        return {"lines": []}

    try:
        with open(BOT_EVENTS_FILE, "r", encoding="utf-8") as fh:
            lines = [line.rstrip("\n") for line in fh]
        return {"lines": lines[-limit:]}
    except Exception:
        return {"lines": []}


def append_ui_action_log(action: str, payload: Dict[str, Any]) -> bool:
    try:
        entry = {
            "ts": datetime.datetime.now().isoformat(),
            "action": str(action or "unknown"),
            "payload": payload or {},
        }
        line = f"[UI-ACTION] {json.dumps(entry, ensure_ascii=False)}"
        with open(BOT_EVENTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except Exception:
        return False


def get_ui_action_log(limit: int = 80) -> Dict[str, Any]:
    limit = min(max(int(limit), 10), 500)
    if not os.path.exists(BOT_EVENTS_FILE):
        return {"lines": []}

    try:
        with open(BOT_EVENTS_FILE, "r", encoding="utf-8") as fh:
            lines = [line.rstrip("\n") for line in fh if "[UI-ACTION]" in line]
        return {"lines": lines[-limit:]}
    except Exception:
        return {"lines": []}


def summarize_dataset() -> Dict[str, Any]:
    if not os.path.exists(ML_DATASET_FILE):
        return {
            "rows": 0,
            "by_symbol": {},
            "by_signal": {},
            "unlabeled": 0,
            "last_timestamp": "",
        }

    rows = 0
    unlabeled = 0
    by_symbol: Dict[str, int] = {}
    by_signal: Dict[str, int] = {}
    last_timestamp = ""

    try:
        with open(ML_DATASET_FILE, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows += 1
                symbol = (row.get("symbol") or "").strip()
                signal = (row.get("signal_regla") or "").strip().upper()
                label = (row.get("future_return_n") or "").strip()
                ts = (row.get("timestamp") or "").strip()

                if symbol:
                    by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
                if signal:
                    by_signal[signal] = by_signal.get(signal, 0) + 1
                if not label:
                    unlabeled += 1
                if ts:
                    last_timestamp = ts
    except Exception:
        return {
            "rows": 0,
            "by_symbol": {},
            "by_signal": {},
            "unlabeled": 0,
            "last_timestamp": "",
        }

    return {
        "rows": rows,
        "by_symbol": by_symbol,
        "by_signal": by_signal,
        "unlabeled": unlabeled,
        "last_timestamp": last_timestamp,
    }


def build_ai_log_learning(rows: list, lookback_hours: int = 24) -> Dict[str, Any]:
    sells = [r for r in rows if r.get("type") == "SELL"]
    closed = len(sells)
    pnls = [float(r.get("pnl", 0.0)) for r in sells]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / closed * 100) if closed else 0.0
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else (gross_profit if gross_profit > 0 else 0.0)
    expectancy = (sum(pnls) / closed) if closed else 0.0

    reason_stats: Dict[str, Dict[str, float]] = {}
    for row in sells:
      reason = row.get("reason", "unknown") or "unknown"
      bucket = reason_stats.setdefault(reason, {"count": 0.0, "pnl": 0.0})
      bucket["count"] += 1
      bucket["pnl"] += float(row.get("pnl", 0.0))

    top_win_reason = "-"
    top_loss_reason = "-"
    if reason_stats:
      ordered = sorted(reason_stats.items(), key=lambda kv: kv[1].get("pnl", 0.0), reverse=True)
      top_win_reason = ordered[0][0]
      top_loss_reason = ordered[-1][0]

    hour_stats: Dict[str, Dict[str, float]] = {}
    for row in sells:
      dt = row.get("dt")
      if not dt:
        continue
      hh = dt.strftime("%H")
      bucket = hour_stats.setdefault(hh, {"count": 0.0, "pnl": 0.0})
      bucket["count"] += 1
      bucket["pnl"] += float(row.get("pnl", 0.0))

    best_hour = "-"
    worst_hour = "-"
    if hour_stats:
      hours = sorted(hour_stats.items(), key=lambda kv: kv[1].get("pnl", 0.0), reverse=True)
      best_hour = hours[0][0]
      worst_hour = hours[-1][0]

    confidence = min(95.0, 25.0 + closed * 0.35)
    confidence = max(5.0, confidence)

    findings = []
    findings.append(f"Trades cerrados analizados: {closed}")
    findings.append(f"Win rate: {win_rate:.2f}% | Profit factor: {pf:.2f} | Expectancy: {expectancy:.4f}")
    findings.append(f"Razon mas rentable: {top_win_reason} | Razon mas costosa: {top_loss_reason}")
    findings.append(f"Hora mas rentable: {best_hour}h | Hora mas debil: {worst_hour}h")

    params = {}
    if pf < 1.0:
      params = {
        "poll_seconds": 4,
        "cooldown_seconds": 4,
        "min_hold_seconds": 20,
        "sell_signal_confirmations": 2,
        "signal_confirmations": 1,
      }
      message = "IA: el sistema esta fragil (PF<1). Recomiendo modo balanceado para recuperar calidad."
    elif pf >= 1.10 and win_rate >= 40:
      params = {
        "poll_seconds": 2,
        "cooldown_seconds": 2,
        "min_hold_seconds": 12,
        "sell_signal_confirmations": 1,
        "signal_confirmations": 1,
        "take_profit_pct": 0.005,
        "trailing_stop_pct": 0.002,
      }
      message = "IA: rendimiento estable. Recomiendo modo frecuencia para escalar numero de operaciones."
    else:
      params = {
        "poll_seconds": 3,
        "cooldown_seconds": 3,
        "min_hold_seconds": 16,
        "sell_signal_confirmations": 1,
        "signal_confirmations": 1,
      }
      message = "IA: zona intermedia. Recomiendo ajuste gradual y monitoreo de PF cada ciclo."

    return {
      "meta": {
        "lookback_hours": lookback_hours,
        "confidence": confidence,
        "generated_at": datetime.datetime.now().isoformat(),
      },
      "summary": {
        "closed_trades": closed,
        "win_rate": win_rate,
        "profit_factor": pf,
        "expectancy": expectancy,
        "top_win_reason": top_win_reason,
        "top_loss_reason": top_loss_reason,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
      },
      "findings": findings,
      "recommendation": {
        "message": message,
        "params": params,
      },
    }


def get_ai_model_status() -> Dict[str, Any]:
    if not os.path.exists(AI_MODEL_FILE):
        return {
            "exists": False,
            "file": AI_MODEL_FILE,
            "meta": {},
            "feature_count": 0,
        }

    try:
        with open(AI_MODEL_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            "exists": True,
            "file": AI_MODEL_FILE,
            "meta": data.get("meta", {}),
            "feature_count": len(data.get("feature_names", [])),
            "type": data.get("type", "unknown"),
        }
    except Exception as exc:
        return {
            "exists": False,
            "file": AI_MODEL_FILE,
            "meta": {},
            "feature_count": 0,
            "error": str(exc),
        }


@app.get("/")
def index():
  return HTMLResponse(HTML_PAGE, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/state")
def get_state():
    return JSONResponse(load_state())


@app.get("/config")
def get_config():
    return JSONResponse(load_config())


@app.post("/config")
def post_config(payload: Dict[str, Any]):
    return JSONResponse(save_config(payload))


@app.get("/trades-analysis")
def get_trades_analysis():
    return JSONResponse(analyze_trades_log())


@app.get("/trades-insights")
def get_trades_insights(
  symbol: str = "",
  from_date: str = "",
  to_date: str = "",
  lookback_hours: int = 0,
  limit_rows: int = 300,
):
  if not os.path.exists(TRADES_LOG_FILE):
    return JSONResponse(build_trades_insights([], limit_rows=limit_rows))

  rows = []
  try:
    with open(TRADES_LOG_FILE, "r", encoding="utf-8") as fh:
      for raw in fh:
        parsed = parse_trade_log_line(raw)
        if parsed:
          rows.append(parsed)
  except Exception:
    return JSONResponse(build_trades_insights([], limit_rows=limit_rows))

  filtered = filter_trade_rows(
      rows,
      symbol=symbol,
      from_date=from_date,
      to_date=to_date,
      lookback_hours=lookback_hours,
  )
  return JSONResponse(build_trades_insights(filtered, limit_rows=limit_rows))


@app.get("/optimizer-events")
def get_optimizer_events():
    """Retorna eventos significativos del optimizer para mostrar como hitos en el timeline."""
    log_file = "optimizer_actions.log"
    if not os.path.exists(log_file):
        return JSONResponse({"events": []})
    events = []
    skip_types = {"skip_no_new_trades", "skip_no_trades", "no_changes", "config_saved"}
    try:
        with open(log_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                etype = obj.get("type", "")
                if etype in skip_types:
                    continue
                ev = {"time": obj.get("time", ""), "type": etype}
                if etype == "remove_pair":
                    ev["label"] = f"Removido {obj.get('symbol', '?')}"
                    ev["detail"] = "; ".join(obj.get("reasons", []))
                    ev["color"] = "#ef4444"
                elif etype == "adjust_param":
                    param = obj.get("param", "?")
                    old_v = obj.get("old_value", 0)
                    new_v = obj.get("new_value", 0)
                    pname = param.replace("_pct", "").replace("_", " ").upper()
                    direction = "▲" if new_v > old_v else "▼"
                    ev["label"] = f"{pname} {direction} {round(new_v * 100, 2)}%"
                    ev["detail"] = obj.get("reason", "")
                    ev["color"] = "#f59e0b"
                elif etype == "adjust_trailing":
                    ev["label"] = f"Trailing → {round(obj.get('new_value', 0) * 100, 2)}%"
                    ev["detail"] = obj.get("reason", "")
                    ev["color"] = "#f59e0b"
                elif etype == "adjust_exit_aggressiveness":
                    ev["label"] = f"Exit aggr: conf={obj.get('new_sell_conf', '?')} hold={obj.get('new_min_hold', '?')}s"
                    ev["detail"] = obj.get("reason", "")
                    ev["color"] = "#a78bfa"
                elif etype == "auto_retrain":
                    acc = obj.get("accuracy", 0)
                    ev["label"] = f"AI retrain (acc={round(acc * 100, 1)}%)"
                    ev["detail"] = f"samples={obj.get('samples', 0)}, f1={round(obj.get('f1', 0) * 100, 1)}%"
                    ev["color"] = "#38bdf8"
                elif etype == "rebalance":
                    ev["label"] = "Rebalanceo capital"
                    ev["detail"] = obj.get("reason", "")
                    ev["color"] = "#22d3ee"
                elif etype.startswith("exit_learning"):
                    ev["label"] = f"Genie: {etype.replace('exit_learning_', '').upper()} ajustado"
                    ev["detail"] = obj.get("reason", "")
                    ev["color"] = "#c084fc"
                elif etype == "coherence_guard":
                    ev["label"] = "R:R corregido"
                    ev["detail"] = obj.get("issue", "")
                    ev["color"] = "#fb923c"
                else:
                    ev["label"] = etype
                    ev["detail"] = str(obj)
                    ev["color"] = "#94a3b8"
                events.append(ev)
    except Exception:
        pass
    return JSONResponse({"events": events})


@app.get("/ai-log-learning")
def get_ai_log_learning(symbol: str = "", lookback_hours: int = 24):
  if not os.path.exists(TRADES_LOG_FILE):
    return JSONResponse(build_ai_log_learning([], lookback_hours=lookback_hours))

  rows = []
  try:
    with open(TRADES_LOG_FILE, "r", encoding="utf-8") as fh:
      for raw in fh:
        parsed = parse_trade_log_line(raw)
        if parsed:
          rows.append(parsed)
  except Exception:
    return JSONResponse(build_ai_log_learning([], lookback_hours=lookback_hours))

  filtered = filter_trade_rows(rows, symbol=symbol, lookback_hours=lookback_hours)
  return JSONResponse(build_ai_log_learning(filtered, lookback_hours=lookback_hours))


@app.get("/ai-model-status")
def get_ai_model_status_endpoint():
    return JSONResponse(get_ai_model_status())


@app.get("/events")
def get_events(limit: int = 120):
    return JSONResponse(tail_events(limit))


@app.get("/ui-action-log")
def get_ui_action_timeline(limit: int = 80):
  return JSONResponse(get_ui_action_log(limit))


@app.post("/ui-action-log")
def post_ui_action_timeline(payload: Dict[str, Any]):
  action = str(payload.get("action", "unknown"))
  data = payload.get("payload", {})
  ok = append_ui_action_log(action, data if isinstance(data, dict) else {})
  return JSONResponse({"ok": ok})


@app.get("/dataset-summary")
def get_dataset_summary():
    return JSONResponse(summarize_dataset())


@app.get("/multi-source")
def get_multi_source(symbol: str = "BTCUSDT"):
  symbol = symbol.strip().upper()
  if not symbol.endswith("USDT") or len(symbol) < 6:
    symbol = "BTCUSDT"
  return JSONResponse(get_multi_source_snapshot(symbol))


@app.get("/multi-source-config")
def get_multi_source_cfg():
  return JSONResponse(load_multi_source_config())


@app.post("/multi-source-config")
def post_multi_source_cfg(payload: Dict[str, Any]):
  return JSONResponse(save_multi_source_config(payload))


# ─── Gestion de .env (API keys y configuracion de entorno) ───
def _load_env_file() -> dict:
    """Lee el archivo .env y devuelve un dict clave=valor."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENV_FILE)
    data = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    data[key.strip()] = val.strip()
    return data


def _save_env_file(data: dict):
    """Escribe el dict al archivo .env."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENV_FILE)
    lines = []
    for key, val in data.items():
        lines.append(f"{key}={val}")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _mask_value(val: str) -> str:
    """Enmascara un valor sensible mostrando solo los ultimos 4 caracteres."""
    if not val or val == "tu_api_key_aqui" or val == "tu_api_secret_aqui":
        return ""
    if len(val) <= 6:
        return "***"
    return "***" + val[-4:]


# Claves conocidas del .env y si son sensibles
_ENV_KEYS = {
    "TRADING_MODE": False,
    "BINANCE_API_KEY": True,
    "BINANCE_API_SECRET": True,
    "BINANCE_TESTNET": False,
}


# ─────────────────────────────────────────────
#  MARKET SCANNER ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/scanner/state")
def get_scanner_state_endpoint():
    if market_scanner is None:
        return JSONResponse({"error": "Market scanner no disponible"}, status_code=503)
    state = market_scanner.get_scanner_state()
    return JSONResponse(state)


@app.get("/scanner/alerts")
def get_scanner_alerts_endpoint():
    if market_scanner is None:
        return JSONResponse({"alerts": []})
    limit = 20
    alerts = market_scanner.get_scanner_alerts(limit=limit)
    return JSONResponse({"alerts": alerts})


@app.post("/scanner/force-scan")
def force_scan_endpoint():
    if market_scanner is None:
        return JSONResponse({"error": "Market scanner no disponible"}, status_code=503)
    result = market_scanner.force_scan()
    return JSONResponse(result)


@app.get("/scanner/stream-scan")
def stream_scan_endpoint():
    if market_scanner is None:
        return JSONResponse({"error": "Market scanner no disponible"}, status_code=503)

    def event_generator():
        for chunk in market_scanner.run_scan_streaming():
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─────────────────── Auto Optimizer Endpoints ───────────────────

@app.get("/smart-optimizer/state")
def get_smart_optimizer_state():
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    return JSONResponse(auto_optimizer.get_optimizer_state())


@app.post("/smart-optimizer/run")
def force_smart_optimize():
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    result = auto_optimizer.force_optimize()
    return JSONResponse(result)


@app.post("/smart-optimizer/config")
async def update_smart_optimizer_config(request: Request):
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    body = await request.json()
    enabled = bool(body.get("enabled", False))
    interval = int(body.get("interval_minutes", 10))
    auto_optimizer.update_optimizer_config(enabled, interval)
    return JSONResponse({"ok": True, "enabled": enabled, "interval_minutes": interval})


@app.get("/smart-optimizer/history")
def get_smart_optimizer_history():
    if auto_optimizer is None:
        return JSONResponse({"history": []})
    state = auto_optimizer.get_optimizer_state()
    return JSONResponse({"history": state.get("history", [])})


# ─────────────────── Grid Search Endpoints ───────────────────

@app.get("/grid-search/state")
def get_grid_search_state():
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    return JSONResponse(auto_optimizer.get_grid_search_state())


@app.post("/grid-search/run")
async def run_grid_search(request: Request):
    """Ejecuta grid search en background thread."""
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)

    gs_state = auto_optimizer.get_grid_search_state()
    if gs_state.get("running"):
        return JSONResponse({"error": "Grid search ya en ejecucion"}, status_code=409)

    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}

    symbol = str(body.get("symbol", "BTCUSDT")).strip().upper()
    limit = min(int(body.get("limit", 1000)), 5000)
    interval = str(body.get("interval", "1m"))
    balance = float(body.get("balance", 100.0))
    fee_pct = float(body.get("fee_pct", 0.001))
    metric = str(body.get("metric", "profit_factor"))

    # Parsear rangos custom o usar defaults
    tp_range = _parse_range(body.get("tp_range"))
    sl_range = _parse_range(body.get("sl_range"))
    trailing_range = _parse_range(body.get("trailing_range"))

    import threading
    def _run():
        auto_optimizer.run_grid_search(
            symbol=symbol, limit=limit, interval=interval, balance=balance,
            tp_range=tp_range, sl_range=sl_range, trailing_range=trailing_range,
            fee_pct=fee_pct, metric=metric,
        )
    threading.Thread(target=_run, daemon=True, name="GridSearch").start()
    return JSONResponse({"ok": True, "message": "Grid search iniciado"})


@app.post("/grid-search/apply")
def apply_grid_search_best():
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    result = auto_optimizer.apply_grid_search_best()
    return JSONResponse(result)


def _parse_range(val):
    """Parsea [min, max, step] a tupla o None."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return (float(val[0]), float(val[1]), float(val[2]))
    return None


# ─────────────────── Regime Params Endpoints ───────────────────

@app.get("/regime-params")
def get_regime_params():
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    return JSONResponse(auto_optimizer.load_regime_params())


@app.post("/regime-params")
async def save_regime_params(request: Request):
    """Guarda perfiles de regimen editados desde el dashboard."""
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalido"}, status_code=400)
    valid_regimes = {"trending_up", "trending_down", "ranging", "volatile"}
    valid_fields = {"take_profit_pct", "stop_loss_pct", "trailing_stop_pct", "trade_pct_mult"}
    cleaned = {}
    for regime, params in body.items():
        if regime not in valid_regimes or not isinstance(params, dict):
            continue
        cleaned[regime] = {}
        for field in valid_fields:
            if field in params:
                val = float(params[field])
                if field == "trade_pct_mult":
                    val = max(0.1, min(2.0, val))
                else:
                    val = max(0.001, min(0.05, val))
                cleaned[regime][field] = val
    if not cleaned:
        return JSONResponse({"error": "Sin datos validos"}, status_code=400)
    current = auto_optimizer.load_regime_params()
    for regime, params in cleaned.items():
        if regime in current:
            current[regime].update(params)
        else:
            current[regime] = params
    auto_optimizer.save_regime_params(current)
    return JSONResponse({"ok": True, "saved": cleaned})


@app.post("/regime-params/update-from-grid")
async def update_regime_from_grid(request: Request):
    """Guarda los mejores params del grid search para un regimen especifico."""
    if auto_optimizer is None:
        return JSONResponse({"error": "Auto optimizer no disponible"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    regime = str(body.get("regime", "")).strip().lower()
    if regime not in ("trending_up", "trending_down", "ranging", "volatile"):
        return JSONResponse({"error": "Regimen invalido"}, status_code=400)
    result = auto_optimizer.update_regime_from_grid_search(regime)
    return JSONResponse(result)


# ─────────────────── Intelligence Engine Endpoints ───────────────────

@app.get("/intelligence/state")
def get_intelligence_state():
    # Leer estado del intel engine del bot (proceso separado) via bot_state.json
    try:
        state_data = load_state()
        intel_from_bot = state_data.get("intel_state")
        if intel_from_bot and intel_from_bot.get("signals_evaluated", 0) > 0:
            return JSONResponse(intel_from_bot)
    except Exception:
        pass
    # Fallback a instancia local del dashboard
    if intelligence_engine is None:
        return JSONResponse({"error": "Inteli Genie no disponible", "signals_evaluated": 0})
    return JSONResponse(intelligence_engine.get_intelligence_state())


@app.post("/intelligence/config")
async def update_intelligence_config(request: Request):
    """Actualiza configuracion de inteligencia en runtime_config."""
    body = await request.json()
    cfg = load_config()
    changed = []
    for key, val in body.items():
        if key.startswith("intel_"):
            cfg[key] = val
            changed.append(key)
    save_config(cfg)
    return JSONResponse({"ok": True, "changed": changed})


@app.post("/intelligence/train-model")
async def train_ml_model(request: Request):
    """Entrena el modelo ML de forma sincrona."""
    if train_model is None:
        return JSONResponse({"error": "train_model no disponible"}, status_code=503)
    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    horizon = int(body.get("horizon", 5))
    threshold = float(body.get("threshold", 0.0005))
    result = train_model.train_and_save(horizon=horizon, threshold=threshold)
    return JSONResponse(result)


@app.get("/intelligence/model-status")
def get_ml_model_status():
    if train_model is None:
        return JSONResponse({"exists": False})
    return JSONResponse(train_model.get_model_status())


@app.post("/backtest")
async def run_backtest_endpoint(request: Request):
    """Ejecuta backtest desde el dashboard con charts inline."""
    if backtest_mod is None:
        return JSONResponse({"error": "backtest module not available"}, status_code=503)
    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    symbol = str(body.get("symbol", "BTCUSDT")).strip().upper()
    limit = min(int(body.get("limit", 1000)), 5000)
    interval = str(body.get("interval", "1m"))
    trailing = float(body.get("trailing_stop_pct", 0.006))
    tp = float(body.get("take_profit_pct", 0.008))
    sl = float(body.get("stop_loss_pct", 0.005))
    balance = float(body.get("balance", 100.0))
    fee = float(body.get("fee_pct", 0.001))
    walk = bool(body.get("walk_forward", False))
    include_chart = bool(body.get("chart", False))
    try:
        raw_klines = backtest_mod.fetch_klines(symbol, interval, limit)
        candles = backtest_mod.parse_klines(raw_klines)
        bt_kwargs = dict(
            initial_balance=balance,
            trailing_stop_pct=trailing,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            fee_pct=fee,
        )
        if walk:
            results = backtest_mod.walk_forward(candles, symbol=symbol, **bt_kwargs)
            summary = []
            for r in results:
                summary.append({
                    "candles": r.total_candles, "trades": r.total_trades,
                    "pnl": r.total_pnl, "pnl_pct": r.total_pnl_pct,
                    "win_rate": r.win_rate, "pf": r.profit_factor,
                    "max_dd_pct": r.max_drawdown_pct, "sharpe": r.sharpe_ratio,
                })
            pnls = [r.total_pnl for r in results]
            resp = {
                "walk_forward": True, "symbol": symbol, "windows": len(results),
                "total_pnl": round(sum(pnls), 4),
                "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 4),
                "profitable_windows": sum(1 for p in pnls if p > 0),
                "details": summary,
            }
            if include_chart and results:
                resp["chart_b64"] = _generate_chart_b64(results[-1])
            return JSONResponse(resp)
        else:
            r = backtest_mod.run_backtest(candles, symbol=symbol, **bt_kwargs)
            trades_data = [
                {"entry": t.entry_price, "exit": t.exit_price, "pnl": round(t.pnl, 4),
                 "pnl_pct": round(t.pnl_pct * 100, 4), "reason": t.reason}
                for t in r.trades
            ]
            # Downsample equity curve si es muy grande (max 500 puntos)
            eq = r.equity_curve
            if len(eq) > 500:
                step = len(eq) / 500
                eq = [eq[int(i * step)] for i in range(500)]
            resp = {
                "symbol": r.symbol, "candles": r.total_candles,
                "balance_initial": r.initial_balance, "balance_final": r.final_balance,
                "total_pnl": r.total_pnl, "total_pnl_pct": r.total_pnl_pct,
                "trades": r.total_trades, "wins": r.wins, "losses": r.losses,
                "win_rate": r.win_rate, "profit_factor": r.profit_factor,
                "max_drawdown_pct": r.max_drawdown_pct, "sharpe": r.sharpe_ratio,
                "expectancy": r.expectancy, "trade_list": trades_data,
                "trades_detail": [
                    {"entry_price": t.entry_price, "exit_price": t.exit_price,
                     "pnl": round(t.pnl, 4), "pnl_pct": round(t.pnl_pct * 100, 4), "reason": t.reason}
                    for t in r.trades
                ],
                "equity_curve": [round(v, 4) for v in eq],
                "take_profit_pct": tp, "stop_loss_pct": sl, "trailing_stop_pct": trailing,
            }
            if include_chart:
                resp["chart_b64"] = _generate_chart_b64(r)
            return JSONResponse(resp)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _generate_chart_b64(r) -> str:
    """Genera chart de backtest como imagen base64 PNG."""
    import base64
    import io
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(
        f"Backtest: {r.symbol} | {r.total_candles} velas | PnL: ${r.total_pnl:+.2f} ({r.total_pnl_pct:+.1f}%)",
        fontsize=11, color="white",
    )
    fig.patch.set_facecolor("#1e293b")
    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # 1. Equity Curve
    ax1 = axes[0]
    if r.equity_curve:
        ax1.plot(r.equity_curve, linewidth=0.8, color="#38bdf8")
        ax1.axhline(y=r.initial_balance, linestyle="--", color="gray", alpha=0.6, linewidth=0.7)
        ax1.fill_between(range(len(r.equity_curve)), r.initial_balance, r.equity_curve,
                         where=[e >= r.initial_balance for e in r.equity_curve], alpha=0.15, color="#22c55e")
        ax1.fill_between(range(len(r.equity_curve)), r.initial_balance, r.equity_curve,
                         where=[e < r.initial_balance for e in r.equity_curve], alpha=0.15, color="#ef4444")
    ax1.set_ylabel("Equity ($)", fontsize=8)
    ax1.set_title("Equity Curve", fontsize=9)
    ax1.grid(True, alpha=0.2, color="#475569")

    # 2. Drawdown
    ax2 = axes[1]
    if r.equity_curve:
        peak = r.equity_curve[0]
        drawdowns = []
        for eq in r.equity_curve:
            if eq > peak:
                peak = eq
            drawdowns.append(-(peak - eq) / peak * 100)
        ax2.fill_between(range(len(drawdowns)), 0, drawdowns, color="#ef4444", alpha=0.4)
        ax2.plot(drawdowns, linewidth=0.6, color="#dc2626")
    ax2.set_ylabel("Drawdown (%)", fontsize=8)
    ax2.set_title(f"Drawdown (max: {r.max_drawdown_pct:.2f}%)", fontsize=9)
    ax2.grid(True, alpha=0.2, color="#475569")

    # 3. PnL Distribution
    ax3 = axes[2]
    if r.trades:
        pnls = [t.pnl for t in r.trades]
        colors = ["#22c55e" if p > 0 else "#ef4444" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, width=0.8)
        ax3.axhline(y=0, color="gray", linewidth=0.5)
        if r.expectancy != 0:
            ax3.axhline(y=r.expectancy, linestyle="--", color="#f59e0b", alpha=0.7, linewidth=0.7)
    ax3.set_ylabel("PnL ($)", fontsize=8)
    ax3.set_xlabel("Trade #", fontsize=8)
    ax3.set_title(f"Trades | WR: {r.win_rate:.1f}% | PF: {r.profit_factor:.2f} | Sharpe: {r.sharpe_ratio:.3f}", fontsize=9)
    ax3.grid(True, alpha=0.2, color="#475569")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


@app.get("/env-config")
def get_env_config():
    data = _load_env_file()
    values = {}
    has_value = {}
    for key, is_sensitive in _ENV_KEYS.items():
        val = data.get(key, "")
        values[key] = _mask_value(val) if is_sensitive else val
        has_value[key] = bool(val and val not in ("tu_api_key_aqui", "tu_api_secret_aqui"))
    return JSONResponse({"values": values, "has_value": has_value})


@app.post("/env-config")
def post_env_config(payload: Dict[str, Any]):
    current = _load_env_file()
    for key in _ENV_KEYS:
        if key in payload:
            new_val = str(payload[key]).strip()
            # Si el valor viene enmascarado (***...) o vacio, no sobreescribir
            if new_val.startswith("***") or new_val == "":
                continue
            current[key] = new_val
    _save_env_file(current)
    return JSONResponse({"status": "ok"})


@app.post("/env-test-connection")
def post_env_test_connection():
    """Ejecuta un test rapido de conexion con las credenciales guardadas."""
    env_data = _load_env_file()
    api_key = env_data.get("BINANCE_API_KEY", "")
    api_secret = env_data.get("BINANCE_API_SECRET", "")
    testnet_str = env_data.get("BINANCE_TESTNET", "True")
    use_testnet = testnet_str.lower() in ("true", "1", "yes")

    results = []
    base = "https://testnet.binance.vision" if use_testnet else "https://api.binance.com"

    # Test 1: Ping
    try:
        resp = requests.get(f"{base}/api/v3/ping", timeout=5)
        results.append({"test": "Ping API", "ok": resp.status_code == 200, "message": "Servidor accesible" if resp.status_code == 200 else f"HTTP {resp.status_code}"})
    except Exception as e:
        results.append({"test": "Ping API", "ok": False, "message": str(e)[:80]})

    # Test 2: Precio
    try:
        resp = requests.get(f"{base}/api/v3/ticker/price", params={"symbol": "BTCUSDT"}, timeout=5)
        if resp.status_code == 200:
            price = resp.json().get("price", "?")
            results.append({"test": "Precio BTC", "ok": True, "message": f"BTC/USDT = ${price}"})
        else:
            results.append({"test": "Precio BTC", "ok": False, "message": f"HTTP {resp.status_code}"})
    except Exception as e:
        results.append({"test": "Precio BTC", "ok": False, "message": str(e)[:80]})

    # Test 3: Auth (si hay keys)
    has_keys = api_key and api_key != "tu_api_key_aqui" and api_secret and api_secret != "tu_api_secret_aqui"
    if has_keys:
        try:
            import hmac, hashlib
            from urllib.parse import urlencode
            params = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
            query = urlencode(params)
            sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            params["signature"] = sig
            resp = requests.get(
                f"{base}/api/v3/account",
                params=params,
                headers={"X-MBX-APIKEY": api_key},
                timeout=5,
            )
            if resp.status_code == 200:
                balances = [b for b in resp.json().get("balances", [])
                            if float(b["free"]) + float(b["locked"]) > 0]
                results.append({"test": "Autenticacion", "ok": True, "message": f"OK - {len(balances)} activos con balance"})
            else:
                err = resp.json().get("msg", str(resp.status_code))
                results.append({"test": "Autenticacion", "ok": False, "message": err})
        except Exception as e:
            results.append({"test": "Autenticacion", "ok": False, "message": str(e)[:80]})
    else:
        results.append({"test": "Autenticacion", "ok": False, "message": "API keys no configuradas"})

    return JSONResponse({"results": results})


# ---------------------------------------------------------------------------
# Analisis periodico cada 12 horas — reporte para humanos
# ---------------------------------------------------------------------------

def _load_periodic_analysis():
    if os.path.exists(PERIODIC_ANALYSIS_FILE):
        try:
            with open(PERIODIC_ANALYSIS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"reports": []}


def _save_periodic_analysis(data):
    with open(PERIODIC_ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _generate_analysis_report() -> dict:
    """Genera un reporte completo con lenguaje humano y recomendaciones."""
    # Leer trades
    rows = []
    if os.path.exists(TRADES_LOG_FILE):
        try:
            with open(TRADES_LOG_FILE, "r", encoding="utf-8") as fh:
                for raw in fh:
                    parsed = parse_trade_log_line(raw)
                    if parsed:
                        rows.append(parsed)
        except Exception:
            pass

    insights = build_trades_insights(rows, limit_rows=50000)
    summary = insights.get("summary", {})
    by_symbol = insights.get("by_symbol", {})
    by_day = insights.get("by_day", [])
    by_reason = insights.get("by_reason", [])

    # Config actual
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass

    wallet_balance = float(cfg.get("wallet_balance", 0))
    target_usdt = float(cfg.get("target_usdt", 0))

    pnl_total = float(summary.get("pnl_total", 0))
    closed = int(summary.get("closed_trades", 0))
    wins = int(summary.get("wins", 0))
    losses = int(summary.get("losses", 0))
    wr = float(summary.get("win_rate", 0))
    pf = float(summary.get("profit_factor", 0))
    dd = float(summary.get("max_drawdown", 0))
    gross_profit = float(summary.get("gross_profit", 0))
    gross_loss = float(summary.get("gross_loss", 0))
    best = float(summary.get("best_trade", 0))
    worst = float(summary.get("worst_trade", 0))

    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    today_entry = next((d for d in by_day if d.get("date") == today_str), None)
    day_pnl = float(today_entry["pnl_total"]) if today_entry else 0.0
    day_trades = int(today_entry["sell_count"]) if today_entry else 0

    # Calcular tendencia de PnL (ultimos 3 dias)
    last3 = by_day[-3:] if len(by_day) >= 3 else by_day
    trend_pnls = [float(d.get("pnl_total", 0)) for d in last3]
    trend_direction = "alcista" if len(trend_pnls) >= 2 and trend_pnls[-1] > trend_pnls[0] else "bajista" if len(trend_pnls) >= 2 and trend_pnls[-1] < trend_pnls[0] else "estable"

    # Trailing stop analysis
    trailing_entry = next((r for r in by_reason if "trailing" in str(r.get("reason", "")).lower()), None)
    trailing_pnl = float(trailing_entry.get("pnl_total", 0)) if trailing_entry else 0
    trailing_count = int(trailing_entry.get("sell_count", 0)) if trailing_entry else 0

    # Dias consecutivos negativos
    consecutive_neg = 0
    for d in reversed(by_day):
        if float(d.get("pnl_total", 0)) < 0:
            consecutive_neg += 1
        else:
            break

    # Progreso al objetivo
    needed = target_usdt - wallet_balance if target_usdt > 0 else 0
    progress_pct = (pnl_total / needed * 100) if needed > 0 else 0

    # ----- Construir secciones del reporte -----

    # 1. Resumen ejecutivo
    if pf >= 1.3 and wr >= 44:
        estado = "excelente"
        estado_emoji = "estrella"
    elif pf >= 1.1 and wr >= 40:
        estado = "bueno"
        estado_emoji = "ok"
    elif pf >= 1.0:
        estado = "ajustado"
        estado_emoji = "atencion"
    else:
        estado = "critico"
        estado_emoji = "peligro"

    resumen_lines = []
    resumen_lines.append(f"El bot lleva {closed} operaciones cerradas con un PnL total de {'+'  if pnl_total >= 0 else ''}${pnl_total:.2f}.")
    resumen_lines.append(f"De cada 100 trades, {wr:.0f} son ganadores (win rate {wr:.1f}%).")
    resumen_lines.append(f"Por cada dolar que pierde, recupera ${pf:.2f} (profit factor).")
    if needed > 0:
        resumen_lines.append(f"Progreso al objetivo: {progress_pct:.1f}% (${pnl_total:.2f} de ${needed:.0f} necesarios).")
    resumen_lines.append(f"Estado general: {estado.upper()}.")

    # 2. Rendimiento por par
    par_lines = []
    sym_sorted = sorted(by_symbol.items(), key=lambda kv: float(kv[1].get("pnl_total", 0)), reverse=True)
    for sym, stats in sym_sorted:
        s_pnl = float(stats.get("pnl_total", 0))
        s_pf = float(stats.get("profit_factor", 0))
        s_wr = float(stats.get("win_rate", 0))
        s_trades = int(stats.get("closed_trades", 0))
        if s_pf >= 1.3:
            nivel = "fuerte"
        elif s_pf >= 1.1:
            nivel = "aceptable"
        elif s_pf >= 1.0:
            nivel = "debil"
        else:
            nivel = "perdedor"
        par_lines.append({
            "symbol": sym,
            "pnl": s_pnl,
            "pf": s_pf,
            "wr": s_wr,
            "trades": s_trades,
            "nivel": nivel,
            "texto": f"{sym}: {'+'  if s_pnl >= 0 else ''}${s_pnl:.2f} en {s_trades} trades (PF {s_pf:.2f}, WR {s_wr:.1f}%) — {nivel.upper()}"
        })

    # 3. Analisis del dia
    dia_lines = []
    if today_entry:
        if day_pnl > 0:
            dia_lines.append(f"Hoy va bien: +${day_pnl:.2f} en {day_trades} trades.")
        elif day_pnl < 0:
            dia_lines.append(f"Hoy esta en negativo: ${day_pnl:.2f} en {day_trades} trades.")
        else:
            dia_lines.append(f"Hoy sin ganancia ni perdida en {day_trades} trades.")
    else:
        dia_lines.append("No hay trades registrados hoy todavia.")

    if consecutive_neg > 0:
        dia_lines.append(f"Atencion: {consecutive_neg} dia(s) consecutivo(s) en negativo al cierre.")

    # 4. Riesgos detectados
    riesgos = []
    if pf < 1.1:
        riesgos.append(f"Profit factor bajo ({pf:.2f}). El margen de ganancia es muy estrecho.")
    if dd > 8:
        riesgos.append(f"Drawdown alto (${dd:.2f}). Hubo un momento donde el bot perdio mucho antes de recuperar.")
    if trailing_pnl < -5:
        riesgos.append(f"Trailing stop ha restado ${abs(trailing_pnl):.2f} en {trailing_count} trades. Si sigue activo, vigilar.")
    if consecutive_neg >= 2:
        riesgos.append(f"{consecutive_neg} dias seguidos en negativo. Posible cambio de regimen de mercado.")
    # Pares debiles
    for sym, stats in by_symbol.items():
        s_pf = float(stats.get("profit_factor", 0))
        s_trades = int(stats.get("closed_trades", 0))
        if s_pf < 1.05 and s_trades > 100:
            riesgos.append(f"{sym} tiene PF {s_pf:.2f} con {s_trades} trades — casi no genera ganancia.")
    if wr < 40:
        riesgos.append(f"Win rate global bajo ({wr:.1f}%). Menos de 4 de cada 10 trades son positivos.")
    if not riesgos:
        riesgos.append("No se detectan riesgos significativos por ahora.")

    # 5. Recomendaciones
    recomendaciones = []
    # Evaluar pares para quitar
    for sym, stats in by_symbol.items():
        s_pf = float(stats.get("profit_factor", 0))
        s_pnl = float(stats.get("pnl_total", 0))
        s_trades = int(stats.get("closed_trades", 0))
        if s_pf < 1.05 and s_trades > 200:
            recomendaciones.append(f"Considerar quitar {sym} del portafolio. Con PF {s_pf:.2f} y {s_trades} trades, es peso muerto. Redirigir su capital a pares mas fuertes.")
        elif s_pf < 1.1 and s_trades > 300:
            recomendaciones.append(f"Reducir el capital asignado a {sym} (PF {s_pf:.2f}). No rinde lo suficiente.")

    # Evaluar pares para aumentar
    for sym, stats in sym_sorted[:2]:
        s_pf = float(stats.get("profit_factor", 0))
        if s_pf >= 1.4:
            recomendaciones.append(f"{sym} es muy rentable (PF {s_pf:.2f}). Considerar aumentar su capital asignado.")

    if pf < 1.15 and pf >= 1.0:
        recomendaciones.append("El profit factor global esta ajustado. Evitar agregar mas pares hasta estabilizar.")

    if dd > 6:
        recomendaciones.append(f"El drawdown maximo es ${dd:.2f}. Considerar bajar el porcentaje de capital por trade (trade_pct) para proteger mas.")

    if trailing_pnl < -8 and trailing_count > 50:
        recomendaciones.append("El trailing stop historico ha costado mucho. Ya se corrigio con el trailing diferido, seguir monitoreando.")

    if progress_pct > 0 and progress_pct < 20:
        days_active = len(by_day) if by_day else 1
        avg_daily = pnl_total / days_active if days_active > 0 else 0
        if avg_daily > 0:
            days_remaining = (needed - pnl_total) / avg_daily
            recomendaciones.append(f"Al ritmo actual (+${avg_daily:.2f}/dia), faltan aprox. {days_remaining:.0f} dias para el objetivo.")
        else:
            recomendaciones.append("El ritmo actual no permite estimar cuando se alcanzara el objetivo.")

    if not recomendaciones:
        recomendaciones.append("El bot esta operando bien. Mantener la configuracion actual y seguir monitoreando.")

    # 6. Comparacion con analisis anterior
    prev_data = _load_periodic_analysis()
    prev_reports = prev_data.get("reports", [])
    comparacion = None
    if prev_reports:
        prev = prev_reports[-1]
        prev_s = prev.get("data", {}).get("summary", {})
        comparacion = {
            "prev_time": prev.get("timestamp", ""),
            "pnl_prev": float(prev_s.get("pnl_total", 0)),
            "pnl_now": pnl_total,
            "pnl_delta": pnl_total - float(prev_s.get("pnl_total", 0)),
            "trades_prev": int(prev_s.get("closed_trades", 0)),
            "trades_now": closed,
            "trades_delta": closed - int(prev_s.get("closed_trades", 0)),
            "pf_prev": float(prev_s.get("profit_factor", 0)),
            "pf_now": pf,
            "wr_prev": float(prev_s.get("win_rate", 0)),
            "wr_now": wr,
        }
        delta_pnl = comparacion["pnl_delta"]
        delta_trades = comparacion["trades_delta"]
        delta_pf = pf - comparacion["pf_prev"]
        comparacion["texto"] = (
            f"Desde el ultimo analisis ({prev.get('timestamp', '?')[:16].replace('T', ' ')}): "
            f"{'+'  if delta_pnl >= 0 else ''}${delta_pnl:.2f} en {delta_trades} trades nuevos. "
            f"PF {'subio' if delta_pf > 0 else 'bajo'} de {comparacion['pf_prev']:.2f} a {pf:.2f}."
        )

    report = {
        "timestamp": now.isoformat(),
        "estado": estado,
        "estado_emoji": estado_emoji,
        "resumen": resumen_lines,
        "pares": par_lines,
        "dia": dia_lines,
        "riesgos": riesgos,
        "recomendaciones": recomendaciones,
        "comparacion": comparacion,
        "data": {
            "summary": {
                "closed_trades": closed,
                "pnl_total": pnl_total,
                "win_rate": wr,
                "profit_factor": pf,
                "max_drawdown": dd,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
            },
            "by_symbol": {sym: {"pnl_total": float(s.get("pnl_total", 0)), "profit_factor": float(s.get("profit_factor", 0)), "closed_trades": int(s.get("closed_trades", 0))} for sym, s in by_symbol.items()},
        },
    }

    # Guardar el reporte (mantener ultimos 20)
    prev_data["reports"].append(report)
    if len(prev_data["reports"]) > 20:
        prev_data["reports"] = prev_data["reports"][-20:]
    _save_periodic_analysis(prev_data)

    return report


@app.get("/periodic-analysis")
def get_periodic_analysis():
    """Devuelve el ultimo reporte periodico, o vacio si no hay."""
    data = _load_periodic_analysis()
    reports = data.get("reports", [])
    if reports:
        return JSONResponse(reports[-1])
    return JSONResponse({"timestamp": None, "estado": "sin_datos", "resumen": ["No hay analisis aun. Se generara automaticamente."], "pares": [], "dia": [], "riesgos": [], "recomendaciones": [], "comparacion": None})


@app.post("/periodic-analysis/run")
def run_periodic_analysis():
    """Ejecuta un analisis bajo demanda."""
    report = _generate_analysis_report()
    return JSONResponse(report)


@app.get("/periodic-analysis/history")
def get_periodic_analysis_history():
    """Devuelve los ultimos reportes para historial."""
    data = _load_periodic_analysis()
    reports = data.get("reports", [])
    slim = [{"timestamp": r.get("timestamp"), "estado": r.get("estado"), "pnl": r.get("data", {}).get("summary", {}).get("pnl_total", 0)} for r in reports]
    return JSONResponse({"reports": slim})


@app.post("/paper-reset")
async def post_paper_reset(request: Request):
    """Borra TODOS los datos de trading para empezar de cero (solo paper)."""
    preserve_training_data = False
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            preserve_training_data = bool(payload.get("preserve_training_data", False))
    except Exception:
        preserve_training_data = False

    # Verificar modo paper
    state = load_state()
    mode = state.get("mode", "paper")
    if mode == "real":
        return JSONResponse({"status": "error", "message": "No se puede resetear en modo real"}, status_code=403)

    deleted = []
    errors = []

    # 1. Borrar todos los archivos de datos
    files_to_delete = [
        TRADES_LOG_FILE,          # trades.log
        BOT_EVENTS_FILE,          # bot_events.log
        PERIODIC_ANALYSIS_FILE,   # periodic_analysis.json
        AI_MODEL_FILE,            # ai_model.json
        STATE_FILE,               # bot_state.json
        MULTI_SOURCE_CONFIG_FILE, # multi_source_config.json
        "optimizer_actions.log",
        "intelligence_events.log",
        "train_model.log",
        "__perf.json",
        "_dash_err.log",
        "_served_dashboard.html",
        "__test.txt",
        "bot_state.json.tmp",
    ]
    if not preserve_training_data:
        files_to_delete.append(ML_DATASET_FILE)  # ml_dataset.csv
    else:
        deleted.append(ML_DATASET_FILE + " (preservado)")
    # Incluir archivos .tmp del state y otros residuales
    for f in os.listdir("."):
        if f.startswith("bot_state.json.") and f.endswith(".tmp"):
            if f not in files_to_delete:
                files_to_delete.append(f)

    # Para archivos que el bot pueda tener abiertos, truncar si no se puede borrar
    for fpath in files_to_delete:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                deleted.append(fpath)
            except PermissionError:
                # El bot tiene el archivo abierto: truncar contenido
                try:
                    with open(fpath, "w", encoding="utf-8") as fh:
                        fh.truncate(0)
                    deleted.append(fpath + " (truncado)")
                except Exception as e2:
                    errors.append({"file": fpath, "error": f"truncate failed: {str(e2)[:50]}"})
            except Exception as e:
                errors.append({"file": fpath, "error": str(e)[:60]})

    # 2. Resetear runtime_config.json preservando config del usuario
    #    Se mantienen: wallet, symbols, sistemas habilitados, parametros de riesgo
    #    Se resetean: solo datos que el optimizador sobreescribira de nuevo
    PRESERVE_KEYS = {
        "symbols", "wallet_balance", "pair_allocations", "pair_targets",
        "trade_pct", "take_profit_pct", "trailing_stop_pct", "target_usdt",
        "update_mode", "poll_seconds", "signal_confirmations", "sell_signal_confirmations",
        "min_hold_seconds", "cooldown_seconds", "daily_loss_limit_usdt", "max_drawdown_pct", "max_trades_per_hour",
        "smart_optimizer_enabled", "smart_optimizer_interval_minutes",
        "optimizer_auto_enabled", "optimizer_auto_minutes",
        "ai_enabled", "ai_mode", "ai_min_confidence",
        "ai_adaptive_sizing", "ai_high_confidence", "ai_low_confidence",
        "ai_high_trade_pct_mult", "ai_low_trade_pct_mult", "ai_high_tp_mult", "ai_low_tp_mult",
        "intel_enabled", "intel_hours_enabled", "intel_regime_enabled",
        "intel_volume_enabled", "intel_score_enabled", "intel_correlation_enabled",
        "intel_adaptive_trailing_enabled", "intel_exit_learning_enabled",
    }
    try:
        new_config = DEFAULT_CONFIG.copy()
        # Leer config actual y preservar claves del usuario
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    current = json.load(f)
                for k in PRESERVE_KEYS:
                    if k in current:
                        new_config[k] = current[k]
            except Exception:
                pass  # Si falla la lectura, usar defaults
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(new_config, f, ensure_ascii=False, indent=2)
        deleted.append(CONFIG_FILE + " (reseteado, config usuario preservada)")
    except Exception as e:
        errors.append({"file": CONFIG_FILE, "error": str(e)[:60]})

    # 3. Crear bot_state.json limpio con pares default
    new_state = default_state()
    new_state["mode"] = "paper"
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(new_state, f, ensure_ascii=False, indent=2)
        deleted.append(STATE_FILE + " (recreado limpio)")
    except Exception as e:
        errors.append({"file": STATE_FILE, "error": str(e)[:60]})

    return JSONResponse({"status": "ok", "deleted": deleted, "errors": errors})


# ─────────────────── Database Endpoints ───────────────────

@app.get("/db/trades")
def db_get_trades(symbol: str = "", limit: int = 200):
    if db_mod is None:
        return JSONResponse({"error": "DB no disponible"}, status_code=503)
    trades = db_mod.get_trades(symbol=symbol or None, limit=min(limit, 1000))
    return JSONResponse({"trades": trades, "total": len(trades)})


@app.get("/db/trades-summary")
def db_get_summary():
    if db_mod is None:
        return JSONResponse({"error": "DB no disponible"}, status_code=503)
    return JSONResponse(db_mod.get_trades_summary())


@app.get("/db/trades-by-day")
def db_trades_by_day(days: int = 30):
    if db_mod is None:
        return JSONResponse({"error": "DB no disponible"}, status_code=503)
    return JSONResponse({"days": db_mod.get_trades_by_day(min(days, 365))})


@app.get("/db/ml-summary")
def db_ml_summary():
    if db_mod is None:
        return JSONResponse({"error": "DB no disponible"}, status_code=503)
    return JSONResponse(db_mod.get_ml_features_summary())


@app.post("/db/migrate")
def db_migrate():
    """Migra trades.log a SQLite."""
    if db_mod is None:
        return JSONResponse({"error": "DB no disponible"}, status_code=503)
    count = db_mod.migrate_trades_log(TRADES_LOG_FILE)
    return JSONResponse({"ok": True, "imported": count})


# ─────────────────── Portfolio Endpoints ───────────────────

@app.get("/portfolio/correlation")
def portfolio_correlation(interval: str = "5m", limit: int = 100):
    if portfolio_mod is None:
        return JSONResponse({"error": "Portfolio no disponible"}, status_code=503)
    try:
        cfg = load_config()
        symbols = cfg.get("symbols", [])
        if not symbols:
            return JSONResponse({"error": "Sin pares activos"}, status_code=400)
        result = portfolio_mod.build_correlation_matrix(symbols, interval, min(limit, 500))
        result["high_correlations"] = portfolio_mod.get_high_correlations(result)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, status_code=500)


@app.websocket("/ws")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    last_payload = ""
    try:
        while True:
            payload = json.dumps(load_state(), ensure_ascii=False)
            if payload != last_payload:
                await websocket.send_text(payload)
                last_payload = payload
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
    except Exception:
        return


# ---------------------------------------------------------------------------
#  RL AGENT ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/rl/state")
async def rl_state():
    if not rl_mod:
        return JSONResponse({"error": "rl_agent not available"}, 404)
    agent = rl_mod.get_agent()
    return JSONResponse(agent.get_stats())


@app.post("/rl/suggestion")
async def rl_suggestion(request: Request):
    if not rl_mod:
        return JSONResponse({"error": "rl_agent not available"}, 404)
    body = await request.json()
    result = rl_mod.get_rl_suggestion(
        ema_diff_pct=float(body.get("ema_diff_pct", 0)),
        rsi=float(body.get("rsi", 50)),
        macd_hist=float(body.get("macd_hist", 0)),
        regime=str(body.get("regime", "ranging")),
        in_position=bool(body.get("in_position", False)),
        hour=int(body.get("hour", 12)),
        greedy=bool(body.get("greedy", True)),
    )
    return JSONResponse(result)


@app.post("/rl/reset")
async def rl_reset():
    if not rl_mod:
        return JSONResponse({"error": "rl_agent not available"}, 404)
    agent = rl_mod.get_agent()
    agent.reset()
    agent.save()
    return JSONResponse({"status": "ok", "message": "RL agent reset"})


@app.post("/rl/save")
async def rl_save():
    if not rl_mod:
        return JSONResponse({"error": "rl_agent not available"}, 404)
    agent = rl_mod.get_agent()
    agent.save()
    return JSONResponse({"status": "ok", "q_table_size": len(agent.q_table)})


# ---------------------------------------------------------------------------
#  EXCHANGE ADAPTER ENDPOINTS
# ---------------------------------------------------------------------------

try:
    import exchange_adapter as exch_mod
except ImportError:
    exch_mod = None


@app.get("/exchanges")
async def list_exchanges():
    if not exch_mod:
        return JSONResponse({"error": "exchange_adapter not available"}, 404)
    return JSONResponse({"exchanges": exch_mod.list_exchanges()})


@app.post("/exchange/price")
async def exchange_price(request: Request):
    if not exch_mod:
        return JSONResponse({"error": "exchange_adapter not available"}, 404)
    body = await request.json()
    exchange = body.get("exchange", "binance")
    symbol = body.get("symbol", "BTCUSDT")
    try:
        adapter = exch_mod.create_adapter(exchange)
        price = adapter.get_price(symbol)
        return JSONResponse({"exchange": exchange, "symbol": symbol, "price": price})
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, 400)


@app.post("/exchange/ping")
async def exchange_ping(request: Request):
    if not exch_mod:
        return JSONResponse({"error": "exchange_adapter not available"}, 404)
    body = await request.json()
    exchange = body.get("exchange", "binance")
    try:
        adapter = exch_mod.create_adapter(exchange)
        ok = adapter.ping()
        return JSONResponse({"exchange": exchange, "alive": ok})
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, 400)


if __name__ == "__main__":
    import socket
    # Verificar si el puerto ya esta en uso antes de arrancar
    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _sock.bind(("0.0.0.0", 9000))
        _sock.close()
    except OSError:
        _sock.close()
        print("ERROR: El puerto 9000 ya esta en uso. ¿Ya hay un dashboard corriendo?")
        print("Usa: Get-Process -Name py,python | ... para verificar.")
        import sys
        sys.exit(1)

    try:
        uvicorn.run("dashboard:app", host="0.0.0.0", port=9000, reload=False)
    except KeyboardInterrupt:
        print("Dashboard detenido por el usuario.")
    except Exception as _exc:
        import traceback, logging as _lg
        _lg.basicConfig(filename="_dash_crash.log", level=_lg.ERROR)
        _lg.error("Dashboard crash: %s\n%s", _exc, traceback.format_exc())
        print(f"CRASH: {_exc}  — ver _dash_crash.log")
        import sys
        sys.exit(1)

