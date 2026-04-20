"""
Bot de Scalping para Binance.
Estrategia: EMA + RSI + MACD
Pares: BTC/USDT y ETH/USDT
Modos: paper y real (Binance)
"""

import datetime
import csv
import hashlib
import hmac
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

try:
    import intelligence_engine
except ImportError:
    intelligence_engine = None

try:
    import telegram_notifier
except ImportError:
    telegram_notifier = None

try:
    import rl_agent
except ImportError:
    rl_agent = None

from config_defaults import (
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_MULTI_SOURCE_CONFIG,
    DEFAULT_SYMBOLS,
    normalize_runtime_config,
)

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────
load_dotenv()

TRADING_MODE = os.getenv("TRADING_MODE", "paper").strip().lower()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").strip().lower() in ("true", "1", "yes")

SYMBOLS = DEFAULT_SYMBOLS.copy()
ACTIVE_SYMBOLS = SYMBOLS.copy()
INTERVAL = "1m"
PAPER_BALANCE = 100.0
TARGET_USDT = 120.0
DEFAULT_WALLET_BALANCE = 100.0
TRAILING_STOP_PCT = 0.004
TAKE_PROFIT_PCT = 0.008
TRADE_PCT = 0.95

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 72
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BASE_URL_LIVE = "https://api.binance.com"
BASE_URL_TESTNET = "https://testnet.binance.vision"
BASE_URL = BASE_URL_TESTNET if BINANCE_TESTNET else BASE_URL_LIVE
BOT_STATE_FILE = "bot_state.json"
RUNTIME_CONFIG_FILE = "runtime_config.json"
BOT_EVENTS_FILE = "bot_events.log"
ML_DATASET_FILE = "ml_dataset.csv"
MULTI_SOURCE_CONFIG_FILE = "multi_source_config.json"
AI_MODEL_FILE = "ai_model.json"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
TRADE_LOGGER = logging.getLogger("trades")
if not TRADE_LOGGER.handlers:
    TRADE_LOGGER.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        "trades.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    TRADE_LOGGER.addHandler(file_handler)

EVENT_LOGGER = logging.getLogger("events")
if not EVENT_LOGGER.handlers:
    EVENT_LOGGER.setLevel(logging.INFO)
    event_handler = RotatingFileHandler(
        BOT_EVENTS_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    event_handler.setFormatter(logging.Formatter("%(message)s"))
    EVENT_LOGGER.addHandler(event_handler)

STATE_LOCK = threading.Lock()
DATASET_LOCK = threading.Lock()
STATE_WRITE_RETRIES = 6
BOT_STATE = {
    "updated_at": "",
    "mode": TRADING_MODE,
    "runtime_config": DEFAULT_RUNTIME_CONFIG.copy(),
    "pairs": {},
}

_AI_MODEL_CACHE = {
    "mtime": None,
    "model": None,
}


class BinanceAPIError(Exception):
    def __init__(self, message: str, code=None):
        super().__init__(message)
        self.code = code


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def now_iso() -> str:
    return datetime.datetime.now().isoformat()


def log(msg: str):
    text = f"[{now()}] {msg}"
    print(text)
    EVENT_LOGGER.info(text)


def base_asset(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return symbol[:-4]
    return symbol


def write_bot_state() -> bool:
    with STATE_LOCK:
        BOT_STATE["updated_at"] = now_iso()
        if intelligence_engine:
            try:
                BOT_STATE["intel_state"] = intelligence_engine.get_intelligence_state()
            except Exception:
                pass
        snapshot = json.dumps(BOT_STATE, ensure_ascii=False, indent=2)

    temp_file = f"{BOT_STATE_FILE}.{threading.get_ident()}.tmp"
    last_exc = None

    for attempt in range(STATE_WRITE_RETRIES):
        try:
            with open(temp_file, "w", encoding="utf-8") as fh:
                fh.write(snapshot)
            os.replace(temp_file, BOT_STATE_FILE)
            return True
        except PermissionError as exc:
            last_exc = exc
        except OSError as exc:
            last_exc = exc
            # WinError 32/13 aparece cuando otro proceso tiene lock transitorio.
            if getattr(exc, "winerror", None) not in {13, 32}:
                break
        finally:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

        time.sleep(0.05 * (attempt + 1))

    if last_exc:
        log(f"[STATE] No se pudo escribir {BOT_STATE_FILE} tras {STATE_WRITE_RETRIES} intentos: {last_exc}")
    return False


def update_pair_state(symbol: str, payload: dict):
    with STATE_LOCK:
        pair_state = BOT_STATE["pairs"].setdefault(symbol, {})
        pair_state.update(payload)
    write_bot_state()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_ai_model(force: bool = False) -> dict:
    try:
        if not os.path.exists(AI_MODEL_FILE):
            _AI_MODEL_CACHE["mtime"] = None
            _AI_MODEL_CACHE["model"] = None
            return {}

        mtime = os.path.getmtime(AI_MODEL_FILE)
        if not force and _AI_MODEL_CACHE["model"] is not None and _AI_MODEL_CACHE["mtime"] == mtime:
            return _AI_MODEL_CACHE["model"]

        with open(AI_MODEL_FILE, "r", encoding="utf-8") as fh:
            model = json.load(fh)

        # Validacion minima de estructura
        if not isinstance(model.get("weights", []), list):
            return {}
        if not isinstance(model.get("feature_names", []), list):
            return {}

        _AI_MODEL_CACHE["mtime"] = mtime
        _AI_MODEL_CACHE["model"] = model
        return model
    except Exception as exc:
        log(f"[AI] No se pudo cargar modelo {AI_MODEL_FILE}: {exc}")
        return {}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ai_predict_probability(model: dict, features: dict) -> float:
    model_type = model.get("model_type", "logistic")
    if model_type == "ensemble":
        return _predict_ensemble(model, features)
    return _predict_logistic(model, features)


def _predict_logistic(model: dict, features: dict) -> float:
    names = model.get("feature_names", [])
    weights = model.get("weights", [])
    means = model.get("means", [])
    stds = model.get("stds", [])
    bias = _safe_float(model.get("bias", 0.0), 0.0)

    if not names or len(names) != len(weights):
        return 0.5

    score = bias
    for i, name in enumerate(names):
        raw = _safe_float(features.get(name, 0.0), 0.0)
        mean = _safe_float(means[i], 0.0) if i < len(means) else 0.0
        std = _safe_float(stds[i], 1.0) if i < len(stds) else 1.0
        if abs(std) < 1e-9:
            std = 1.0
        norm = (raw - mean) / std
        score += _safe_float(weights[i], 0.0) * norm
    return _sigmoid(score)


def _predict_ensemble(model: dict, features: dict) -> float:
    """Predicción con ensemble de gradient boosted stumps."""
    stumps = model.get("stumps", [])
    names = model.get("feature_names", [])
    base_score = _safe_float(model.get("base_score", 0.5), 0.5)
    lr = _safe_float(model.get("learning_rate", 0.1), 0.1)

    if not stumps:
        return 0.5

    raw_score = math.log(base_score / max(1 - base_score, 1e-9))
    for stump in stumps:
        feat_name = stump.get("feature", "")
        threshold = _safe_float(stump.get("threshold", 0.0), 0.0)
        left_val = _safe_float(stump.get("left_value", 0.0), 0.0)
        right_val = _safe_float(stump.get("right_value", 0.0), 0.0)
        feat_val = _safe_float(features.get(feat_name, 0.0), 0.0)
        if feat_val <= threshold:
            raw_score += lr * left_val
        else:
            raw_score += lr * right_val
    return _sigmoid(raw_score)


def build_ai_features(price: float, ema_fast: float, ema_slow: float, rsi: float, macd_line: float, macd_signal_line: float, macd_hist: float, raw_signal: str, in_position: bool, update_mode: str, candles: list = None, closes: list = None) -> dict:
    price_safe = max(price, 1e-9)
    ema_diff_pct = (ema_fast - ema_slow) / price_safe
    feats = {
        "ema_diff_pct": ema_diff_pct,
        "rsi_norm": (rsi - 50.0) / 50.0,
        "macd": macd_line,
        "macd_signal": macd_signal_line,
        "macd_hist": macd_hist,
        "signal_buy": 1.0 if raw_signal == "BUY" else 0.0,
        "in_position": 1.0 if in_position else 0.0,
        "update_mode_tick": 1.0 if str(update_mode).lower() == "tick" else 0.0,
    }
    # Features V3
    if candles:
        atr = calc_atr(candles)
        feats["atr_norm"] = atr / price_safe
        feats["volume_ratio"] = calc_volume_ratio(candles)
    else:
        feats["atr_norm"] = 0.0
        feats["volume_ratio"] = 1.0
    if closes:
        feats["bb_pct"] = calc_bollinger_pct(closes)
    else:
        feats["bb_pct"] = 0.5
    # Hora encodificada (ciclo 24h)
    hour = time.localtime().tm_hour + time.localtime().tm_min / 60.0
    feats["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    feats["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    return feats


def load_runtime_config() -> dict:
    if not os.path.exists(RUNTIME_CONFIG_FILE):
        save_runtime_config(DEFAULT_RUNTIME_CONFIG)
        return DEFAULT_RUNTIME_CONFIG.copy()
    try:
        with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as fh:
            return normalize_runtime_config(json.load(fh))
    except Exception:
        save_runtime_config(DEFAULT_RUNTIME_CONFIG)
        return DEFAULT_RUNTIME_CONFIG.copy()


def save_runtime_config(payload: dict):
    cfg = normalize_runtime_config(payload)
    with open(RUNTIME_CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)


def get_runtime_config() -> dict:
    cfg = load_runtime_config()
    with STATE_LOCK:
        BOT_STATE["runtime_config"] = cfg
    return cfg


def log_trade(symbol: str, trade_type: str, price: float, pnl: float, reason: str):
    dt = datetime.datetime.now()
    TRADE_LOGGER.info(
        f"{dt.strftime('%Y-%m-%d')},{dt.strftime('%H:%M:%S')},symbol={symbol},tipo={trade_type},precio={price:.2f},pnl={pnl:.2f},razon={reason}"
    )


def ensure_dataset_file():
    if os.path.exists(ML_DATASET_FILE):
        return
    headers = [
        "timestamp",
        "symbol",
        "price",
        "ema_fast",
        "ema_slow",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "signal_regla",
        "in_position",
        "balance",
        "future_return_n",
        "update_mode",
        "atr_norm",
        "bb_pct",
        "volume_ratio",
        "hour_sin",
        "hour_cos",
    ]
    with open(ML_DATASET_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)


def append_dataset_row(row: dict):
    ensure_dataset_file()
    with DATASET_LOCK:
        with open(ML_DATASET_FILE, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    row.get("timestamp", ""),
                    row.get("symbol", ""),
                    row.get("price", 0.0),
                    row.get("ema_fast", 0.0),
                    row.get("ema_slow", 0.0),
                    row.get("rsi", 50.0),
                    row.get("macd", 0.0),
                    row.get("macd_signal", 0.0),
                    row.get("macd_hist", 0.0),
                    row.get("signal_regla", "HOLD"),
                    row.get("in_position", False),
                    row.get("balance", 0.0),
                    row.get("future_return_n", ""),
                    row.get("update_mode", "candle"),
                    row.get("atr_norm", 0.0),
                    row.get("bb_pct", 0.5),
                    row.get("volume_ratio", 1.0),
                    row.get("hour_sin", 0.0),
                    row.get("hour_cos", 0.0),
                ]
            )


def normalize_multi_source_config(payload: dict) -> dict:
    cfg = DEFAULT_MULTI_SOURCE_CONFIG.copy()
    cfg.update(payload or {})
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["threshold_pct"] = min(max(float(cfg.get("threshold_pct", 0.15)), 0.01), 5.0)
    action = str(cfg.get("action", "alert")).strip().lower()
    cfg["action"] = action if action in {"alert", "reduce", "pause"} else "alert"
    cfg["refresh_seconds"] = min(max(int(float(cfg.get("refresh_seconds", 5))), 2), 30)
    return cfg


def load_multi_source_config() -> dict:
    if not os.path.exists(MULTI_SOURCE_CONFIG_FILE):
        return DEFAULT_MULTI_SOURCE_CONFIG.copy()
    try:
        with open(MULTI_SOURCE_CONFIG_FILE, "r", encoding="utf-8") as fh:
            return normalize_multi_source_config(json.load(fh))
    except Exception:
        return DEFAULT_MULTI_SOURCE_CONFIG.copy()


def _fetch_price_binance(symbol: str) -> dict:
    t0 = time.perf_counter()
    url = f"{BASE_URL}/api/v3/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=3)
    resp.raise_for_status()
    latency = (time.perf_counter() - t0) * 1000
    return {"name": "Binance", "price": float(resp.json()["price"]), "latency_ms": latency, "ok": True}


def _fetch_price_coinbase(symbol: str) -> dict:
    base = "BTC" if symbol.startswith("BTC") else "ETH"
    product = f"{base}-USD"
    t0 = time.perf_counter()
    url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()
    latency = (time.perf_counter() - t0) * 1000
    return {"name": "Coinbase", "price": float(resp.json()["price"]), "latency_ms": latency, "ok": True}


def _fetch_price_kraken(symbol: str) -> dict:
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


def get_multi_source_snapshot(symbol: str) -> dict:
    cfg = load_multi_source_config()
    fetchers = [_fetch_price_binance]
    if symbol.startswith("BTC") or symbol.startswith("ETH"):
        fetchers.extend([_fetch_price_coinbase, _fetch_price_kraken])
    sources = []

    for fn in fetchers:
        try:
            sources.append(fn(symbol))
        except Exception:
            name = fn.__name__.replace("_fetch_price_", "").capitalize()
            sources.append({"name": name, "price": None, "latency_ms": None, "ok": False})

    binance_price = next((s["price"] for s in sources if s["name"] == "Binance" and s.get("price") is not None), None)
    valid_prices = [s["price"] for s in sources if s.get("price") is not None]

    reference_price = sum(valid_prices) / len(valid_prices) if valid_prices else 0.0
    max_dev = 0.0
    for src in sources:
        if src.get("price") is None or not binance_price:
            src["deviation_pct"] = None
            continue
        dev = abs(src["price"] - binance_price) / max(binance_price, 1e-9) * 100
        src["deviation_pct"] = dev
        max_dev = max(max_dev, dev)

    recommendation = "ok"
    if cfg["enabled"] and max_dev >= cfg["threshold_pct"]:
        recommendation = cfg["action"]

    return {
        "symbol": symbol,
        "reference_price": reference_price,
        "max_deviation_pct": max_dev,
        "recommendation": recommendation,
        "config": cfg,
        "sources": sources,
        "updated_at": now_iso(),
    }


# ─────────────────────────────────────────────
#  CLIENTE BINANCE PUBLICO
# ─────────────────────────────────────────────

def _api_retry(fn, *args, max_retries=3, base_delay=1.0, **kwargs):
    """Ejecuta fn con reintentos y backoff exponencial."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (requests.exceptions.RequestException, ValueError) as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            log(f"API retry {attempt+1}/{max_retries} tras error: {exc}. Esperando {delay:.1f}s...")
            time.sleep(delay)


def get_klines(symbol: str, interval: str, limit: int = 100) -> list:
    def _fetch():
        url = f"{BASE_URL}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    return _api_retry(_fetch)


def get_price(symbol: str) -> float:
    def _fetch():
        url = f"{BASE_URL}/api/v3/ticker/price"
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["price"])
    return _api_retry(_fetch)


def parse_candle(kline: list) -> dict:
    return {
        "x": int(kline[0]),
        "o": float(kline[1]),
        "h": float(kline[2]),
        "l": float(kline[3]),
        "c": float(kline[4]),
        "v": float(kline[5]) if len(kline) > 5 else 0.0,
    }


def upsert_tick_candle(candles: deque, price: float):
    candle_ts = int((time.time() // 60) * 60000)
    if not candles or candle_ts != candles[-1]["x"]:
        candles.append({"x": candle_ts, "o": price, "h": price, "l": price, "c": price, "v": 0.0})
        return

    candles[-1]["h"] = max(candles[-1]["h"], price)
    candles[-1]["l"] = min(candles[-1]["l"], price)
    candles[-1]["c"] = price


# ─────────────────────────────────────────────
#  INDICADORES TECNICOS
# ─────────────────────────────────────────────

def calc_ema(prices: list, period: int) -> float:
    if not prices:
        return 0.0
    if len(prices) < period:
        return sum(prices) / len(prices)
    k = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def calc_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(closes: list, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0

    k_fast = 2 / (fast + 1)
    k_slow = 2 / (slow + 1)

    ema_fast = closes[0]
    ema_slow = closes[0]
    macd_series = []

    for price in closes:
        ema_fast = price * k_fast + ema_fast * (1 - k_fast)
        ema_slow = price * k_slow + ema_slow * (1 - k_slow)
        macd_series.append(ema_fast - ema_slow)

    k_signal = 2 / (signal + 1)
    signal_line = macd_series[0]
    for value in macd_series[1:]:
        signal_line = value * k_signal + signal_line * (1 - k_signal)

    macd_line = macd_series[-1]
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_ema_series(closes: list, period: int) -> list:
    out = []
    for i in range(len(closes)):
        out.append(calc_ema(closes[: i + 1], period))
    return out


def calc_rsi_series(closes: list, period: int = 14) -> list:
    out = []
    for i in range(len(closes)):
        out.append(calc_rsi(closes[: i + 1], period))
    return out


def calc_atr(candles: list, period: int = 14) -> float:
    """Average True Range sobre velas OHLC."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i].get("h", candles[i].get("high", 0))
        l = candles[i].get("l", candles[i].get("low", 0))
        prev_c = candles[i - 1].get("c", candles[i - 1].get("close", 0))
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)
    atr = sum(trs[-period:]) / period
    return atr


def calc_bollinger_pct(closes: list, period: int = 20) -> float:
    """Bollinger %B: posición del precio entre bandas (0=lower, 1=upper)."""
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    sma = sum(window) / period
    variance = sum((x - sma) ** 2 for x in window) / period
    std = variance ** 0.5
    if std < 1e-9:
        return 0.5
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = upper - lower
    if band_width < 1e-9:
        return 0.5
    return (closes[-1] - lower) / band_width


def calc_volume_ratio(candles, period: int = 20) -> float:
    """Ratio del volumen actual vs promedio (1.0 = normal, >1 = alto)."""
    if len(candles) < period:
        return 1.0
    recent = list(candles)[-period:]
    volumes = [c.get("v", c.get("volume", 0)) for c in recent]
    avg = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
    if avg < 1e-9:
        return 1.0
    return volumes[-1] / avg


# ─────────────────────────────────────────────
#  SENALES DE TRADING
# ─────────────────────────────────────────────

def get_signal(closes: list) -> str:
    min_len = max(EMA_SLOW + 5, MACD_SLOW + MACD_SIGNAL + 2)
    if len(closes) < min_len:
        return "HOLD"

    ema_fast_now = calc_ema(closes, EMA_FAST)
    ema_slow_now = calc_ema(closes, EMA_SLOW)
    ema_fast_prev = calc_ema(closes[:-1], EMA_FAST)
    ema_slow_prev = calc_ema(closes[:-1], EMA_SLOW)
    rsi = calc_rsi(closes, RSI_PERIOD)
    macd_now, macd_signal_now, macd_hist_now = calc_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_prev, macd_signal_prev, macd_hist_prev = calc_macd(closes[:-1], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
    bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now
    macd_positive = macd_now > macd_signal_now
    macd_hist_rising = macd_hist_now >= macd_hist_prev
    trend_up = ema_fast_now > ema_slow_now
    macd_negative_cross = macd_prev >= macd_signal_prev and macd_now < macd_signal_now
    price_above_slow = closes[-1] > ema_slow_now

    # Entrada por continuidad: permite operar cuando la tendencia ya esta activa
    # aunque no haya cruce exacto en este tick.
    continuation_buy = trend_up and macd_positive and macd_hist_rising and (RSI_OVERSOLD < rsi < 68) and price_above_slow

    if (bullish_cross and rsi < RSI_OVERBOUGHT and macd_positive and price_above_slow) or continuation_buy:
        return "BUY"
    if bearish_cross or rsi > RSI_OVERBOUGHT or macd_negative_cross:
        return "SELL"
    return "HOLD"


def get_htf_trend(closes_htf: list) -> str:
    """Determina tendencia en timeframe superior (5m/15m).

    Retorna 'up', 'down' o 'neutral'.
    Solo necesita EMA fast/slow para confirmar dirección.
    """
    min_len = EMA_SLOW + 5
    if len(closes_htf) < min_len:
        return "neutral"
    ema_f = calc_ema(closes_htf, EMA_FAST)
    ema_s = calc_ema(closes_htf, EMA_SLOW)
    rsi = calc_rsi(closes_htf, RSI_PERIOD)
    if ema_f > ema_s and rsi > 40:
        return "up"
    elif ema_f < ema_s and rsi < 60:
        return "down"
    return "neutral"


# ─────────────────────────────────────────────
#  RATE LIMITER
# ─────────────────────────────────────────────

class RateLimiter:
    """Controla requests por ventana de tiempo para respetar límites de Binance."""

    def __init__(self, max_requests: int = 1100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: list = []
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now_ts = time.time()
            cutoff = now_ts - self.window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.max_requests:
                sleep_time = self._timestamps[0] - cutoff + 0.1
                log(f"[RATE] Limite alcanzado, esperando {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self._timestamps.append(time.time())


_rate_limiter = RateLimiter()


# ─────────────────────────────────────────────
#  WALLETS
# ─────────────────────────────────────────────

class BaseWallet:
    def __init__(self, symbol: str, initial_usdt: float, config_getter):
        self.symbol = symbol
        self.config_getter = config_getter
        self.usdt = initial_usdt
        self.asset = 0.0
        self.trades = []
        self.entry_price = None
        self.peak_price = None
        self.trailing_stop_price = None
        self.active_take_profit_pct = None

    @property
    def in_position(self) -> bool:
        return self.asset > 0

    def _record_trade(self, trade: dict):
        self.trades.append(trade)
        log_trade(self.symbol, trade["type"], trade["price"], trade.get("pnl", 0.0), trade.get("reason", ""))
        # Persistir en SQLite
        try:
            import db
            db.insert_trade(
                symbol=self.symbol,
                side=trade["type"],
                price=trade["price"],
                qty=trade.get("qty", 0),
                pnl=trade.get("pnl", 0),
                reason=trade.get("reason", ""),
                balance_after=self.total_value(trade["price"]),
            )
        except Exception:
            pass
        # RL: registrar recompensa en SELL trades
        if trade["type"] == "SELL" and rl_agent and hasattr(self, "_last_rl_state"):
            try:
                pnl_pct = trade.get("pnl", 0) / max(self.total_value(trade["price"]), 1)
                reward = rl_agent.calculate_reward(pnl_pct, 2, pnl_pct > 0)
                rl_agent.record_experience(
                    *self._last_rl_state, 2, reward,
                    *self._last_rl_state[:3],
                    self._last_rl_state[3],
                    False,
                    self._last_rl_state[5],
                )
            except Exception:
                pass

    def _reset_position(self):
        self.asset = 0.0
        self.entry_price = None
        self.peak_price = None
        self.trailing_stop_price = None
        self.active_take_profit_pct = None

    def update_trailing_stop(self, price: float):
        if not self.in_position:
            return
        cfg = self.config_getter()
        trailing_stop_pct = cfg["trailing_stop_pct"]
        # Hard cap: trailing stop nunca puede estar más abajo que entry * (1 - max_trailing_loss_pct)
        max_trailing_loss_pct = cfg.get("max_trailing_loss_pct", 0.003)
        if self.peak_price is None or price > self.peak_price:
            self.peak_price = price
            new_stop = self.peak_price * (1 - trailing_stop_pct)
            # Floor: no dejar que el stop caiga más allá del cap de pérdida máxima
            if self.entry_price:
                floor_stop = self.entry_price * (1 - max_trailing_loss_pct)
                new_stop = max(new_stop, floor_stop)
            if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop

    def total_value(self, price: float) -> float:
        return self.usdt + self.asset * price

    def summary(self, price: float):
        mode = "PAPER" if isinstance(self, PaperWallet) else "REAL"
        wins = [t for t in self.trades if t.get("pnl", 0) > 0]
        total_trades = len([t for t in self.trades if t["type"] == "SELL"])
        print("\n" + "=" * 50)
        print(f"       RESUMEN FINAL {self.symbol} ({mode})")
        print("=" * 50)
        print(f"  Trades totales : {total_trades}")
        print(f"  Trades ganados : {len(wins)}")
        print(f"  Win rate       : {len(wins) / max(total_trades, 1) * 100:.1f}%")
        print(f"  Balance final  : ${self.total_value(price):.2f}")
        print("=" * 50)


class PaperWallet(BaseWallet):
    def __init__(self, symbol: str, initial_usdt: float, config_getter):
        super().__init__(symbol, initial_usdt, config_getter)

    def buy(self, price: float, trade_pct_override: float = None):
        cfg = self.config_getter()
        trade_pct = trade_pct_override if trade_pct_override is not None else cfg["trade_pct"]
        fee_pct = cfg.get("fee_pct", 0.001)
        amount_usdt = self.usdt * trade_pct
        if amount_usdt <= 0:
            return None
        raw_qty = amount_usdt / price
        self.asset = raw_qty * (1 - fee_pct)  # Descuento comisión
        self.usdt -= amount_usdt
        self.entry_price = price
        self.peak_price = price
        self.trailing_stop_price = None

        trade = {
            "type": "BUY",
            "symbol": self.symbol,
            "price": price,
            "qty": self.asset,
            "pnl": 0.0,
            "reason": "entrada",
            "time": now(),
        }
        self._record_trade(trade)
        log(
            f"  [{self.symbol}] COMPRA | Precio: ${price:,.2f} | Qty: {self.asset:.6f} | USDT restante: ${self.usdt:.2f}"
        )
        if telegram_notifier:
            telegram_notifier.notify_trade(self.symbol, "BUY", price)
        return trade

    def sell(self, price: float, reason: str = "senal"):
        if self.asset <= 0:
            return None
        fee_pct = self.config_getter().get("fee_pct", 0.001)
        gross_proceeds = self.asset * price
        proceeds = gross_proceeds * (1 - fee_pct)  # Descuento comisión
        pnl = proceeds - (self.asset * self.entry_price)
        self.usdt += proceeds
        trade = {
            "type": "SELL",
            "symbol": self.symbol,
            "price": price,
            "qty": self.asset,
            "pnl": pnl,
            "reason": reason,
            "time": now(),
        }
        self._record_trade(trade)
        log(
            f"  [{self.symbol}] VENTA  | Precio: ${price:,.2f} | PnL: ${pnl:+.2f} | Balance: ${self.usdt:.2f} | Razon: {reason}"
        )
        if telegram_notifier:
            telegram_notifier.notify_trade(self.symbol, "SELL", price, pnl, reason)
        self._reset_position()
        return trade


class RealWallet(BaseWallet):
    def __init__(self, symbol: str, initial_usdt: float, api_key: str, api_secret: str, config_getter):
        super().__init__(symbol, initial_usdt, config_getter)
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.asset_symbol = base_asset(symbol)
        self.initial_usdt = initial_usdt

        self.usdt = 0.0
        self.asset = 0.0
        self.refresh_balances()
        self._reconcile_position()

    def _reconcile_position(self):
        """Detecta si hay posición abierta residual al iniciar (crash recovery)."""
        if self.asset > 0:
            try:
                price = get_price(self.symbol)
                if price and price > 0:
                    self.entry_price = price
                    self.peak_price = price
                    log(f"[{self.symbol}] Reconciliación: posición abierta detectada ({self.asset:.6f} @ ~${price:,.2f})")
            except Exception as exc:
                log(f"[{self.symbol}] Error en reconciliación: {exc}")

    def signed_request(self, method: str, endpoint: str, params: dict) -> dict:
        _rate_limiter.acquire()
        payload = dict(params)
        payload["timestamp"] = int(time.time() * 1000)
        payload["recvWindow"] = 5000
        query = urlencode(payload)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload["signature"] = signature
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{BASE_URL}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=payload,
                headers=headers,
                timeout=10,
            )
        except requests.Timeout as exc:
            raise BinanceAPIError(f"Timeout en {endpoint}") from exc
        except requests.RequestException as exc:
            raise BinanceAPIError(f"Error de red en {endpoint}: {exc}") from exc

        if response.status_code >= 400:
            try:
                data = response.json()
            except ValueError:
                raise BinanceAPIError(f"HTTP {response.status_code}: {response.text}")
            raise BinanceAPIError(data.get("msg", "Error Binance"), data.get("code"))

        try:
            return response.json()
        except ValueError as exc:
            raise BinanceAPIError("Respuesta no JSON de Binance") from exc

    def get_balance(self, asset: str) -> float:
        data = self.signed_request("GET", "/api/v3/account", {})
        for item in data.get("balances", []):
            if item.get("asset") == asset:
                free = float(item.get("free", "0"))
                locked = float(item.get("locked", "0"))
                return free + locked
        return 0.0

    def _get_symbol_filters(self, symbol: str) -> dict:
        """Obtiene filtros del par (min_notional, step_size, min_qty) de exchangeInfo."""
        try:
            _rate_limiter.acquire()
            resp = requests.get(f"{BASE_URL}/api/v3/exchangeInfo", params={"symbol": symbol}, timeout=10)
            if resp.status_code != 200:
                return {}
            info = resp.json()
            sym_info = next((s for s in info.get("symbols", []) if s["symbol"] == symbol), None)
            if not sym_info:
                return {}
            filters = {}
            for f in sym_info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    filters["step_size"] = float(f["stepSize"])
                    filters["min_qty"] = float(f["minQty"])
                elif f["filterType"] == "NOTIONAL":
                    filters["min_notional"] = float(f.get("minNotional", 0))
                elif f["filterType"] == "MIN_NOTIONAL":
                    filters["min_notional"] = float(f.get("minNotional", 0))
            return filters
        except Exception:
            return {}

    def _adjust_quantity(self, qty: float, step_size: float) -> float:
        """Ajusta cantidad al step_size de Binance (redondeo hacia abajo)."""
        if step_size <= 0:
            return qty
        precision = max(0, int(round(-math.log10(step_size))))
        return math.floor(qty * (10 ** precision)) / (10 ** precision)

    def execute_order(self, side: str, symbol: str, quantity: float) -> dict:
        quantity = max(quantity, 0.0)
        if quantity <= 0:
            raise BinanceAPIError("Cantidad invalida para la orden")

        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.6f}",
        }
        result = self.signed_request("POST", "/api/v3/order", params)

        order_id = result.get("orderId", "?")
        status = result.get("status", "?")
        executed_qty = result.get("executedQty", "0")
        log(f"  [{symbol}] Orden {side} #{order_id}: status={status} qty={executed_qty}")

        return result

    def _verify_order_status(self, symbol: str, order_id) -> dict:
        """Consulta estado de una orden (para verificar si se ejecuto tras error ambiguo)."""
        try:
            return self.signed_request("GET", "/api/v3/order", {
                "symbol": symbol,
                "orderId": order_id,
            })
        except BinanceAPIError:
            return {}

    def refresh_balances(self):
        self.usdt = self.get_balance("USDT")
        self.asset = self.get_balance(self.asset_symbol)

    def buy(self, price: float, trade_pct_override: float = None):
        try:
            cfg = self.config_getter()
            self.refresh_balances()
            effective_trade_pct = trade_pct_override if trade_pct_override is not None else cfg["trade_pct"]
            spend_usdt = self.usdt * effective_trade_pct

            # Validar min notional
            filters = self._get_symbol_filters(self.symbol)
            min_notional = filters.get("min_notional", 5.0)
            if spend_usdt < min_notional:
                log(f"[{self.symbol}] BUY rechazado: ${spend_usdt:.2f} < min_notional ${min_notional:.2f}")
                return None

            qty = spend_usdt / price

            # Ajustar al step_size
            step_size = filters.get("step_size", 0)
            if step_size > 0:
                qty = self._adjust_quantity(qty, step_size)
                min_qty = filters.get("min_qty", 0)
                if qty < min_qty:
                    log(f"[{self.symbol}] BUY rechazado: qty {qty:.8f} < min_qty {min_qty:.8f}")
                    return None

            order = self.execute_order("BUY", self.symbol, qty)
            executed_qty = float(order.get("executedQty", qty))

            fills = order.get("fills", [])
            fill_price = price
            if fills:
                fill_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / max(
                    sum(float(f["qty"]) for f in fills), 1e-12
                )

            self.refresh_balances()
            self.entry_price = fill_price
            self.peak_price = fill_price
            self.trailing_stop_price = None

            trade = {
                "type": "BUY",
                "symbol": self.symbol,
                "price": fill_price,
                "qty": executed_qty,
                "pnl": 0.0,
                "reason": "entrada",
                "time": now(),
            }
            self._record_trade(trade)
            log(
                f"  [{self.symbol}] COMPRA REAL | Precio: ${fill_price:,.2f} | Qty: {executed_qty:.6f} | USDT: ${self.usdt:.2f}"
            )
            return trade
        except BinanceAPIError as exc:
            log(f"[{self.symbol}] Error BUY real: {exc}")
            return None

    def sell(self, price: float, reason: str = "senal"):
        try:
            self.refresh_balances()
            qty = self.asset

            # Ajustar al step_size
            filters = self._get_symbol_filters(self.symbol)
            step_size = filters.get("step_size", 0)
            if step_size > 0:
                qty = self._adjust_quantity(qty, step_size)
            if qty <= 0:
                log(f"[{self.symbol}] SELL rechazado: qty ajustada = 0")
                return None

            order = self.execute_order("SELL", self.symbol, qty)
            executed_qty = float(order.get("executedQty", qty))

            fills = order.get("fills", [])
            fill_price = price
            if fills:
                fill_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / max(
                    sum(float(f["qty"]) for f in fills), 1e-12
                )

            pnl = 0.0
            if self.entry_price is not None:
                pnl = (fill_price - self.entry_price) * executed_qty

            self.refresh_balances()

            trade = {
                "type": "SELL",
                "symbol": self.symbol,
                "price": fill_price,
                "qty": executed_qty,
                "pnl": pnl,
                "reason": reason,
                "time": now(),
            }
            self._record_trade(trade)
            log(
                f"  [{self.symbol}] VENTA REAL  | Precio: ${fill_price:,.2f} | PnL: ${pnl:+.2f} | USDT: ${self.usdt:.2f} | Razon: {reason}"
            )

            self._reset_position()
            return trade
        except BinanceAPIError as exc:
            log(f"[{self.symbol}] Error SELL real: {exc}")
            return None

    def total_value(self, price: float) -> float:
        try:
            self.refresh_balances()
        except BinanceAPIError as exc:
            log(f"[{self.symbol}] Error al refrescar balances: {exc}")
        return self.usdt + self.asset * price


# ─────────────────────────────────────────────
#  EJECUCION DEL BOT
# ─────────────────────────────────────────────

def build_wallet(symbol: str, initial_balance: float):
    if TRADING_MODE == "real":
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise RuntimeError("Faltan BINANCE_API_KEY/BINANCE_API_SECRET para modo real")
        return RealWallet(symbol, initial_balance, BINANCE_API_KEY, BINANCE_API_SECRET, get_runtime_config)
    return PaperWallet(symbol, initial_balance, get_runtime_config)


def emit_runtime_state(
    symbol: str,
    wallet,
    price: float,
    signal: str,
    rsi: float,
    candles: list,
    closes: list,
    event_type: str,
    initial_pair_balance: float,
):
    cfg = get_runtime_config()
    ema_fast_series = calc_ema_series(closes, EMA_FAST)[-50:]
    ema_slow_series = calc_ema_series(closes, EMA_SLOW)[-50:]
    rsi_series = calc_rsi_series(closes, RSI_PERIOD)[-50:]

    balance = wallet.total_value(price)
    gain_pct = ((balance - initial_pair_balance) / max(initial_pair_balance, 1e-9)) * 100

    update_pair_state(
        symbol,
        {
            "symbol": symbol,
            "price": price,
            "signal": signal,
            "rsi": rsi,
            "in_position": wallet.in_position,
            "balance": balance,
            "initial_balance": initial_pair_balance,
            "gain_pct": gain_pct,
            "candles": candles[-50:],
            "ema_fast": ema_fast_series,
            "ema_slow": ema_slow_series,
            "rsi_series": rsi_series,
            "trades": wallet.trades[-20:],
            "event": event_type,
            "runtime_config": cfg,
            "updated_at": now_iso(),
        },
    )


def run_symbol(symbol: str, initial_balance: float, target_balance: float, total_pairs: int):
    wallet = build_wallet(symbol, initial_balance)
    closes = deque(maxlen=max(EMA_SLOW + 60, MACD_SLOW + MACD_SIGNAL + 60))
    candles = deque(maxlen=120)
    iteration = 0
    previous_signal = "HOLD"
    previous_balance = wallet.total_value(get_price(symbol))
    buy_signal_streak = 0
    sell_signal_streak = 0
    next_trade_after = 0.0
    entry_ts = None
    ms_snapshot = None
    ms_last_refresh_ts = 0.0
    day_key = datetime.date.today().isoformat()
    day_pnl = 0.0
    day_trades = 0
    sells_last_hour = deque()
    kill_switch_announced = False
    drawdown_kill_announced = False
    trades_cap_announced = False
    ai_model_missing_announced = False
    # Multi-timeframe state
    htf_closes: list = []
    htf_last_refresh_ts = 0.0
    htf_trend = "neutral"

    try:
        klines = get_klines(symbol, INTERVAL, limit=120)
        for raw in klines:
            candle = parse_candle(raw)
            candles.append(candle)
            closes.append(candle["c"])

        cfg_boot = get_runtime_config()
        log(
            f"[{symbol}] Historial cargado ({len(closes)} velas). Iniciando loop... modo={cfg_boot['update_mode']} poll={cfg_boot['poll_seconds']}s"
        )

        while True:
            cfg = get_runtime_config()
            now_ts = time.time()

            # Si el optimizer removio este par de la config, detener el loop
            active_symbols = cfg.get("symbols", [])
            if symbol not in active_symbols:
                if wallet.in_position:
                    wallet.sell(closes[-1] if closes else 0, "par_removido_por_optimizer")
                    log(f"[{symbol}] Par removido por optimizer. Posicion cerrada.")
                log(f"[{symbol}] Par removido de la configuracion. Deteniendo loop.")
                break

            current_day = datetime.date.today().isoformat()
            if current_day != day_key:
                day_key = current_day
                day_pnl = 0.0
                day_trades = 0
                kill_switch_announced = False
                log(f"[{symbol}] Nuevo dia detectado. Reiniciando contador de perdida diaria.")

            while sells_last_hour and (now_ts - sells_last_hour[0]) > 3600:
                sells_last_hour.popleft()

            daily_kill_active = day_pnl <= -cfg["daily_loss_limit_usdt"]
            # Max drawdown kill switch
            current_value = wallet.total_value(closes[-1] if closes else 0)
            drawdown_from_initial = (initial_balance - current_value) / max(initial_balance, 1e-9)
            drawdown_kill_active = drawdown_from_initial >= cfg["max_drawdown_pct"]
            trades_per_hour_reached = len(sells_last_hour) >= cfg["max_trades_per_hour"]
            max_daily = cfg.get("max_trades_per_day", 0)
            trades_per_day_reached = max_daily > 0 and day_trades >= max_daily

            target_balance = cfg.get("pair_targets", {}).get(symbol, cfg["target_usdt"] / max(total_pairs, 1))
            latency_ms = 0.0
            if cfg["update_mode"] == "tick":
                t0 = time.perf_counter()
                price = get_price(symbol)
                latency_ms = (time.perf_counter() - t0) * 1000
                upsert_tick_candle(candles, price)
            else:
                t0 = time.perf_counter()
                last_candle_raw = get_klines(symbol, INTERVAL, limit=1)[0]
                latency_ms = (time.perf_counter() - t0) * 1000
                fresh_candle = parse_candle(last_candle_raw)
                if not candles or fresh_candle["x"] != candles[-1]["x"]:
                    candles.append(fresh_candle)
                else:
                    candles[-1] = fresh_candle
                price = fresh_candle["c"] if cfg["update_mode"] != "tick" else price

            # Derivar closes siempre de candles (1 close por vela de 1 min)
            close_list = [c["c"] for c in candles]
            raw_signal = get_signal(close_list)

            # ── Multi-timeframe filter ──
            if cfg.get("mtf_enabled", False) and raw_signal == "BUY":
                mtf_tf = cfg.get("mtf_timeframe", "5m")
                mtf_limit = cfg.get("mtf_candles", 30)
                # Refrescar HTF cada 60s para no saturar API
                if time.time() - htf_last_refresh_ts > 60:
                    try:
                        htf_raw = get_klines(symbol, mtf_tf, limit=mtf_limit)
                        htf_closes = [float(k[4]) for k in htf_raw]
                        htf_last_refresh_ts = time.time()
                        htf_trend = get_htf_trend(htf_closes)
                    except Exception:
                        pass  # Mantiene ultimo valor
                if htf_trend == "down":
                    raw_signal = "HOLD"
                    if iteration % 10 == 0:
                        log(f"[{symbol}] MTF {mtf_tf} tendencia bajista — BUY bloqueado")

            rsi = calc_rsi(close_list, RSI_PERIOD)
            ema_fast = calc_ema(close_list, EMA_FAST)
            ema_slow = calc_ema(close_list, EMA_SLOW)
            macd_line, macd_signal_line, macd_hist = calc_macd(close_list, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

            ai_probability = None
            ai_decision = "off"
            if raw_signal == "BUY":
                buy_signal_streak += 1
            else:
                buy_signal_streak = 0

            if raw_signal == "SELL":
                sell_signal_streak += 1
            else:
                sell_signal_streak = 0

            required_confirmations = cfg["signal_confirmations"]
            signal = raw_signal

            ms_recommendation = "ok"
            ms_deviation_pct = 0.0
            ms_cfg = load_multi_source_config()
            if raw_signal == "BUY" and ms_cfg["enabled"]:
                now_ts = time.time()
                if (ms_snapshot is None) or (now_ts - ms_last_refresh_ts >= ms_cfg["refresh_seconds"]):
                    ms_snapshot = get_multi_source_snapshot(symbol)
                    ms_last_refresh_ts = now_ts

                ms_recommendation = ms_snapshot.get("recommendation", "ok")
                ms_deviation_pct = float(ms_snapshot.get("max_deviation_pct", 0.0) or 0.0)
                if ms_recommendation == "reduce":
                    required_confirmations += 1

            if raw_signal == "BUY" and buy_signal_streak < required_confirmations:
                signal = "HOLD"

            if cfg.get("ai_enabled", False) and cfg.get("ai_mode", "off") != "off":
                model = load_ai_model()
                if model:
                    feats = build_ai_features(
                        price=price,
                        ema_fast=ema_fast,
                        ema_slow=ema_slow,
                        rsi=rsi,
                        macd_line=macd_line,
                        macd_signal_line=macd_signal_line,
                        macd_hist=macd_hist,
                        raw_signal=raw_signal,
                        in_position=wallet.in_position,
                        update_mode=cfg.get("update_mode", "candle"),
                        candles=candles,
                        closes=close_list,
                    )
                    ai_probability = ai_predict_probability(model, feats)
                    ai_mode = cfg.get("ai_mode", "filter")
                    if ai_mode == "filter" and raw_signal == "BUY":
                        if ai_probability < cfg.get("ai_min_confidence", 0.60):
                            signal = "HOLD"
                            ai_decision = "blocked"
                        else:
                            ai_decision = "approved"
                    else:
                        ai_decision = "advice"
                    ai_model_missing_announced = False
                else:
                    if not ai_model_missing_announced:
                        log(f"[{symbol}] AI habilitada pero sin modelo {AI_MODEL_FILE}. Fallback a estrategia tecnica.")
                        ai_model_missing_announced = True
                    ai_decision = "fallback_no_model"

            # ── Intelligence Engine: evaluacion multi-capa ──
            intel_eval = None
            intel_blocked = False
            if intelligence_engine and cfg.get("intel_enabled", True) and signal == "BUY":
                try:
                    intel_eval = intelligence_engine.evaluate_entry(
                        symbol,
                        price=price,
                        ema_fast=ema_fast,
                        ema_slow=ema_slow,
                        rsi=rsi,
                        macd_hist=macd_hist,
                        candles=list(candles),
                        raw_signal=raw_signal,
                    )
                    if not intel_eval["approved"]:
                        signal = "HOLD"
                        intel_blocked = True
                        if iteration % 5 == 0:
                            log(f"[{symbol}] GENIE bloqueó BUY: score={intel_eval['score']}, razones={intel_eval['blocks']}")
                except Exception as exc:
                    if iteration % 20 == 0:
                        log(f"[{symbol}] GENIE error (fallback): {exc}")

            balance = wallet.total_value(price)
            iteration += 1

            append_dataset_row(
                {
                    "timestamp": now_iso(),
                    "symbol": symbol,
                    "price": price,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "rsi": rsi,
                    "macd": macd_line,
                    "macd_signal": macd_signal_line,
                    "macd_hist": macd_hist,
                    "signal_regla": signal,
                    "in_position": wallet.in_position,
                    "balance": balance,
                    "future_return_n": "",
                    "update_mode": cfg["update_mode"],
                    "atr_norm": calc_atr(candles) / max(price, 1e-9),
                    "bb_pct": calc_bollinger_pct(close_list),
                    "volume_ratio": calc_volume_ratio(candles),
                    "hour_sin": math.sin(2 * math.pi * (time.localtime().tm_hour + time.localtime().tm_min / 60.0) / 24.0),
                    "hour_cos": math.cos(2 * math.pi * (time.localtime().tm_hour + time.localtime().tm_min / 60.0) / 24.0),
                }
            )

            high_latency = latency_ms > cfg["max_api_latency_ms"]
            in_cooldown = time.time() < next_trade_after
            cooldown_remaining = max(0, int(next_trade_after - time.time())) if in_cooldown else 0

            emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "price", initial_balance)
            update_pair_state(
                symbol,
                {
                    "api_latency_ms": latency_ms,
                    "raw_signal": raw_signal,
                    "buy_signal_streak": buy_signal_streak,
                    "sell_signal_streak": sell_signal_streak,
                    "high_latency": high_latency,
                    "cooldown_remaining": cooldown_remaining,
                    "ms_recommendation": ms_recommendation,
                    "ms_deviation_pct": ms_deviation_pct,
                    "ms_required_confirmations": required_confirmations,
                    "day_pnl": day_pnl,
                    "daily_kill_active": daily_kill_active,
                    "trades_last_hour": len(sells_last_hour),
                    "trades_per_hour_reached": trades_per_hour_reached,
                    "ai_probability": ai_probability,
                    "ai_decision": ai_decision,
                    "intel_score": intel_eval["score"] if intel_eval else None,
                    "intel_grade": intel_eval["grade"] if intel_eval else None,
                    "intel_blocked": intel_blocked,
                    "intel_blocks": intel_eval["blocks"] if intel_eval and not intel_eval["approved"] else [],
                    "intel_summary": intelligence_engine.get_intel_summary_for_pair(symbol) if intelligence_engine else {},
                },
            )

            if signal != previous_signal:
                emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "signal", initial_balance)
                previous_signal = signal

            if abs(balance - previous_balance) > 1e-8:
                emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "balance", initial_balance)
                previous_balance = balance

            if iteration % 10 == 0:
                closed_trades = len([t for t in wallet.trades if t["type"] == "SELL"])
                log(
                    f"[{symbol}] Estado | Balance: ${balance:.2f} | Precio: ${price:,.2f} | RSI: {rsi:.1f} | Senal: {signal} | Trades: {closed_trades}"
                )

            if balance >= target_balance:
                if wallet.in_position:
                    trade = wallet.sell(price, reason="objetivo alcanzado")
                    if trade:
                        if intelligence_engine:
                            intelligence_engine.register_position_close(symbol)
                        emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "trade", initial_balance)
                log(f"[{symbol}] Objetivo de ${target_balance:.2f} alcanzado. Hilo finalizado.")
                break

            if high_latency:
                log(
                    f"[{symbol}] Guard latencia activo: {latency_ms:.0f}ms > {cfg['max_api_latency_ms']}ms. Se evitan entradas y salidas por senal."
                )

            if daily_kill_active and not kill_switch_announced:
                log(
                    f"[{symbol}] Kill switch diario activo: PnL dia ${day_pnl:.2f} <= -${cfg['daily_loss_limit_usdt']:.2f}. Se bloquean nuevas entradas."
                )
                kill_switch_announced = True
                if telegram_notifier:
                    telegram_notifier.notify_kill_switch(symbol, "daily_loss", f"PnL dia ${day_pnl:.2f}")

            if drawdown_kill_active and not drawdown_kill_announced:
                log(
                    f"[{symbol}] Kill switch drawdown activo: DD {drawdown_from_initial*100:.2f}% >= {cfg['max_drawdown_pct']*100:.1f}%. Se bloquean nuevas entradas."
                )
                drawdown_kill_announced = True
                if telegram_notifier:
                    telegram_notifier.notify_kill_switch(symbol, "max_drawdown", f"DD {drawdown_from_initial*100:.2f}%")
            if not drawdown_kill_active:
                drawdown_kill_announced = False

            if trades_per_hour_reached and not trades_cap_announced:
                log(
                    f"[{symbol}] Limite por hora activo: {len(sells_last_hour)} SELL en 60m (max {cfg['max_trades_per_hour']}). Se bloquean nuevas entradas."
                )
                trades_cap_announced = True
            if not trades_per_hour_reached and not trades_per_day_reached:
                trades_cap_announced = False
            if trades_per_day_reached and not trades_cap_announced:
                log(
                    f"[{symbol}] Limite diario activo: {day_trades} trades hoy (max {max_daily}). Se bloquean nuevas entradas."
                )
                trades_cap_announced = True

            if wallet.in_position:
                # ── Adaptive trailing: usar ATR del par si intel esta habilitado ──
                if intelligence_engine and cfg.get("intel_adaptive_trailing_enabled", True) and cfg.get("intel_enabled", True):
                    try:
                        _adaptive_trail = intelligence_engine.get_adaptive_trailing(
                            symbol, list(candles), cfg.get("intel_trailing_atr_mult", 1.5)
                        )
                        # Temporalmente overridear trailing del wallet
                        _orig_trailing = cfg.get("trailing_stop_pct", 0)
                        if _adaptive_trail > 0 and _orig_trailing > 0:
                            # Usar el mayor entre config global y adaptativo
                            cfg["trailing_stop_pct"] = max(_orig_trailing, _adaptive_trail)
                    except Exception:
                        pass
                wallet.update_trailing_stop(price)
                change = 0.0
                if wallet.entry_price:
                    change = (price - wallet.entry_price) / wallet.entry_price
                held_seconds = int(time.time() - entry_ts) if entry_ts else 0

                # Break-even stop: Cuando ganancia >= break_even_pct, protege la ganancia moviendo stop a entry_price
                if change >= cfg.get("break_even_pct", 0.005) and wallet.trailing_stop_price is None:
                    wallet.trailing_stop_price = wallet.entry_price
                    log(
                        f"[{symbol}] Break-even stop activado en ${wallet.entry_price:.2f} (+{change*100:.2f}% ganancia)"
                    )

                # Loss prevention: Si se pierde > max_initial_loss_pct en primeros 10s, cierra inmediatamente
                if held_seconds < 10 and change < -cfg.get("max_initial_loss_pct", 0.003):
                    trade = wallet.sell(price, reason="loss prevention cutoff")
                    if trade:
                        day_pnl += float(trade.get("pnl", 0.0))
                        sells_last_hour.append(time.time())
                        day_trades += 1
                        entry_ts = None
                        next_trade_after = time.time() + cfg["cooldown_seconds"]
                        if intelligence_engine:
                            intelligence_engine.register_position_close(symbol)
                        emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "trade", initial_balance)
                elif wallet.trailing_stop_price is not None and price <= wallet.trailing_stop_price:
                    trade = wallet.sell(price, reason="trailing stop")
                    if trade:
                        day_pnl += float(trade.get("pnl", 0.0))
                        sells_last_hour.append(time.time())
                        day_trades += 1
                        entry_ts = None
                        next_trade_after = time.time() + cfg["cooldown_seconds"]
                        if intelligence_engine:
                            intelligence_engine.register_position_close(symbol)
                        emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "trade", initial_balance)
                elif change >= (wallet.active_take_profit_pct or cfg["take_profit_pct"]):
                    trade = wallet.sell(price, reason="take profit")
                    if trade:
                        day_pnl += float(trade.get("pnl", 0.0))
                        sells_last_hour.append(time.time())
                        day_trades += 1
                        entry_ts = None
                        next_trade_after = time.time() + cfg["cooldown_seconds"]
                        if intelligence_engine:
                            intelligence_engine.register_position_close(symbol)
                        emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "trade", initial_balance)
                elif (
                    signal == "SELL"
                    and not high_latency
                    and not in_cooldown
                    and sell_signal_streak >= cfg["sell_signal_confirmations"]
                    and held_seconds >= cfg["min_hold_seconds"]
                ):
                    # ── Intel sell filter: si el trade va en positivo y la tendencia es favorable, esperar ──
                    intel_hold_override = False
                    if intelligence_engine and cfg.get("intel_enabled", True) and change > 0:
                        try:
                            regime_info = intelligence_engine.get_intelligence_state().get("regimes", {}).get(symbol, {})
                            current_regime = regime_info.get("regime", "")
                            if current_regime == "trending_up" and change < cfg["take_profit_pct"] * 0.8:
                                intel_hold_override = True
                                if iteration % 10 == 0:
                                    log(f"[{symbol}] GENIE mantiene posicion: regime={current_regime}, profit={change*100:.2f}%, esperando TP")
                        except Exception:
                            pass

                    if not intel_hold_override:
                        trade = wallet.sell(price, reason="senal EMA/RSI/MACD")
                    if trade:
                        day_pnl += float(trade.get("pnl", 0.0))
                        sells_last_hour.append(time.time())
                        day_trades += 1
                        entry_ts = None
                        next_trade_after = time.time() + cfg["cooldown_seconds"]
                        if intelligence_engine:
                            intelligence_engine.register_position_close(symbol)
                        emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "trade", initial_balance)
            elif (
                signal == "BUY"
                and wallet.usdt > 10
                and not high_latency
                and not in_cooldown
                and not daily_kill_active
                and not drawdown_kill_active
                and not trades_per_hour_reached
                and not trades_per_day_reached
            ):
                if ms_recommendation == "pause":
                    log(
                        f"[{symbol}] Entrada bloqueada por consenso multifuente: desvio={ms_deviation_pct:.3f}% accion=pause"
                    )
                else:
                    # ── Regime-based params ──
                    effective_trade_pct = cfg["trade_pct"]
                    effective_tp = cfg["take_profit_pct"]
                    effective_sl = cfg.get("stop_loss_pct", 0.005)
                    effective_trailing = cfg["trailing_stop_pct"]

                    try:
                        import auto_optimizer as _ao
                        regime_state = intelligence_engine.get_intelligence_state().get("regimes", {}).get(symbol, {}) if intelligence_engine else {}
                        current_regime = regime_state.get("regime", "")
                        if current_regime and current_regime != "unknown":
                            rp = _ao.get_params_for_regime(current_regime)
                            effective_tp = rp.get("take_profit_pct", effective_tp)
                            effective_sl = rp.get("stop_loss_pct", effective_sl)
                            effective_trailing = rp.get("trailing_stop_pct", effective_trailing)
                            regime_mult = rp.get("trade_pct_mult", 1.0)
                            effective_trade_pct *= regime_mult
                    except Exception:
                        pass

                    # ── Adaptive sizing por IA ──
                    if cfg.get("ai_adaptive_sizing") and ai_probability is not None and ai_probability > 0:
                        hi = cfg["ai_high_confidence"]
                        lo = cfg["ai_low_confidence"]
                        if ai_probability >= hi:
                            effective_trade_pct *= cfg["ai_high_trade_pct_mult"]
                            effective_tp *= cfg["ai_high_tp_mult"]
                        elif ai_probability <= lo:
                            effective_trade_pct *= cfg["ai_low_trade_pct_mult"]
                            effective_tp *= cfg["ai_low_tp_mult"]
                    # Intel adjustments (trade_pct_mult de régimen volátil)
                    if intel_eval and "trade_pct_mult" in intel_eval.get("adjustments", {}):
                        effective_trade_pct *= intel_eval["adjustments"]["trade_pct_mult"]
                    # Clamp a rangos seguros
                    effective_trade_pct = min(max(effective_trade_pct, 0.10), 1.0)
                    effective_tp = min(max(effective_tp, 0.0001), 0.5)

                    trade = wallet.buy(price, trade_pct_override=effective_trade_pct)
                    if trade:
                        wallet.active_take_profit_pct = effective_tp
                        entry_ts = time.time()
                        next_trade_after = time.time() + cfg["cooldown_seconds"]
                        if intelligence_engine:
                            intelligence_engine.register_position_open(symbol)
                        emit_runtime_state(symbol, wallet, price, signal, rsi, list(candles), close_list, "trade", initial_balance)

            time.sleep(cfg["poll_seconds"])

    except Exception as exc:
        log(f"[{symbol}] Error: {exc}")
    finally:
        try:
            final_price = get_price(symbol)
            wallet.summary(final_price)
        except Exception as exc:
            log(f"[{symbol}] Error al imprimir summary: {exc}")


def preflight_check_real(symbols: list) -> bool:
    """Valida conectividad, API keys y permisos antes de operar en modo real."""
    print("\n--- Preflight Check ---")
    errors = []

    # 1. Conectividad: ping al servidor
    try:
        resp = requests.get(f"{BASE_URL}/api/v3/ping", timeout=10)
        resp.raise_for_status()
        print(f"  [OK] Conectividad a {BASE_URL}")
    except Exception as exc:
        errors.append(f"No se pudo conectar a {BASE_URL}: {exc}")
        print(f"  [FAIL] Conectividad: {exc}")

    # 2. Server time: verificar sincronización de reloj
    try:
        resp = requests.get(f"{BASE_URL}/api/v3/time", timeout=10)
        server_time = resp.json().get("serverTime", 0)
        local_time = int(time.time() * 1000)
        drift_ms = abs(local_time - server_time)
        if drift_ms > 5000:
            errors.append(f"Reloj desincronizado: drift={drift_ms}ms (max 5000ms)")
            print(f"  [FAIL] Sincronización reloj: drift={drift_ms}ms")
        else:
            print(f"  [OK] Sincronización reloj: drift={drift_ms}ms")
    except Exception as exc:
        errors.append(f"No se pudo verificar hora del servidor: {exc}")

    # 3. API Key: verificar autenticación y permisos
    try:
        ts = int(time.time() * 1000)
        payload = f"timestamp={ts}&recvWindow=5000"
        sig = hmac.new(
            BINANCE_API_SECRET.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        resp = requests.get(
            f"{BASE_URL}/api/v3/account",
            params={"timestamp": ts, "recvWindow": 5000, "signature": sig},
            headers={"X-MBX-APIKEY": BINANCE_API_KEY},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            can_trade = data.get("canTrade", False)
            balances = {b["asset"]: float(b["free"]) for b in data.get("balances", []) if float(b["free"]) > 0}
            usdt_balance = balances.get("USDT", 0.0)
            print(f"  [OK] API Key válida — canTrade={can_trade}")
            print(f"  [OK] Balance USDT: ${usdt_balance:.2f}")
            if not can_trade:
                errors.append("La API key no tiene permisos de trading (canTrade=false)")
        else:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            errors.append(f"API Key inválida: HTTP {resp.status_code} — {err_data.get('msg', resp.text[:100])}")
            print(f"  [FAIL] API Key: HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"Error verificando API Key: {exc}")
        print(f"  [FAIL] API Key: {exc}")

    # 4. Verificar que los pares existen en el exchange
    try:
        resp = requests.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=15)
        exchange_symbols = {s["symbol"] for s in resp.json().get("symbols", [])}
        for sym in symbols:
            if sym in exchange_symbols:
                print(f"  [OK] Par {sym} disponible")
            else:
                errors.append(f"Par {sym} no encontrado en el exchange")
                print(f"  [FAIL] Par {sym} no disponible")
    except Exception as exc:
        errors.append(f"No se pudo verificar pares: {exc}")

    # 5. Verificar que se puede obtener precio de cada par
    for sym in symbols:
        try:
            p = get_price(sym)
            if p and p > 0:
                print(f"  [OK] Precio {sym}: ${p:,.2f}")
            else:
                errors.append(f"Precio inválido para {sym}: {p}")
        except Exception as exc:
            errors.append(f"Error obteniendo precio de {sym}: {exc}")

    print("--- Fin Preflight ---\n")

    if errors:
        print("ERRORES DETECTADOS:")
        for e in errors:
            print(f"  [X] {e}")
        log(f"Preflight FALLIDO: {len(errors)} errores — {errors}")
        return False

    log("Preflight OK: conectividad, API key, permisos y pares verificados.")
    return True


def confirm_real_mode_or_exit():
    if TRADING_MODE != "real":
        return True

    env_label = "TESTNET" if BINANCE_TESTNET else "MAINNET (DINERO REAL)"
    print("\n" + "!" * 60)
    print(f"MODO REAL ACTIVADO — {env_label}")
    print(f"  URL: {BASE_URL}")
    print("Se enviarán órdenes reales a Binance.")
    print("Revisa API keys, riesgos y permisos antes de continuar.")
    print("Escribi SI para confirmar y operar en real.")
    print("!" * 60)
    confirm = input("Confirmacion: ").strip().upper()
    if confirm != "SI":
        log("Modo real cancelado por el usuario.")
        return False
    return True


def run():
    global ACTIVE_SYMBOLS
    if TRADING_MODE not in {"paper", "real"}:
        raise RuntimeError("TRADING_MODE invalido. Usa 'paper' o 'real'.")

    cfg = get_runtime_config()
    ACTIVE_SYMBOLS = cfg.get("symbols", SYMBOLS.copy())
    pair_count = max(len(ACTIVE_SYMBOLS), 1)

    # Iniciar motor de inteligencia
    if intelligence_engine and cfg.get("intel_enabled", True):
        try:
            intelligence_engine.start_intelligence()
            log("[GENIE] Inteli Genie iniciado.")
        except Exception as exc:
            log(f"[GENIE] Error al iniciar Inteli Genie: {exc}")

    wallet_balance = cfg.get("wallet_balance", PAPER_BALANCE)
    pair_allocations = cfg.get("pair_allocations", {})
    pair_targets = cfg.get("pair_targets", {})
    global_target = cfg["target_usdt"]

    # Calcular balance por par: si hay allocation explicita se usa, si no se reparte equitativamente
    allocated_total = sum(pair_allocations.get(s, 0) for s in ACTIVE_SYMBOLS if s in pair_allocations)
    unallocated_symbols = [s for s in ACTIVE_SYMBOLS if s not in pair_allocations]
    remaining = max(wallet_balance - allocated_total, 0)
    equal_share = remaining / max(len(unallocated_symbols), 1) if unallocated_symbols else 0

    balances = {}
    targets = {}
    for sym in ACTIVE_SYMBOLS:
        balances[sym] = pair_allocations.get(sym, equal_share)
        targets[sym] = pair_targets.get(sym, global_target / pair_count)

    print("=" * 50)
    mode_label = TRADING_MODE.upper()
    if TRADING_MODE == "real" and BINANCE_TESTNET:
        mode_label += " (TESTNET)"
    print(f"   BOT SCALPING MULTI-PAR — {mode_label}")
    print(f"   Pares           : {', '.join(ACTIVE_SYMBOLS)}")
    print(f"   Billetera total : ${wallet_balance:.2f} USDT")
    for sym in ACTIVE_SYMBOLS:
        print(f"     {sym:10s}    : ${balances[sym]:.2f} -> objetivo ${targets[sym]:.2f}")
    print(f"   Objetivo global : ${global_target:.2f} USDT")
    print(f"   Trailing stop   : {cfg['trailing_stop_pct'] * 100:.2f}%")
    print(f"   Take profit     : {cfg['take_profit_pct'] * 100:.2f}%")
    print(f"   Trade pct       : {cfg['trade_pct'] * 100:.1f}%")
    print(f"   Update mode     : {cfg['update_mode']}")
    print(f"   Poll segundos   : {cfg['poll_seconds']}s")
    print(f"   Confirmaciones  : {cfg['signal_confirmations']}")
    print(f"   Confirm SELL    : {cfg['sell_signal_confirmations']}")
    print(f"   Min hold        : {cfg['min_hold_seconds']}s")
    print(f"   Max latencia    : {cfg['max_api_latency_ms']} ms")
    print(f"   Cooldown trade  : {cfg['cooldown_seconds']}s")
    print(f"   Loss diario max : ${cfg['daily_loss_limit_usdt']:.2f}")
    print(f"   Max trades/hora : {cfg['max_trades_per_hour']}")
    print(f"   AI enabled      : {cfg['ai_enabled']}")
    print(f"   AI mode         : {cfg['ai_mode']}")
    print(f"   AI min conf     : {cfg['ai_min_confidence']:.2f}")
    print("=" * 50 + "\n")

    with STATE_LOCK:
        BOT_STATE["runtime_config"] = cfg
    write_bot_state()

    if not confirm_real_mode_or_exit():
        return

    # Preflight en modo real: verifica conectividad, api keys, permisos, pares
    if TRADING_MODE == "real":
        if not preflight_check_real(ACTIVE_SYMBOLS):
            log("Abortando: preflight check fallido.")
            return

    threads = {}
    for symbol in ACTIVE_SYMBOLS:
        thread = threading.Thread(
            target=run_symbol,
            args=(symbol, balances[symbol], targets[symbol], pair_count),
            daemon=True,
        )
        thread.start()
        threads[symbol] = thread

    try:
        while True:
            # Revisar si se agregaron nuevos pares a la config
            try:
                live_cfg = get_runtime_config()
                live_symbols = live_cfg.get("symbols", [])
                live_allocs = live_cfg.get("pair_allocations", {})
                live_targets = live_cfg.get("pair_targets", {})
                live_count = max(len(live_symbols), 1)
                for sym in live_symbols:
                    if sym not in threads or not threads[sym].is_alive():
                        alloc = live_allocs.get(sym, live_cfg["wallet_balance"] / live_count)
                        target = live_targets.get(sym, live_cfg["target_usdt"] / live_count)
                        log(f"[MAIN] Nuevo par detectado: {sym} (${alloc:.2f} -> ${target:.2f}). Lanzando hilo...")
                        t = threading.Thread(
                            target=run_symbol,
                            args=(sym, alloc, target, live_count),
                            daemon=True,
                        )
                        t.start()
                        threads[sym] = t
            except Exception:
                pass

            # Verificar si todos los hilos terminaron
            alive = [t for t in threads.values() if t.is_alive()]
            if not alive:
                log("[MAIN] Todos los hilos finalizaron.")
                break
            time.sleep(15)
    except KeyboardInterrupt:
        log("Bot detenido por el usuario.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("Bot detenido por el usuario.")
    except Exception as _exc:
        import traceback, logging as _lg
        _lg.basicConfig(filename="_bot_crash.log", level=_lg.ERROR)
        _lg.error("Bot crash: %s\n%s", _exc, traceback.format_exc())
        print(f"CRASH: {_exc}  — ver _bot_crash.log")
        import sys
        sys.exit(1)
