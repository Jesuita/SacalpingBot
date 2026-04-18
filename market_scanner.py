"""
Market Scanner v2 — Analisis avanzado de pares para scalping.

Mejoras sobre v1:
- MACD, Bollinger Bands, ATR ademas de EMA/RSI
- Deteccion de regimen de mercado (trending vs ranging)
- Score de aptitud para scalping (liquidez, rango intradiario, mean-reversion)
- Categorias de oportunidad: Hot / Steady / Risky
- Historial de scores para detectar tendencias
- Mejor analisis de underperformers (lee trades.log)
"""

import json
import math
import os
import requests
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────

SCAN_INTERVAL_SECONDS = 300   # Cada 5 minutos
TOP_PAIRS_TO_SCAN = 40        # Top 40 pares USDT por volumen
MIN_VOLUME_24H_USDT = 30_000_000  # Min $30M volumen 24h
KLINE_INTERVAL = "5m"
KLINE_LIMIT = 100             # 100 velas de 5m = ~8.3 horas de datos

# Pares a excluir siempre (stablecoins, leveraged, etc.)
EXCLUDED_PAIRS = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT",
    "USDPUSDT", "EURUSDT", "GBPUSDT", "USD1USDT",
}

RUNTIME_CONFIG_FILE = "runtime_config.json"
TRADES_LOG_FILE = "trades.log"
BINANCE_API = "https://api.binance.com"

# ─────────────────────────────────────────────
#  ESTADO GLOBAL DEL SCANNER
# ─────────────────────────────────────────────

_scanner_lock = threading.Lock()
_scanner_state: Dict[str, Any] = {
    "last_scan": None,
    "scanned_pairs": 0,
    "suggestions": [],
    "pair_scores": {},
    "underperformers": [],
    "market_overview": {},
    "running": False,
    "error": None,
}

_alerts: deque = deque(maxlen=100)
_score_history: Dict[str, List[Dict[str, Any]]] = {}


# ─────────────────────────────────────────────
#  INDICADORES TECNICOS
# ─────────────────────────────────────────────

def _ema(data: List[float], period: int) -> List[float]:
    """Calcula EMA completa como serie."""
    if not data or period < 1:
        return []
    result = []
    k = 2.0 / (period + 1)
    ema = sum(data[:period]) / period if len(data) >= period else data[0]
    for i, val in enumerate(data):
        if i < period - 1:
            ema = sum(data[:i+1]) / (i + 1)
        else:
            ema = val * k + ema * (1 - k)
        result.append(ema)
    return result


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(d if d > 0 else 0.0)
        losses.append(-d if d < 0 else 0.0)
    if len(gains) < period:
        return 50.0
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al < 1e-12:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _macd(closes: List[float]) -> Tuple[float, float, float]:
    """Retorna (macd_line, signal_line, histogram)."""
    if len(closes) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema(macd_line, 9)
    hist = macd_line[-1] - signal[-1] if signal else 0.0
    return macd_line[-1], signal[-1] if signal else 0.0, hist


def _bollinger(closes: List[float], period: int = 20, std_mult: float = 2.0) -> Dict[str, float]:
    """Bandas de Bollinger: upper, middle, lower, width_pct, position."""
    if len(closes) < period:
        p = closes[-1] if closes else 0.0
        return {"upper": p, "middle": p, "lower": p, "width_pct": 0.0, "position": 0.5}
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    width_pct = (upper - lower) / middle * 100 if middle > 0 else 0
    price = closes[-1]
    position = (price - lower) / (upper - lower) if (upper - lower) > 1e-12 else 0.5
    return {"upper": upper, "middle": middle, "lower": lower, "width_pct": width_pct, "position": position}


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Average True Range."""
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0
    atr_val = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
    return atr_val


def _volume_profile(volumes: List[float], window: int = 10) -> Dict[str, float]:
    """Analisis de perfil de volumen."""
    if len(volumes) < window + 5:
        return {"trend": 1.0, "spike": 1.0, "consistency": 0.5}
    recent = volumes[-window:]
    older = volumes[:-window]
    avg_recent = sum(recent) / len(recent) if recent else 1
    avg_older = sum(older) / len(older) if older else 1
    trend = avg_recent / max(avg_older, 1e-9)
    max_recent = max(recent) if recent else 1
    spike = max_recent / max(avg_recent, 1e-9)
    if avg_recent > 0:
        deviations = [abs(v - avg_recent) / avg_recent for v in recent]
        consistency = max(0, 1.0 - sum(deviations) / len(deviations))
    else:
        consistency = 0.0
    return {"trend": trend, "spike": spike, "consistency": consistency}


def _detect_regime(closes: List[float], ema_fast_s: List[float], ema_slow_s: List[float], bb: Dict) -> str:
    """Detecta regimen: trending_up, trending_down, ranging, volatile."""
    if len(closes) < 30:
        return "unknown"
    lookback = min(30, len(ema_fast_s), len(ema_slow_s))
    crossovers = 0
    for i in range(-lookback + 1, 0):
        if (ema_fast_s[i] > ema_slow_s[i]) != (ema_fast_s[i-1] > ema_slow_s[i-1]):
            crossovers += 1
    bb_width = bb.get("width_pct", 0)
    ema_diff = (ema_fast_s[-1] - ema_slow_s[-1]) / max(closes[-1], 1e-9) * 100

    if bb_width > 3.0 and crossovers >= 4:
        return "volatile"
    elif abs(ema_diff) > 0.3 and crossovers <= 2:
        return "trending_up" if ema_diff > 0 else "trending_down"
    else:
        return "ranging"


# ─────────────────────────────────────────────
#  SCORING SYSTEM
# ─────────────────────────────────────────────

def _score_scalping_fitness(atr_pct, volume_usdt_m, vol_consistency, bb_width, regime, rsi):
    """Score 0-100 de que tan apto es un par para scalping."""
    score = 0.0

    # ATR como % del precio: ideal 0.15% - 1.0% (max 30pts)
    if 0.15 <= atr_pct <= 1.0:
        score += 30 if 0.25 <= atr_pct <= 0.7 else 20
    elif atr_pct > 1.0:
        score += 10
    else:
        score += 5

    # Liquidez (max 25pts)
    if volume_usdt_m >= 500:
        score += 25
    elif volume_usdt_m >= 200:
        score += 20
    elif volume_usdt_m >= 100:
        score += 15
    elif volume_usdt_m >= 50:
        score += 10
    else:
        score += 5

    # Consistencia de volumen (max 15pts)
    score += vol_consistency * 15

    # Regimen: ranging ideal para scalping (max 20pts)
    regime_scores = {"ranging": 20, "trending_up": 15, "trending_down": 10, "volatile": 5, "unknown": 8}
    score += regime_scores.get(regime, 8)

    # RSI zona neutra (max 10pts)
    if 35 <= rsi <= 65:
        score += 10
    elif 25 <= rsi <= 75:
        score += 6
    else:
        score += 2

    return max(0, min(100, score))


def _score_opportunity(rsi, macd_hist, ema_diff_pct, bb_position, price_change_pct, vol_trend):
    """Score 0-100 de oportunidad inmediata."""
    score = 0.0

    # RSI sobreventa = oportunidad (max 25pts)
    if rsi < 25: score += 25
    elif rsi < 35: score += 20
    elif rsi < 45: score += 12
    elif rsi > 75: score += 3
    elif rsi > 65: score += 5
    else: score += 8

    # MACD histogram (max 20pts)
    if macd_hist > 0:
        score += min(20, 10 + abs(macd_hist) * 500)
    else:
        score += max(0, 5 - abs(macd_hist) * 200)

    # EMA crossover (max 20pts)
    if ema_diff_pct > 0.3: score += 20
    elif ema_diff_pct > 0.1: score += 15
    elif ema_diff_pct > 0: score += 10
    elif ema_diff_pct > -0.1: score += 5

    # Bollinger position: near lower = buy (max 20pts)
    if bb_position < 0.2: score += 20
    elif bb_position < 0.4: score += 14
    elif bb_position < 0.6: score += 8
    elif bb_position < 0.8: score += 4

    # Volumen trend (max 15pts)
    if vol_trend > 1.5: score += 15
    elif vol_trend > 1.2: score += 10
    elif vol_trend > 1.0: score += 5
    else: score += 2

    return max(0, min(100, score))


def _classify_opportunity(scalping_score, opportunity_score, regime):
    """Clasifica la oportunidad en categoria."""
    combined = scalping_score * 0.6 + opportunity_score * 0.4

    if combined >= 70 and regime in ("ranging", "trending_up"):
        return {"category": "hot", "label": "Oportunidad caliente", "emoji": "\U0001f525",
                "description": "Excelentes condiciones para scalping ahora", "combined_score": round(combined, 1)}
    elif combined >= 55 and scalping_score >= 50:
        return {"category": "steady", "label": "Par estable", "emoji": "\u2705",
                "description": "Buena liquidez y volatilidad controlada", "combined_score": round(combined, 1)}
    elif opportunity_score >= 65 and scalping_score < 45:
        return {"category": "risky", "label": "Arriesgado", "emoji": "\u26a1",
                "description": "Buena senal tecnica pero scalping no ideal", "combined_score": round(combined, 1)}
    elif combined >= 40:
        return {"category": "watch", "label": "Para vigilar", "emoji": "\U0001f440",
                "description": "Podria mejorar, mantener en radar", "combined_score": round(combined, 1)}
    else:
        return {"category": "skip", "label": "No recomendado", "emoji": "\u2b1c",
                "description": "No apto para scalping ahora", "combined_score": round(combined, 1)}


# ─────────────────────────────────────────────
#  API DE BINANCE
# ─────────────────────────────────────────────

def _get_top_usdt_pairs() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{BINANCE_API}/api/v3/ticker/24hr", timeout=10)
        resp.raise_for_status()
        usdt_pairs = []
        for t in resp.json():
            symbol = t.get("symbol", "")
            if not symbol.endswith("USDT") or symbol in EXCLUDED_PAIRS:
                continue
            vol = float(t.get("quoteVolume", 0))
            if vol < MIN_VOLUME_24H_USDT:
                continue
            usdt_pairs.append({
                "symbol": symbol,
                "price": float(t.get("lastPrice", 0)),
                "change_pct": float(t.get("priceChangePercent", 0)) / 100,
                "volume_usdt": vol,
                "high_24h": float(t.get("highPrice", 0)),
                "low_24h": float(t.get("lowPrice", 0)),
                "trades_24h": int(t.get("count", 0)),
            })
        usdt_pairs.sort(key=lambda x: x["volume_usdt"], reverse=True)
        return usdt_pairs[:TOP_PAIRS_TO_SCAN]
    except Exception as e:
        with _scanner_lock:
            _scanner_state["error"] = f"Error tickers: {e}"
        return []


def _get_klines(symbol: str) -> List[Dict[str, float]]:
    try:
        resp = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={"symbol": symbol, "interval": KLINE_INTERVAL, "limit": KLINE_LIMIT},
            timeout=8,
        )
        resp.raise_for_status()
        return [{"close": float(k[4]), "high": float(k[2]), "low": float(k[3]),
                 "open": float(k[1]), "volume": float(k[5])} for k in resp.json()]
    except Exception:
        return []


# ─────────────────────────────────────────────
#  ANALISIS DE RENDIMIENTO (desde trades.log)
# ─────────────────────────────────────────────

def _get_active_pairs() -> List[str]:
    try:
        if os.path.exists(RUNTIME_CONFIG_FILE):
            with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("symbols", [])
    except Exception:
        pass
    return []


def _analyze_performance_from_log() -> Dict[str, Dict[str, Any]]:
    """Lee trades.log para rendimiento real por par."""
    if not os.path.exists(TRADES_LOG_FILE):
        return {}
    perf: Dict[str, Dict[str, Any]] = {}
    try:
        with open(TRADES_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 5:
                    continue
                row = {}
                for item in parts[2:]:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        row[k.strip()] = v.strip()
                if row.get("tipo") != "SELL":
                    continue
                sym = row.get("symbol", "")
                if not sym:
                    continue
                try:
                    pnl = float(row.get("pnl", 0))
                except ValueError:
                    pnl = 0.0
                if sym not in perf:
                    perf[sym] = {"pnl": 0.0, "wins": 0, "losses": 0, "trades": 0, "pnls": []}
                perf[sym]["trades"] += 1
                perf[sym]["pnl"] += pnl
                perf[sym]["pnls"].append(pnl)
                if pnl > 0:
                    perf[sym]["wins"] += 1
                else:
                    perf[sym]["losses"] += 1
    except Exception:
        pass

    for sym, p in perf.items():
        p["win_rate"] = p["wins"] / p["trades"] * 100 if p["trades"] > 0 else 0
        gp = sum(x for x in p["pnls"] if x > 0)
        gl = sum(x for x in p["pnls"] if x < 0)
        p["profit_factor"] = gp / abs(gl) if gl < 0 else (gp if gp > 0 else 0)
        p["expectancy"] = p["pnl"] / p["trades"] if p["trades"] > 0 else 0
        recent = p["pnls"][-50:]
        p["recent_pnl"] = sum(recent)
        p["recent_wr"] = len([x for x in recent if x > 0]) / len(recent) * 100 if recent else 0
        del p["pnls"]

    return perf


def _analyze_underperformers(performance, active):
    """Identifica pares activos con bajo rendimiento."""
    result = []
    for sym in active:
        p = performance.get(sym)
        if not p or p["trades"] < 5:
            continue
        issues = []
        severity = 0

        if p["profit_factor"] < 0.8 and p["trades"] >= 20:
            issues.append(f"Profit Factor muy bajo: {p['profit_factor']:.2f}")
            severity += 4
        elif p["profit_factor"] < 1.0 and p["trades"] >= 15:
            issues.append(f"PF negativo: {p['profit_factor']:.2f}")
            severity += 3

        if p["win_rate"] < 30 and p["trades"] >= 15:
            issues.append(f"Win Rate critico: {p['win_rate']:.1f}%")
            severity += 3
        elif p["win_rate"] < 38 and p["trades"] >= 20:
            issues.append(f"Win Rate bajo: {p['win_rate']:.1f}%")
            severity += 2

        if p["expectancy"] < -0.005 and p["trades"] >= 20:
            issues.append(f"Expectancy negativa: ${p['expectancy']:.4f}/trade")
            severity += 3

        if p.get("recent_pnl", 0) < -1.0:
            issues.append(f"Ultimos trades negativos: ${p['recent_pnl']:.2f}")
            severity += 2

        if issues:
            if severity >= 6:
                recommendation = f"Considerar quitar {sym}"
                urgency = "alta"
            elif severity >= 3:
                recommendation = f"Reducir capital de {sym}"
                urgency = "media"
            else:
                recommendation = f"Vigilar {sym}"
                urgency = "baja"
            result.append({
                "symbol": sym, "issues": issues, "severity": severity,
                "urgency": urgency, "recommendation": recommendation,
                "pnl": round(p["pnl"], 2), "trades": p["trades"],
                "win_rate": round(p["win_rate"], 1),
                "profit_factor": round(p["profit_factor"], 2),
                "recent_pnl": round(p.get("recent_pnl", 0), 2),
            })
    result.sort(key=lambda x: x["severity"], reverse=True)
    return result


# ─────────────────────────────────────────────
#  MARKET OVERVIEW
# ─────────────────────────────────────────────

def _compute_market_overview(pair_analyses):
    if not pair_analyses:
        return {"sentiment": "neutral", "avg_rsi": 50, "trending_pct": 0, "volatility_avg": 0}
    rsis = [p["rsi"] for p in pair_analyses.values()]
    regimes = [p["regime"] for p in pair_analyses.values()]
    vols = [p["atr_pct"] for p in pair_analyses.values()]
    avg_rsi = sum(rsis) / len(rsis)
    trending = len([r for r in regimes if r.startswith("trending")]) / len(regimes) * 100
    ranging = len([r for r in regimes if r == "ranging"]) / len(regimes) * 100
    avg_vol = sum(vols) / len(vols)

    if avg_rsi < 40: sentiment = "miedo"
    elif avg_rsi > 60: sentiment = "codicia"
    else: sentiment = "neutral"

    if ranging > 50 and 0.2 < avg_vol < 0.8: scalping_cond = "excelentes"
    elif ranging > 30: scalping_cond = "buenas"
    elif trending > 60: scalping_cond = "dificiles"
    else: scalping_cond = "normales"

    return {
        "sentiment": sentiment, "avg_rsi": round(avg_rsi, 1),
        "trending_pct": round(trending, 1), "ranging_pct": round(ranging, 1),
        "volatility_avg": round(avg_vol, 3), "scalping_conditions": scalping_cond,
        "total_scanned": len(pair_analyses),
    }


# ─────────────────────────────────────────────
#  SCAN PRINCIPAL
# ─────────────────────────────────────────────

def run_scan() -> Dict[str, Any]:
    active_pairs = _get_active_pairs()
    performance = _analyze_performance_from_log()
    underperformers = _analyze_underperformers(performance, active_pairs)

    top_pairs = _get_top_usdt_pairs()
    if not top_pairs:
        return {"error": "No se pudieron obtener pares del mercado"}

    pair_analyses: Dict[str, Dict] = {}
    suggestions = []

    for info in top_pairs:
        symbol = info["symbol"]
        candles = _get_klines(symbol)
        if len(candles) < 40:
            continue

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]
        price = closes[-1]

        rsi_val = _rsi(closes)
        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        ema_diff_pct = (ema9[-1] - ema21[-1]) / max(price, 1e-9) * 100 if ema9 and ema21 else 0
        macd_line, signal_line, macd_hist = _macd(closes)
        bb = _bollinger(closes)
        atr_val = _atr(highs, lows, closes)
        atr_pct = atr_val / max(price, 1e-9) * 100
        vol_prof = _volume_profile(volumes)
        regime = _detect_regime(closes, ema9, ema21, bb)
        volume_usdt_m = info["volume_usdt"] / 1_000_000

        scalping_score = _score_scalping_fitness(atr_pct, volume_usdt_m, vol_prof["consistency"], bb["width_pct"], regime, rsi_val)
        opportunity_score = _score_opportunity(rsi_val, macd_hist, ema_diff_pct, bb["position"], info["change_pct"], vol_prof["trend"])
        classification = _classify_opportunity(scalping_score, opportunity_score, regime)

        analysis = {
            "scalping_score": round(scalping_score, 1),
            "opportunity_score": round(opportunity_score, 1),
            "combined_score": classification["combined_score"],
            "category": classification["category"],
            "category_label": classification["label"],
            "category_emoji": classification["emoji"],
            "rsi": round(rsi_val, 1),
            "ema_diff_pct": round(ema_diff_pct, 3),
            "macd_hist": round(macd_hist, 6),
            "bb_position": round(bb["position"], 2),
            "bb_width": round(bb["width_pct"], 2),
            "atr_pct": round(atr_pct, 3),
            "regime": regime,
            "vol_trend": round(vol_prof["trend"], 2),
            "vol_consistency": round(vol_prof["consistency"], 2),
            "price": price,
            "change_pct": round(info["change_pct"] * 100, 2),
            "volume_usdt_m": round(volume_usdt_m, 1),
            "trades_24h": info["trades_24h"],
            "is_active": symbol in active_pairs,
        }

        if symbol in performance:
            p = performance[symbol]
            analysis["perf_pnl"] = round(p["pnl"], 2)
            analysis["perf_pf"] = round(p["profit_factor"], 2)
            analysis["perf_wr"] = round(p["win_rate"], 1)
            analysis["perf_trades"] = p["trades"]

        pair_analyses[symbol] = analysis

        _score_history.setdefault(symbol, [])
        _score_history[symbol].append({
            "time": time.strftime("%H:%M"),
            "combined": classification["combined_score"],
            "scalping": round(scalping_score, 1),
            "opportunity": round(opportunity_score, 1),
        })
        if len(_score_history[symbol]) > 30:
            _score_history[symbol] = _score_history[symbol][-30:]

        if classification["category"] in ("hot", "steady") and symbol not in active_pairs:
            reasons = []
            if rsi_val < 35: reasons.append(f"RSI en sobreventa ({rsi_val:.0f})")
            if macd_hist > 0: reasons.append("MACD positivo con impulso")
            if ema_diff_pct > 0.1: reasons.append("Tendencia alcista activa")
            if bb["position"] < 0.3: reasons.append("Precio cerca de banda inferior")
            if volume_usdt_m > 200: reasons.append(f"Alta liquidez (${volume_usdt_m:.0f}M)")
            if vol_prof["trend"] > 1.3: reasons.append(f"Volumen creciendo ({vol_prof['trend']:.1f}x)")
            if regime == "ranging": reasons.append("Mercado lateral ideal para scalping")
            if atr_pct > 0.25: reasons.append(f"Rango ATR aprovechable ({atr_pct:.2f}%)")

            suggestions.append({
                "type": "add_pair", "symbol": symbol,
                "category": classification["category"],
                "category_emoji": classification["emoji"],
                "category_label": classification["label"],
                "combined_score": classification["combined_score"],
                "scalping_score": round(scalping_score, 1),
                "opportunity_score": round(opportunity_score, 1),
                "reason": ". ".join(reasons) if reasons else classification["description"],
                "price": price, "rsi": round(rsi_val, 1), "regime": regime,
                "volume_usdt_m": round(volume_usdt_m, 1), "atr_pct": round(atr_pct, 3),
                "time": time.strftime("%H:%M:%S"),
            })

        time.sleep(0.12)

    for up in underperformers:
        suggestions.append({
            "type": "review_pair", "symbol": up["symbol"],
            "category": "underperformer", "category_emoji": "\u26a0\ufe0f",
            "category_label": "Bajo rendimiento", "combined_score": 0,
            "reason": " | ".join(up["issues"]),
            "recommendation": up["recommendation"], "urgency": up["urgency"],
            "pnl": up["pnl"], "trades": up["trades"],
            "win_rate": up["win_rate"], "profit_factor": up["profit_factor"],
            "time": time.strftime("%H:%M:%S"),
        })

    type_order = {"hot": 0, "steady": 1, "risky": 2, "watch": 3, "underperformer": 4}
    suggestions.sort(key=lambda x: (type_order.get(x.get("category", ""), 5), -x.get("combined_score", 0)))
    add_sug = [s for s in suggestions if s["type"] == "add_pair"][:8]
    rev_sug = [s for s in suggestions if s["type"] == "review_pair"]
    suggestions = add_sug + rev_sug

    market_overview = _compute_market_overview(pair_analyses)

    for s in suggestions:
        _alerts.append({
            "time": s["time"], "type": s["type"], "symbol": s["symbol"],
            "category": s.get("category", ""), "message": s.get("reason", ""),
        })

    return {
        "last_scan": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned_pairs": len(pair_analyses),
        "suggestions": suggestions, "pair_scores": pair_analyses,
        "underperformers": underperformers, "market_overview": market_overview,
        "active_pairs": active_pairs, "error": None, "running": False,
    }


# ─────────────────────────────────────────────
#  BACKGROUND THREAD
# ─────────────────────────────────────────────

_scanner_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _scanner_loop():
    _stop_event.wait(timeout=30)
    while not _stop_event.is_set():
        try:
            with _scanner_lock:
                _scanner_state["running"] = True
            result = run_scan()
            with _scanner_lock:
                _scanner_state.update(result)
        except Exception as e:
            with _scanner_lock:
                _scanner_state["error"] = str(e)
                _scanner_state["running"] = False
        _stop_event.wait(timeout=SCAN_INTERVAL_SECONDS)
    with _scanner_lock:
        _scanner_state["running"] = False


def start_scanner():
    global _scanner_thread
    if _scanner_thread and _scanner_thread.is_alive():
        return
    _stop_event.clear()
    _scanner_thread = threading.Thread(target=_scanner_loop, daemon=True, name="MarketScanner")
    _scanner_thread.start()


def stop_scanner():
    _stop_event.set()
    if _scanner_thread:
        _scanner_thread.join(timeout=5)


def get_scanner_state() -> Dict[str, Any]:
    with _scanner_lock:
        state = dict(_scanner_state)
    # Re-leer pares activos para filtrar los recien agregados
    active = _get_active_pairs()
    state["active_pairs"] = active
    # Quitar sugerencias de add_pair para pares que ya estan activos
    state["suggestions"] = [
        s for s in state.get("suggestions", [])
        if not (s["type"] == "add_pair" and s["symbol"] in active)
    ]
    # Actualizar is_active en pair_scores
    for sym, data in state.get("pair_scores", {}).items():
        data["is_active"] = sym in active
    return state


def get_scanner_alerts(limit: int = 30) -> List[Dict[str, Any]]:
    with _scanner_lock:
        return list(_alerts)[-limit:]


def get_score_history(symbol: str = "") -> Dict[str, List]:
    if symbol:
        return {symbol: _score_history.get(symbol, [])}
    return dict(_score_history)


def force_scan() -> Dict[str, Any]:
    result = run_scan()
    with _scanner_lock:
        _scanner_state.update(result)
    return result
