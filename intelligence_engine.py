"""
Intelligence Engine — Motor de inteligencia autonomo para el bot de scalping.

Capas de inteligencia (todas autonomas, configurables via runtime_config):
1. Horarios rentables: bloquea franjas horarias toxicas
2. Regimen de volatilidad: clasifica mercado (trending/ranging/volatile) via ATR
3. Filtro de volumen anomalo: detecta spikes y sequias de volumen
4. Score de confianza compuesto: 0-100 por senal, combina todo
5. Correlacion entre pares: limita entradas simultaneas correlacionadas
6. Trailing adaptativo por par: trailing = f(ATR del par)
7. Aprendizaje de exits: analiza cuanto se deja en la mesa, sugiere TP
8. ML signal filter: clasificador binario entrenado con ml_dataset.csv

Uso desde el bot:
    from intelligence_engine import evaluate_entry, get_adaptive_trailing, start_intelligence
    start_intelligence()  # arranca analisis background
    eval = evaluate_entry(symbol, context)
    if not eval["approved"]: signal = "HOLD"
    trailing = get_adaptive_trailing(symbol, candles)
"""

import json
import math
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────

TRADES_LOG = "trades.log"
RUNTIME_CONFIG = "runtime_config.json"
INTELLIGENCE_LOG = "intelligence_events.log"
ML_DATASET_FILE = "ml_dataset.csv"

# Defaults para las capas de inteligencia
DEFAULT_INTEL_CONFIG = {
    # Master switch
    "intel_enabled": True,
    # Horarios
    "intel_hours_enabled": True,
    "intel_hours_min_trades": 10,        # Min trades por franja para decidir
    "intel_hours_min_pf": 0.5,           # PF minimo para permitir franja
    # Regimen
    "intel_regime_enabled": True,
    "intel_regime_block_ranging": True,   # Bloquear en ranging
    "intel_regime_reduce_volatile": True, # Reducir size en volatil
    "intel_regime_volatile_reduce_pct": 0.5,  # Reducir a 50% del size
    # Volumen
    "intel_volume_enabled": True,
    "intel_volume_spike_mult": 3.0,      # >3x promedio = spike
    "intel_volume_drought_mult": 0.3,    # <0.3x promedio = sequia
    # Score de confianza
    "intel_score_enabled": True,
    "intel_score_min_buy": 40,           # Score minimo para BUY
    # Correlacion
    "intel_correlation_enabled": True,
    "intel_correlation_max_entries": 3,   # Max pares abiertos simultaneos
    # Trailing adaptativo
    "intel_adaptive_trailing_enabled": True,
    "intel_trailing_atr_mult": 1.5,      # trailing = ATR * mult
    # Exit learning
    "intel_exit_learning_enabled": True,
}

# ─────────────────────────────────────────────
#  ESTADO GLOBAL (thread-safe)
# ─────────────────────────────────────────────

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "started": False,
    "last_analysis": None,
    # Hora analisis
    "hourly_profile": {},         # {hora: {trades, wins, pf, pnl, blocked}}
    "blocked_hours": [],          # [0,1,2,...] horas UTC bloqueadas
    # Regimen por par
    "regimes": {},                # {symbol: {regime, atr, atr_pct, updated}}
    # Volumen por par
    "volume_state": {},           # {symbol: {avg_vol, current_vol, ratio, status}}
    # Correlacion
    "open_positions": set(),      # Pares con posicion abierta
    # Exit learning
    "exit_analysis": {},          # {symbol: {avg_left_on_table, suggested_tp, trades_analyzed}}
    # Trailing por par
    "adaptive_trailing": {},      # {symbol: trailing_pct}
    # Ultimo score por par
    "last_scores": {},            # {symbol: {score, components, timestamp}}
    # Contadores
    "signals_evaluated": 0,
    "signals_blocked": 0,
    "block_reasons": {},          # {reason: count}
}

_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def _read_config() -> Dict:
    try:
        with open(RUNTIME_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(cfg: Dict):
    with open(RUNTIME_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _log_intel(event_type: str, details: Dict):
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "type": event_type, **details}
    try:
        with open(INTELLIGENCE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_intel_config() -> Dict:
    """Retorna config de inteligencia mergeada con defaults."""
    cfg = _read_config()
    result = DEFAULT_INTEL_CONFIG.copy()
    for k, v in cfg.items():
        if k.startswith("intel_"):
            result[k] = v
    return result


def _parse_trades() -> List[Dict]:
    """Lee trades.log → lista de trades SELL."""
    if not os.path.exists(TRADES_LOG):
        return []
    trades = []
    try:
        with open(TRADES_LOG, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                row = {}
                for item in parts:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        row[k.strip()] = v.strip()
                    elif len(parts) >= 2 and item == parts[0]:
                        row["date"] = item
                    elif len(parts) >= 2 and item == parts[1]:
                        row["time"] = item
                if row.get("tipo") != "SELL":
                    continue
                try:
                    row["pnl_f"] = float(row.get("pnl", 0))
                except (ValueError, TypeError):
                    row["pnl_f"] = 0.0
                # Extraer hora del time field (HH:MM:SS)
                time_str = row.get("time", "")
                try:
                    row["hour"] = int(time_str.split(":")[0])
                except (ValueError, IndexError):
                    row["hour"] = -1
                trades.append(row)
    except Exception:
        pass
    return trades


# ─────────────────────────────────────────────
#  CAPA 1: HORARIOS RENTABLES
# ─────────────────────────────────────────────

def _analyze_hourly_performance(trades: List[Dict], min_trades: int, min_pf: float) -> Tuple[Dict, List[int]]:
    """Analiza rendimiento por hora UTC. Retorna perfil y horas bloqueadas."""
    by_hour: Dict[int, Dict] = {}
    for t in trades:
        h = t.get("hour", -1)
        if h < 0 or h > 23:
            continue
        if h not in by_hour:
            by_hour[h] = {"trades": 0, "wins": 0, "pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}
        by_hour[h]["trades"] += 1
        by_hour[h]["pnl"] += t["pnl_f"]
        if t["pnl_f"] > 0:
            by_hour[h]["wins"] += 1
            by_hour[h]["gross_profit"] += t["pnl_f"]
        elif t["pnl_f"] < 0:
            by_hour[h]["gross_loss"] += t["pnl_f"]

    profile = {}
    blocked = []
    for h in range(24):
        data = by_hour.get(h, {"trades": 0, "wins": 0, "pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0})
        pf = 0.0
        if data["trades"] > 0:
            gp = data["gross_profit"]
            gl = abs(data["gross_loss"])
            pf = gp / gl if gl > 0 else (gp if gp > 0 else 0)
        wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0

        is_blocked = False
        if data["trades"] >= min_trades and pf < min_pf:
            is_blocked = True
            blocked.append(h)

        profile[h] = {
            "trades": data["trades"],
            "wins": data["wins"],
            "wr": round(wr, 1),
            "pf": round(pf, 2),
            "pnl": round(data["pnl"], 3),
            "blocked": is_blocked,
        }

    return profile, blocked


def is_hour_blocked(hour: int) -> bool:
    """Retorna si una hora esta bloqueada."""
    with _lock:
        return hour in _state["blocked_hours"]


# ─────────────────────────────────────────────
#  CAPA 2: REGIMEN DE VOLATILIDAD
# ─────────────────────────────────────────────

def _calc_atr(candles: List[Dict], period: int = 14) -> float:
    """Calcula Average True Range desde candles OHLC."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["h"]
        l = candles[i]["l"]
        prev_c = candles[i - 1]["c"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    if not trs:
        return 0.0
    # EMA del TR
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (tr - atr) * (2 / (period + 1)) + atr
    return atr


def detect_regime(candles: List[Dict]) -> Dict[str, Any]:
    """
    Clasifica el regimen de mercado actual.
    Retorna: {regime, atr, atr_pct, details}
    regime: 'trending_up', 'trending_down', 'ranging', 'volatile'
    """
    if len(candles) < 30:
        return {"regime": "unknown", "atr": 0, "atr_pct": 0, "details": "datos insuficientes"}

    recent = candles[-30:]
    atr = _calc_atr(recent, 14)
    current_price = recent[-1]["c"]
    atr_pct = (atr / current_price * 100) if current_price > 0 else 0

    # Calcular tendencia: pendiente de los closes
    closes = [c["c"] for c in recent]
    n = len(closes)
    x_mean = (n - 1) / 2.0
    y_mean = sum(closes) / n
    num = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0
    slope_pct = (slope / y_mean * 100) if y_mean > 0 else 0

    # Rango: diferencia entre max y min relative
    high = max(c["h"] for c in recent)
    low = min(c["l"] for c in recent)
    range_pct = (high - low) / current_price * 100 if current_price > 0 else 0

    # Contar cambios de direccion (para detectar ranging/chop)
    direction_changes = 0
    for i in range(2, len(closes)):
        prev_dir = closes[i - 1] - closes[i - 2]
        curr_dir = closes[i] - closes[i - 1]
        if prev_dir * curr_dir < 0:
            direction_changes += 1
    choppiness = direction_changes / max(n - 2, 1)

    # Clasificacion
    if atr_pct > 0.8 and choppiness > 0.6:
        regime = "volatile"
    elif choppiness > 0.55 and abs(slope_pct) < 0.01:
        regime = "ranging"
    elif slope_pct > 0.005:
        regime = "trending_up"
    elif slope_pct < -0.005:
        regime = "trending_down"
    else:
        regime = "ranging"

    return {
        "regime": regime,
        "atr": round(atr, 4),
        "atr_pct": round(atr_pct, 4),
        "slope_pct": round(slope_pct, 6),
        "choppiness": round(choppiness, 3),
        "range_pct": round(range_pct, 4),
        "direction_changes": direction_changes,
    }


# ─────────────────────────────────────────────
#  CAPA 3: FILTRO DE VOLUMEN ANOMALO
# ─────────────────────────────────────────────

def analyze_volume(candles: List[Dict], spike_mult: float = 3.0, drought_mult: float = 0.3) -> Dict[str, Any]:
    """
    Analiza volumen de las ultimas velas.
    candles deben tener campo 'v' (volume).
    Retorna: {status, avg_vol, current_vol, ratio}
    status: 'normal', 'spike', 'drought', 'no_data'
    """
    volumes = [c.get("v", 0) for c in candles if c.get("v", 0) > 0]
    if len(volumes) < 10:
        return {"status": "no_data", "avg_vol": 0, "current_vol": 0, "ratio": 0}

    current = volumes[-1]
    # Promedio de las ultimas 20 velas (excluyendo la actual)
    lookback = volumes[-21:-1] if len(volumes) > 20 else volumes[:-1]
    avg = sum(lookback) / len(lookback) if lookback else 1

    ratio = current / avg if avg > 0 else 0

    if ratio >= spike_mult:
        status = "spike"
    elif ratio <= drought_mult:
        status = "drought"
    else:
        status = "normal"

    return {
        "status": status,
        "avg_vol": round(avg, 2),
        "current_vol": round(current, 2),
        "ratio": round(ratio, 2),
    }


# ─────────────────────────────────────────────
#  CAPA 4: SCORE DE CONFIANZA COMPUESTO
# ─────────────────────────────────────────────

def calculate_confidence_score(
    *,
    ema_fast: float,
    ema_slow: float,
    rsi: float,
    macd_hist: float,
    price: float,
    regime: str,
    volume_status: str,
    volume_ratio: float,
    hour: int,
    hourly_pf: float,
    atr_pct: float,
) -> Dict[str, Any]:
    """
    Calcula score 0-100 combinando multiples factores.
    Mayor score = mayor confianza en la senal.
    """
    components = {}

    # 1. Fuerza del cruce EMA (0-25 pts)
    if price > 0:
        ema_diff_pct = (ema_fast - ema_slow) / price * 100
    else:
        ema_diff_pct = 0
    # Cruce reciente y fuerte: 0.01-0.1% es ideal para scalping
    ema_score = min(25, max(0, ema_diff_pct * 250))  # 0.1% diff = 25pts
    components["ema_strength"] = round(ema_score, 1)

    # 2. RSI posicion (0-20 pts)
    # Ideal para compra: RSI 35-55 (zona de valor sin sobreventa extrema)
    if 35 <= rsi <= 55:
        rsi_score = 20
    elif 30 <= rsi <= 60:
        rsi_score = 15
    elif 25 <= rsi <= 65:
        rsi_score = 10
    else:
        rsi_score = max(0, 20 - abs(rsi - 45) * 0.5)
    components["rsi_position"] = round(rsi_score, 1)

    # 3. MACD momentum (0-15 pts)
    if macd_hist > 0:
        macd_score = min(15, macd_hist * 1000)  # Escalar
    else:
        macd_score = 0
    components["macd_momentum"] = round(macd_score, 1)

    # 4. Regimen (0-15 pts)
    regime_scores = {
        "trending_up": 15,
        "trending_down": 0,
        "ranging": 3,
        "volatile": 5,
        "unknown": 7,
    }
    regime_score = regime_scores.get(regime, 7)
    components["regime"] = regime_score

    # 5. Volumen (0-10 pts)
    if volume_status == "normal":
        vol_score = 10
    elif volume_status == "spike":
        vol_score = 2  # Peligroso
    elif volume_status == "drought":
        vol_score = 3  # Sin liquidez
    else:
        vol_score = 7  # Sin data, neutro
    components["volume"] = vol_score

    # 6. Hora historica (0-10 pts)
    if hourly_pf >= 1.5:
        hour_score = 10
    elif hourly_pf >= 1.0:
        hour_score = 7
    elif hourly_pf >= 0.5:
        hour_score = 3
    else:
        hour_score = 0
    components["hour_performance"] = hour_score

    # 7. Volatilidad ATR (0-5 pts)
    # ATR ideal para scalping: 0.1-0.4%
    if 0.1 <= atr_pct <= 0.4:
        atr_score = 5
    elif 0.05 <= atr_pct <= 0.7:
        atr_score = 3
    else:
        atr_score = 1
    components["atr_range"] = atr_score

    total = sum(components.values())
    # Normalizar a 0-100
    max_possible = 100
    normalized = min(100, round(total / max_possible * 100))

    return {
        "score": normalized,
        "total_raw": round(total, 1),
        "components": components,
        "grade": _score_grade(normalized),
    }


def _score_grade(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "F"


# ─────────────────────────────────────────────
#  CAPA 5: CORRELACION ENTRE PARES
# ─────────────────────────────────────────────

def register_position_open(symbol: str):
    """Registra que un par abrio posicion."""
    with _lock:
        _state["open_positions"].add(symbol)


def register_position_close(symbol: str):
    """Registra que un par cerro posicion."""
    with _lock:
        _state["open_positions"].discard(symbol)


def get_open_position_count() -> int:
    with _lock:
        return len(_state["open_positions"])


def is_correlation_blocked(symbol: str, max_entries: int) -> bool:
    """Retorna True si ya hay demasiados pares abiertos."""
    with _lock:
        current = len(_state["open_positions"])
        # Si el par ya tiene posicion, no bloquear (es re-evaluacion)
        if symbol in _state["open_positions"]:
            return False
        return current >= max_entries


# ─────────────────────────────────────────────
#  CAPA 6: TRAILING ADAPTATIVO POR PAR
# ─────────────────────────────────────────────

def get_adaptive_trailing(symbol: str, candles: List[Dict], atr_mult: float = 1.5) -> float:
    """
    Calcula trailing stop basado en ATR del par.
    Retorna trailing_stop_pct (ej: 0.004 = 0.4%).
    Pares volatiles → trailing mas ancho.
    Pares estables → trailing mas tight.
    """
    if len(candles) < 20:
        return 0.004  # Default

    atr = _calc_atr(candles[-30:] if len(candles) >= 30 else candles, 14)
    price = candles[-1]["c"]
    if price <= 0:
        return 0.004

    atr_pct = atr / price
    trailing = atr_pct * atr_mult

    # Limites
    trailing = max(0.002, min(0.015, trailing))

    with _lock:
        _state["adaptive_trailing"][symbol] = round(trailing, 5)

    return round(trailing, 5)


# ─────────────────────────────────────────────
#  CAPA 7: APRENDIZAJE DE EXITS
# ─────────────────────────────────────────────

def _analyze_exits(trades: List[Dict]) -> Dict[str, Dict]:
    """
    Analiza trades para detectar si se esta saliendo demasiado temprano o tarde.
    Calcula 'left on table' basado en razon de cierre y PnL.

    Modelo simplificado:
    - Si TP tiene PnL alto y constante → el TP funciona
    - Si signal exit tiene PnL bajo → se esta cortando proffit
    - Si SL domina las salidas → entradas son malas
    - Si trailing domina con PnL negativo → trailing esta atrapando
    """
    by_symbol: Dict[str, Dict] = {}

    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        reason = t.get("razon", t.get("reason", "unknown"))
        pnl = t["pnl_f"]

        if sym not in by_symbol:
            by_symbol[sym] = {
                "tp_trades": [], "sl_trades": [], "signal_trades": [], "trailing_trades": [],
                "total_trades": 0, "total_pnl": 0.0,
            }
        by_symbol[sym]["total_trades"] += 1
        by_symbol[sym]["total_pnl"] += pnl

        if "take" in reason.lower() or "tp" in reason.lower() or "profit" in reason.lower():
            by_symbol[sym]["tp_trades"].append(pnl)
        elif "stop" in reason.lower() and "trailing" not in reason.lower():
            by_symbol[sym]["sl_trades"].append(pnl)
        elif "trailing" in reason.lower():
            by_symbol[sym]["trailing_trades"].append(pnl)
        else:
            by_symbol[sym]["signal_trades"].append(pnl)

    analysis = {}
    for sym, data in by_symbol.items():
        if data["total_trades"] < 5:
            continue

        tp_avg = sum(data["tp_trades"]) / len(data["tp_trades"]) if data["tp_trades"] else 0
        sl_avg = sum(data["sl_trades"]) / len(data["sl_trades"]) if data["sl_trades"] else 0
        sig_avg = sum(data["signal_trades"]) / len(data["signal_trades"]) if data["signal_trades"] else 0
        trail_avg = sum(data["trailing_trades"]) / len(data["trailing_trades"]) if data["trailing_trades"] else 0

        tp_count = len(data["tp_trades"])
        sl_count = len(data["sl_trades"])
        sig_count = len(data["signal_trades"])
        trail_count = len(data["trailing_trades"])
        total = data["total_trades"]

        suggestions = []

        # Si TP es exitoso y frecuente → podemos dejarlo correr mas
        if tp_count > 0 and tp_avg > 0 and tp_count / total > 0.3:
            if tp_avg > abs(sl_avg) * 0.5:
                suggestions.append({
                    "action": "widen_tp",
                    "reason": f"TP exitoso (avg +${tp_avg:.3f}, {tp_count}/{total} trades). Puede dejarse correr mas.",
                    "confidence": min(0.9, tp_count / total),
                })

        # Si signal exits dominan con PnL bajo → podria mejorar con TP mas bajo
        if sig_count > 0 and sig_avg < tp_avg * 0.3 and sig_count / total > 0.3:
            suggestions.append({
                "action": "tighten_tp",
                "reason": f"Salidas por senal (avg ${sig_avg:.3f}) dejan ${tp_avg - sig_avg:.3f} vs TP. TP mas bajo capturaria mas.",
                "confidence": 0.6,
            })

        # Si SL domina → las entradas son malas o SL muy tight
        if sl_count > 0 and sl_count / total > 0.4:
            suggestions.append({
                "action": "review_entries",
                "reason": f"SL domina ({sl_count}/{total} trades, avg ${sl_avg:.3f}). Posibles malas entradas.",
                "confidence": 0.7,
            })

        # Si trailing pierde dinero → desactivar o ampliar
        if trail_count >= 3 and trail_avg < 0:
            suggestions.append({
                "action": "widen_trailing",
                "reason": f"Trailing pierde (avg ${trail_avg:.3f} en {trail_count} trades). Ampliar o desactivar.",
                "confidence": 0.8,
            })

        analysis[sym] = {
            "total_trades": total,
            "tp": {"count": tp_count, "avg_pnl": round(tp_avg, 4), "pct": round(tp_count / total * 100, 1)},
            "sl": {"count": sl_count, "avg_pnl": round(sl_avg, 4), "pct": round(sl_count / total * 100, 1)},
            "signal": {"count": sig_count, "avg_pnl": round(sig_avg, 4), "pct": round(sig_count / total * 100, 1)},
            "trailing": {"count": trail_count, "avg_pnl": round(trail_avg, 4), "pct": round(trail_count / total * 100, 1)},
            "suggestions": suggestions,
            "total_pnl": round(data["total_pnl"], 4),
        }

    return analysis


# ─────────────────────────────────────────────
#  EVALUACION PRINCIPAL DE ENTRADA
# ─────────────────────────────────────────────

def evaluate_entry(
    symbol: str,
    *,
    price: float,
    ema_fast: float,
    ema_slow: float,
    rsi: float,
    macd_hist: float,
    candles: List[Dict],
    raw_signal: str,
) -> Dict[str, Any]:
    """
    Evaluacion principal: combina TODAS las capas de inteligencia.
    Retorna dict con:
    - approved: bool (True = dejar pasar, False = bloquear)
    - score: int (0-100)
    - blocks: list de razones de bloqueo
    - adjustments: dict de ajustes sugeridos (trade_pct, trailing)
    - details: dict con info de cada capa
    """
    icfg = _get_intel_config()

    if not icfg.get("intel_enabled", True):
        return {
            "approved": True,
            "score": 50,
            "blocks": [],
            "adjustments": {},
            "details": {"intel_disabled": True},
        }

    blocks = []
    adjustments = {}
    details = {}
    current_hour = int(time.strftime("%H"))

    # ── Capa 1: Horarios ──
    if icfg.get("intel_hours_enabled", True):
        with _lock:
            profile = _state.get("hourly_profile", {})
            blocked_hours = _state.get("blocked_hours", [])
        if current_hour in blocked_hours:
            hour_data = profile.get(current_hour, {})
            blocks.append(f"hora_toxica:{current_hour}h (PF={hour_data.get('pf', 0)}, PnL=${hour_data.get('pnl', 0)})")
        hourly_pf = profile.get(current_hour, {}).get("pf", 1.0)
        details["hour"] = {"current": current_hour, "pf": hourly_pf, "blocked": current_hour in blocked_hours}
    else:
        hourly_pf = 1.0
        details["hour"] = {"disabled": True}

    # ── Capa 2: Regimen de volatilidad ──
    if icfg.get("intel_regime_enabled", True) and candles:
        regime_data = detect_regime(candles)
        regime = regime_data["regime"]
        atr_pct = regime_data.get("atr_pct", 0)

        with _lock:
            _state["regimes"][symbol] = {**regime_data, "updated": time.strftime("%H:%M:%S")}

        if regime == "ranging" and icfg.get("intel_regime_block_ranging", True):
            blocks.append(f"regimen_ranging (chop={regime_data.get('choppiness', 0)}, slope={regime_data.get('slope_pct', 0)})")

        if regime == "volatile" and icfg.get("intel_regime_reduce_volatile", True):
            reduce = icfg.get("intel_regime_volatile_reduce_pct", 0.5)
            adjustments["trade_pct_mult"] = reduce

        details["regime"] = regime_data
    else:
        regime = "unknown"
        atr_pct = 0
        details["regime"] = {"disabled": True}

    # ── Capa 3: Volumen ──
    if icfg.get("intel_volume_enabled", True) and candles:
        vol_data = analyze_volume(
            candles,
            spike_mult=icfg.get("intel_volume_spike_mult", 3.0),
            drought_mult=icfg.get("intel_volume_drought_mult", 0.3),
        )
        with _lock:
            _state["volume_state"][symbol] = vol_data

        if vol_data["status"] == "spike":
            blocks.append(f"volumen_spike (ratio={vol_data['ratio']}x, posible manipulacion)")
        elif vol_data["status"] == "drought":
            blocks.append(f"volumen_sequia (ratio={vol_data['ratio']}x, baja liquidez)")

        details["volume"] = vol_data
    else:
        vol_data = {"status": "no_data", "ratio": 1.0}
        details["volume"] = {"disabled": True}

    # ── Capa 5: Correlacion ──
    if icfg.get("intel_correlation_enabled", True):
        max_entries = icfg.get("intel_correlation_max_entries", 3)
        if is_correlation_blocked(symbol, max_entries):
            count = get_open_position_count()
            blocks.append(f"correlacion_max ({count}/{max_entries} pares abiertos)")
        details["correlation"] = {
            "open_count": get_open_position_count(),
            "max": max_entries,
            "blocked": is_correlation_blocked(symbol, max_entries),
        }
    else:
        details["correlation"] = {"disabled": True}

    # ── Capa 6: Trailing adaptativo ──
    if icfg.get("intel_adaptive_trailing_enabled", True) and candles:
        mult = icfg.get("intel_trailing_atr_mult", 1.5)
        adaptive_trail = get_adaptive_trailing(symbol, candles, mult)
        adjustments["adaptive_trailing"] = adaptive_trail
        details["adaptive_trailing"] = {"value": adaptive_trail, "atr_mult": mult}
    else:
        details["adaptive_trailing"] = {"disabled": True}

    # ── Capa 4: Score de confianza (usa datos de todas las otras capas) ──
    if icfg.get("intel_score_enabled", True):
        score_data = calculate_confidence_score(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi,
            macd_hist=macd_hist,
            price=price,
            regime=regime,
            volume_status=vol_data.get("status", "no_data"),
            volume_ratio=vol_data.get("ratio", 1.0),
            hour=current_hour,
            hourly_pf=hourly_pf,
            atr_pct=atr_pct,
        )
        min_score = icfg.get("intel_score_min_buy", 40)
        if score_data["score"] < min_score:
            blocks.append(f"score_bajo ({score_data['score']}/{min_score} min, grade={score_data['grade']})")

        with _lock:
            _state["last_scores"][symbol] = {
                **score_data,
                "timestamp": time.strftime("%H:%M:%S"),
            }
        details["confidence"] = score_data
    else:
        score_data = {"score": 50, "grade": "?"}
        details["confidence"] = {"disabled": True}

    # ── Resultado final ──
    approved = len(blocks) == 0

    with _lock:
        _state["signals_evaluated"] += 1
        if not approved:
            _state["signals_blocked"] += 1
            for b in blocks:
                reason_key = b.split(":")[0].split(" ")[0]
                _state["block_reasons"][reason_key] = _state["block_reasons"].get(reason_key, 0) + 1

    if not approved:
        _log_intel("entry_blocked", {
            "symbol": symbol,
            "price": price,
            "signal": raw_signal,
            "score": score_data.get("score", 0),
            "blocks": blocks,
        })

    return {
        "approved": approved,
        "score": score_data.get("score", 50),
        "grade": score_data.get("grade", "?"),
        "blocks": blocks,
        "adjustments": adjustments,
        "details": details,
    }


# ─────────────────────────────────────────────
#  BACKGROUND ANALYSIS THREAD
# ─────────────────────────────────────────────

def _run_background_analysis():
    """Actualiza analisis periodicamente (cada 5 min)."""
    _stop_event.wait(30)  # Esperar 30s al inicio
    while not _stop_event.is_set():
        try:
            icfg = _get_intel_config()

            if not icfg.get("intel_enabled", True):
                _stop_event.wait(60)
                continue

            trades = _parse_trades()

            # Actualizar perfil horario
            if icfg.get("intel_hours_enabled", True):
                min_trades = icfg.get("intel_hours_min_trades", 10)
                min_pf = icfg.get("intel_hours_min_pf", 0.5)
                profile, blocked = _analyze_hourly_performance(trades, min_trades, min_pf)
                with _lock:
                    old_blocked = set(_state["blocked_hours"])
                    _state["hourly_profile"] = profile
                    _state["blocked_hours"] = blocked
                    new_blocked = set(blocked)
                if new_blocked != old_blocked:
                    _log_intel("hours_updated", {
                        "blocked": blocked,
                        "added": list(new_blocked - old_blocked),
                        "removed": list(old_blocked - new_blocked),
                    })

            # Actualizar exit analysis
            if icfg.get("intel_exit_learning_enabled", True):
                exit_data = _analyze_exits(trades)
                with _lock:
                    _state["exit_analysis"] = exit_data

            with _lock:
                _state["last_analysis"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        except Exception as e:
            _log_intel("analysis_error", {"error": str(e)})

        _stop_event.wait(300)  # Cada 5 minutos


def start_intelligence():
    """Inicia el motor de inteligencia en background."""
    global _thread
    if _thread and _thread.is_alive():
        return

    with _lock:
        _state["started"] = True

    _stop_event.clear()
    _thread = threading.Thread(target=_run_background_analysis, daemon=True, name="IntelligenceEngine")
    _thread.start()
    _log_intel("started", {})


def stop_intelligence():
    """Detiene el motor de inteligencia."""
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
    with _lock:
        _state["started"] = False


def get_intelligence_state() -> Dict[str, Any]:
    """Retorna el estado completo del motor de inteligencia (para dashboard)."""
    icfg = _get_intel_config()
    with _lock:
        state = {
            "enabled": icfg.get("intel_enabled", True),
            "started": _state["started"],
            "last_analysis": _state["last_analysis"],
            "signals_evaluated": _state["signals_evaluated"],
            "signals_blocked": _state["signals_blocked"],
            "block_rate": round(
                _state["signals_blocked"] / max(_state["signals_evaluated"], 1) * 100, 1
            ),
            "block_reasons": dict(_state["block_reasons"]),
            "layers": {
                "hours": {
                    "enabled": icfg.get("intel_hours_enabled", True),
                    "blocked_hours": list(_state["blocked_hours"]),
                    "profile": dict(_state["hourly_profile"]),
                },
                "regime": {
                    "enabled": icfg.get("intel_regime_enabled", True),
                    "by_pair": {k: dict(v) for k, v in _state["regimes"].items()},
                },
                "volume": {
                    "enabled": icfg.get("intel_volume_enabled", True),
                    "by_pair": {k: dict(v) for k, v in _state["volume_state"].items()},
                },
                "confidence": {
                    "enabled": icfg.get("intel_score_enabled", True),
                    "min_score": icfg.get("intel_score_min_buy", 40),
                    "last_scores": {k: dict(v) for k, v in _state["last_scores"].items()},
                },
                "correlation": {
                    "enabled": icfg.get("intel_correlation_enabled", True),
                    "max_entries": icfg.get("intel_correlation_max_entries", 3),
                    "open_positions": list(_state["open_positions"]),
                },
                "adaptive_trailing": {
                    "enabled": icfg.get("intel_adaptive_trailing_enabled", True),
                    "by_pair": dict(_state["adaptive_trailing"]),
                    "atr_mult": icfg.get("intel_trailing_atr_mult", 1.5),
                },
                "exit_learning": {
                    "enabled": icfg.get("intel_exit_learning_enabled", True),
                    "analysis": {k: dict(v) for k, v in _state["exit_analysis"].items()},
                },
            },
            "config": icfg,
        }
    return state


def get_intel_summary_for_pair(symbol: str) -> Dict[str, Any]:
    """Resumen rapido de inteligencia para un par (para el estado runtime)."""
    with _lock:
        return {
            "regime": _state["regimes"].get(symbol, {}).get("regime", "unknown"),
            "atr_pct": _state["regimes"].get(symbol, {}).get("atr_pct", 0),
            "volume_status": _state["volume_state"].get(symbol, {}).get("status", "no_data"),
            "volume_ratio": _state["volume_state"].get(symbol, {}).get("ratio", 0),
            "confidence_score": _state["last_scores"].get(symbol, {}).get("score", None),
            "confidence_grade": _state["last_scores"].get(symbol, {}).get("grade", "?"),
            "adaptive_trailing": _state["adaptive_trailing"].get(symbol, None),
            "open_positions_count": len(_state["open_positions"]),
            "hour_blocked": int(time.strftime("%H")) in _state["blocked_hours"],
        }
