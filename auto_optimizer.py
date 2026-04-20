"""
Auto-Optimizer Inteligente — Ajusta estrategia en base a rendimiento real.

Cada N minutos (configurable):
1. Lee trades.log y calcula metricas por par y globales
2. Analiza razones de cierre (trailing, TP, SL, senal)
3. Toma decisiones:
   - Quitar pares toxicos
   - Ajustar TP/SL/trailing segun distribucion real de trades
   - Rebalancear capital hacia pares rentables
4. Aplica cambios a runtime_config.json
5. Registra todas las acciones en optimizer_actions.log
"""

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────

RUNTIME_CONFIG_FILE = "runtime_config.json"
TRADES_LOG_FILE = "trades.log"
OPTIMIZER_LOG_FILE = "optimizer_actions.log"
OPTIMIZER_STATE_FILE = "optimizer_state.json"  # Persistir estado entre reinicios

# Umbrales de decision
MIN_TRADES_FOR_DECISION = 5      # Min trades para tomar decisiones sobre un par
MIN_TRADES_GLOBAL = 7            # Min trades globales para ajustar parametros

# Limites de parametros (ajustados para scalping)
TP_MIN = 0.003      # 0.3% — minimo viable cubriendo fees
TP_MAX = 0.012      # 1.2% — maximo realista en scalping 1m
SL_MIN = 0.002      # 0.2%
SL_MAX = 0.006      # 0.6% — max sensato para scalping
TRAILING_MIN = 0.002  # 0.2% (minimo funcional)
TRAILING_MAX = 0.008  # 0.8% — no mas que SL

# Coherencia risk:reward
MIN_RR_RATIO = 0.6   # TP debe ser >= SL * MIN_RR_RATIO

# Cooldown: minimo horas entre ajustes del mismo parametro en la misma direccion
PARAM_COOLDOWN_HOURS = 2

# Limites de rebalanceo
MIN_ALLOC_PCT = 0.05   # 5% minimo del balance por par
MAX_ALLOC_PCT = 0.40   # 40% maximo del balance por par
MAX_REBALANCE_SHIFT_PCT = 20  # Max % de cambio por par por ciclo (evita rebalanceos extremos)
MIN_PAIRS = 2           # Nunca bajar de 2 pares

# ─────────────────────────────────────────────
#  ESTADO Y THREAD CONTROL
# ─────────────────────────────────────────────

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "enabled": False,
    "interval_minutes": 10,
    "last_run": None,
    "last_actions": [],
    "history": [],         # Ultimas 50 ejecuciones
    "running": False,
    "error": None,
    "cycle_count": 0,
    "last_trades_hash": "",  # Hash de trades para detectar datos nuevos
    "param_last_adjusted": {},  # {"take_profit_pct:tighten": "2026-04-17T12:00:00", ...}
}

_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _load_persistent_state():
    """Carga estado persistente (hash, cooldowns) desde archivo."""
    if os.path.exists(OPTIMIZER_STATE_FILE):
        try:
            with open(OPTIMIZER_STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            with _lock:
                _state["last_trades_hash"] = saved.get("last_trades_hash", "")
                _state["param_last_adjusted"] = saved.get("param_last_adjusted", {})
        except Exception:
            pass


def _save_persistent_state():
    """Guarda estado persistente a archivo."""
    with _lock:
        saved = {
            "last_trades_hash": _state["last_trades_hash"],
            "param_last_adjusted": _state["param_last_adjusted"],
        }
    try:
        with open(OPTIMIZER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _can_adjust_param(param_name: str, direction: str) -> bool:
    """Verifica si un parametro puede ser ajustado (respeta cooldown)."""
    key = f"{param_name}:{direction}"
    with _lock:
        last_ts = _state["param_last_adjusted"].get(key)
    if not last_ts:
        return True
    try:
        from datetime import datetime
        last = datetime.fromisoformat(last_ts)
        now = datetime.now()
        hours_since = (now - last).total_seconds() / 3600
        return hours_since >= PARAM_COOLDOWN_HOURS
    except Exception:
        return True


def _mark_param_adjusted(param_name: str, direction: str):
    """Registra que un parametro fue ajustado ahora."""
    key = f"{param_name}:{direction}"
    with _lock:
        _state["param_last_adjusted"][key] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_persistent_state()


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

def _log_action(action_type: str, details: Dict[str, Any]):
    """Registra una accion del optimizer en archivo."""
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": action_type,
        **details,
    }
    try:
        with open(OPTIMIZER_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return entry


# ─────────────────────────────────────────────
#  LECTURA DE DATOS
# ─────────────────────────────────────────────

def _read_config() -> Dict[str, Any]:
    try:
        with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(cfg: Dict[str, Any]):
    # Asegurar limites criticos antes de escribir
    if "trailing_stop_pct" in cfg:
        cfg["trailing_stop_pct"] = min(max(float(cfg["trailing_stop_pct"]), TRAILING_MIN), TRAILING_MAX)
    if "take_profit_pct" in cfg:
        cfg["take_profit_pct"] = min(max(float(cfg["take_profit_pct"]), TP_MIN), TP_MAX)
    if "stop_loss_pct" in cfg:
        cfg["stop_loss_pct"] = min(max(float(cfg["stop_loss_pct"]), SL_MIN), SL_MAX)
    with open(RUNTIME_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _parse_trades() -> List[Dict[str, Any]]:
    """Lee trades.log y retorna lista de trades SELL con sus datos."""
    if not os.path.exists(TRADES_LOG_FILE):
        return []
    trades = []
    try:
        with open(TRADES_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 4:
                    continue
                row = {}
                for item in parts:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        row[k.strip()] = v.strip()
                if row.get("tipo") != "SELL":
                    continue
                try:
                    row["pnl_f"] = float(row.get("pnl", 0))
                except (ValueError, TypeError):
                    row["pnl_f"] = 0.0
                trades.append(row)
    except Exception:
        pass
    return trades


def _compute_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """Calcula metricas globales y por par desde trades."""
    if not trades:
        return {"global": {}, "by_symbol": {}, "by_reason": {}}

    # Global
    pnls = [t["pnl_f"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)

    global_metrics = {
        "trades": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(pnls) * 100 if pnls else 0,
        "pnl": sum(pnls),
        "pf": gross_profit / abs(gross_loss) if gross_loss < 0 else (gross_profit if gross_profit > 0 else 0),
        "avg_win": gross_profit / len(wins) if wins else 0,
        "avg_loss": abs(gross_loss) / len(losses) if losses else 0,
        "best": max(pnls) if pnls else 0,
        "worst": min(pnls) if pnls else 0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }

    # Por simbolo
    by_symbol: Dict[str, Dict] = {}
    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "pnls": []}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"] += t["pnl_f"]
        by_symbol[sym]["pnls"].append(t["pnl_f"])
        if t["pnl_f"] > 0:
            by_symbol[sym]["wins"] += 1
        elif t["pnl_f"] < 0:
            by_symbol[sym]["losses"] += 1

    for sym, data in by_symbol.items():
        data["wr"] = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
        gp = sum(p for p in data["pnls"] if p > 0)
        gl = sum(p for p in data["pnls"] if p < 0)
        data["pf"] = gp / abs(gl) if gl < 0 else (gp if gp > 0 else 0)
        data["avg_win"] = gp / data["wins"] if data["wins"] > 0 else 0
        data["avg_loss"] = abs(gl) / data["losses"] if data["losses"] > 0 else 0
        data["expectancy"] = data["pnl"] / data["trades"] if data["trades"] > 0 else 0
        # Score compuesto: PF ponderado por raiz de trades (mas trades = mas confianza)
        data["score"] = data["pf"] * math.sqrt(data["trades"]) if data["trades"] >= 3 else 0
        del data["pnls"]

    # Por razon de cierre
    by_reason: Dict[str, Dict] = {}
    for t in trades:
        reason = t.get("razon", t.get("reason", "unknown"))
        if reason not in by_reason:
            by_reason[reason] = {"count": 0, "pnl": 0.0}
        by_reason[reason]["count"] += 1
        by_reason[reason]["pnl"] += t["pnl_f"]

    return {"global": global_metrics, "by_symbol": by_symbol, "by_reason": by_reason}


# ─────────────────────────────────────────────
#  DECISIONES INTELIGENTES
# ─────────────────────────────────────────────

def _decide_pair_removal(by_symbol: Dict, active_pairs: List[str]) -> Optional[Dict]:
    """Decide si hay un par toxico que quitar."""
    if len(active_pairs) <= MIN_PAIRS:
        return None

    worst = None
    worst_severity = 0

    for sym in active_pairs:
        data = by_symbol.get(sym)
        if not data or data["trades"] < MIN_TRADES_FOR_DECISION:
            continue

        severity = 0
        reasons = []
        n = data["trades"]

        # PF catastrofico (umbrales escalonados por cantidad de trades)
        if data["pf"] < 0.3 and n >= 5:
            severity += 5
            reasons.append(f"PF catastrofico: {data['pf']:.2f}")
        elif data["pf"] < 0.5 and n >= 8:
            severity += 4
            reasons.append(f"PF muy bajo: {data['pf']:.2f}")
        elif data["pf"] < 0.6 and n >= 15:
            severity += 3
            reasons.append(f"PF bajo: {data['pf']:.2f}")

        # WR catastrofico
        if data["wr"] < 20 and n >= 5:
            severity += 4
            reasons.append(f"WR destruido: {data['wr']:.1f}%")
        elif data["wr"] < 30 and n >= 8:
            severity += 3
            reasons.append(f"WR muy bajo: {data['wr']:.1f}%")

        # Expectancy muy negativa
        if data["expectancy"] < -0.1 and n >= 5:
            severity += 3
            reasons.append(f"Expectancy negativa: ${data['expectancy']:.3f}/trade")

        # PnL absoluto muy negativo
        if data["pnl"] < -1.0:
            severity += 2
            reasons.append(f"PnL acumulado: ${data['pnl']:.2f}")

        if severity >= 7 and (worst is None or severity > worst_severity):
            worst = {"symbol": sym, "severity": severity, "reasons": reasons, **data}
            worst_severity = severity

    return worst


def _decide_trailing_adjustment(by_reason: Dict, current_trailing: float, global_metrics: Dict) -> Optional[Dict]:
    """Decide si ajustar el trailing stop."""
    trailing_data = None
    for reason, data in by_reason.items():
        if "trailing" in reason.lower():
            trailing_data = data
            break

    if not trailing_data:
        return None

    trailing_pnl = trailing_data["pnl"]
    trailing_count = trailing_data["count"]

    # Si el trailing esta generando perdida consistente
    if trailing_count >= 3 and trailing_pnl < -0.15:
        # Ampliar trailing (nunca desactivar a 0.0)
        if current_trailing < TRAILING_MAX and _can_adjust_param("trailing_stop_pct", "widen"):
            new_val = min(current_trailing + 0.002, TRAILING_MAX)
            return {
                "action": "widen_trailing",
                "reason": f"Trailing stop genera ${trailing_pnl:.2f} en {trailing_count} activaciones. Ampliando.",
                "new_value": max(new_val, TRAILING_MIN),
            }
        return None  # Ya esta al maximo o en cooldown
    elif trailing_count >= 2 and trailing_pnl < -0.05 and current_trailing > 0:
        if _can_adjust_param("trailing_stop_pct", "widen"):
            new_val = min(current_trailing + 0.002, TRAILING_MAX)
            if new_val != current_trailing:
                return {
                    "action": "widen_trailing",
                    "reason": f"Trailing ajustado (${trailing_pnl:.2f} en {trailing_count} act). Ampliar de {current_trailing:.4f} a {new_val:.4f}",
                    "new_value": new_val,
                }
    elif trailing_count >= 10 and trailing_pnl > 0:
        if _can_adjust_param("trailing_stop_pct", "tighten"):
            new_val = max(current_trailing - 0.001, TRAILING_MIN + 0.002)
            if new_val != current_trailing:
                return {
                    "action": "tighten_trailing",
                    "reason": f"Trailing rentable (+${trailing_pnl:.2f}). Ajustar de {current_trailing:.4f} a {new_val:.4f}",
                    "new_value": new_val,
                }

    return None


def _decide_tp_sl_adjustment(global_metrics: Dict, current_tp: float, current_sl: float, by_reason: Optional[Dict] = None) -> List[Dict]:
    """Decide si ajustar TP y SL."""
    actions = []

    if global_metrics["trades"] < MIN_TRADES_GLOBAL:
        return actions

    avg_win = global_metrics["avg_win"]
    avg_loss = global_metrics["avg_loss"]
    pf = global_metrics["pf"]

    # Si las perdidas promedio son mucho mayores que las ganancias -> apretar SL
    if avg_loss > 0 and avg_win > 0:
        ratio = avg_loss / avg_win
        if ratio > 2.0 and current_sl > SL_MIN and _can_adjust_param("stop_loss_pct", "tighten"):
            new_sl = max(current_sl - 0.001, SL_MIN)
            if new_sl != current_sl:
                actions.append({
                    "param": "stop_loss_pct",
                    "action": "tighten_sl",
                    "old": current_sl,
                    "new": new_sl,
                    "reason": f"Perdida promedio (${avg_loss:.3f}) es {ratio:.1f}x mayor que ganancia (${avg_win:.3f}). Apretar SL.",
                })
        # NOTA: Ya no se amplia SL automaticamente. El drift SL era catastrofico.
        # Solo se ajusta SL hacia abajo (mas seguro). Ampliarlo requiere grid search.

    # Ajustar TP basado en hit rate y distribucion real
    if global_metrics["trades"] >= MIN_TRADES_GLOBAL:
        tp_hits = by_reason.get("take_profit", {}).get("count", 0) if by_reason else 0
        tp_hit_rate = tp_hits / global_metrics["trades"] * 100 if global_metrics["trades"] > 0 else 0

        if tp_hit_rate < 5 and global_metrics["trades"] >= 10 and current_tp > TP_MIN:
            # TP casi nunca se alcanza -> bajarlo para capturar profit reales
            # Paso agresivo (-0.002) si PF < 0.5, moderado (-0.001) si PF >= 0.5
            step = 0.002 if pf < 0.5 else 0.001
            if _can_adjust_param("take_profit_pct", "tighten"):
                new_tp = max(current_tp - step, TP_MIN)
                if new_tp != current_tp:
                    actions.append({
                        "param": "take_profit_pct",
                        "action": "tighten_tp",
                        "old": current_tp,
                        "new": new_tp,
                        "reason": f"TP hit rate muy bajo ({tp_hit_rate:.1f}%, {tp_hits}/{global_metrics['trades']}). Bajar para capturar profit (step={step}).",
                    })
        elif pf < 0.8 and avg_win > 0 and avg_win < avg_loss * 0.6:
            if _can_adjust_param("take_profit_pct", "tighten"):
                new_tp = max(current_tp - 0.001, TP_MIN)
                if new_tp != current_tp:
                    actions.append({
                        "param": "take_profit_pct",
                        "action": "tighten_tp",
                        "old": current_tp,
                        "new": new_tp,
                        "reason": f"PF bajo ({pf:.2f}), ganancias chicas. Tomar profit antes.",
                    })
        elif tp_hit_rate > 30 and pf > 1.5 and avg_win > 0 and avg_win > avg_loss * 1.5:
            # TP se alcanza frecuentemente y el PF es bueno -> podemos subir
            if _can_adjust_param("take_profit_pct", "widen"):
                new_tp = min(current_tp + 0.001, TP_MAX)
                if new_tp != current_tp:
                    actions.append({
                        "param": "take_profit_pct",
                        "action": "widen_tp",
                        "old": current_tp,
                        "new": new_tp,
                        "reason": f"TP hit rate alto ({tp_hit_rate:.1f}%) y PF {pf:.2f}. Dejar correr mas.",
                    })

    return actions


def _decide_rebalance(by_symbol: Dict, active_pairs: List[str], balance: float, current_alloc: Dict) -> Optional[Dict]:
    """Decide si rebalancear capital entre pares."""
    scored_pairs = {}
    for sym in active_pairs:
        data = by_symbol.get(sym)
        if data and data["trades"] >= 5:
            scored_pairs[sym] = data["score"]
        else:
            # Par nuevo sin data suficiente -> score neutro
            scored_pairs[sym] = 3.0

    if not scored_pairs:
        return None

    total_score = sum(scored_pairs.values())
    if total_score <= 0:
        return None

    # Calcular nueva asignacion proporcional al score
    new_alloc = {}
    min_per_pair = balance * MIN_ALLOC_PCT
    max_per_pair = balance * MAX_ALLOC_PCT

    for sym in active_pairs:
        raw = (scored_pairs.get(sym, 1) / total_score) * balance
        new_alloc[sym] = max(min_per_pair, min(max_per_pair, round(raw, 2)))

    # Normalizar para que sume exactamente el balance
    total_alloc = sum(new_alloc.values())
    if total_alloc > 0:
        factor = balance / total_alloc
        for sym in new_alloc:
            new_alloc[sym] = round(new_alloc[sym] * factor, 2)

    # Cap máximo de cambio por ciclo: no mover más de MAX_REBALANCE_SHIFT_PCT por par
    capped = False
    for sym in active_pairs:
        old = current_alloc.get(sym, 0)
        proposed = new_alloc.get(sym, 0)
        if old > 0:
            pct_change = (proposed - old) / old * 100
            if abs(pct_change) > MAX_REBALANCE_SHIFT_PCT:
                if pct_change > 0:
                    new_alloc[sym] = round(old * (1 + MAX_REBALANCE_SHIFT_PCT / 100), 2)
                else:
                    new_alloc[sym] = round(old * (1 - MAX_REBALANCE_SHIFT_PCT / 100), 2)
                capped = True

    # Re-normalizar después del cap
    if capped:
        total_alloc = sum(new_alloc.values())
        if total_alloc > 0:
            factor = balance / total_alloc
            for sym in new_alloc:
                new_alloc[sym] = round(new_alloc[sym] * factor, 2)
        # Forzar límites min/max
        for sym in new_alloc:
            new_alloc[sym] = max(balance * MIN_ALLOC_PCT, min(balance * MAX_ALLOC_PCT, new_alloc[sym]))

    # Solo aplicar si hay diferencia significativa (>5% de cambio en algun par)
    significant = False
    changes = {}
    for sym in active_pairs:
        old = current_alloc.get(sym, 0)
        new = new_alloc.get(sym, 0)
        if old > 0:
            pct_change = abs(new - old) / old * 100
            if pct_change > 5:
                significant = True
                changes[sym] = {"old": old, "new": new, "change_pct": round(pct_change, 1)}

    if not significant:
        return None

    return {
        "new_allocations": new_alloc,
        "changes": changes,
        "reason": "Rebalanceo basado en rendimiento: mas capital a pares rentables",
    }


# ─────────────────────────────────────────────
#  CICLO PRINCIPAL DE OPTIMIZACION
# ─────────────────────────────────────────────

def run_optimization_cycle() -> Dict[str, Any]:
    """Ejecuta un ciclo completo de optimizacion."""
    cfg = _read_config()
    if not cfg:
        return {"error": "No se pudo leer config", "actions": []}

    trades = _parse_trades()

    # Skip si no hay nuevos trades desde el ultimo ciclo
    import hashlib
    trades_hash = hashlib.md5(str(len(trades)).encode() + str(sum(t["pnl_f"] for t in trades)).encode()).hexdigest()
    with _lock:
        if trades_hash == _state["last_trades_hash"]:
            _log_action("skip_no_new_trades", {"trades": len(trades)})
            return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "actions": [], "skipped": True}
        _state["last_trades_hash"] = trades_hash
    _save_persistent_state()

    metrics = _compute_metrics(trades)
    gm = metrics["global"]
    by_sym = metrics["by_symbol"]
    by_reason = metrics["by_reason"]

    # Si no hay trades suficientes, no hacer nada
    if not gm or gm.get("trades", 0) == 0:
        _log_action("skip_no_trades", {"reason": "0 trades en el log"})
        return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "actions": [], "skipped": True}

    active_pairs = cfg.get("symbols", [])
    balance = cfg.get("wallet_balance", 100)
    current_tp = cfg.get("take_profit_pct", 0.006)
    current_sl = cfg.get("stop_loss_pct", 0.004)
    current_trailing = cfg.get("trailing_stop_pct", 0.004)
    allocations = cfg.get("pair_allocations", {})
    targets = cfg.get("pair_targets", {})

    actions = []
    config_changed = False

    # ── 1. Evaluar remocion de par toxico ──
    removal = _decide_pair_removal(by_sym, active_pairs)
    if removal:
        sym = removal["symbol"]
        active_pairs = [s for s in active_pairs if s != sym]
        allocations.pop(sym, None)
        targets.pop(sym, None)
        # Redistribuir al resto
        per_pair = round(balance / len(active_pairs), 2) if active_pairs else 0
        for s in active_pairs:
            allocations[s] = per_pair
            targets[s] = round(per_pair * 1.25, 2)
        cfg["symbols"] = active_pairs
        cfg["pair_allocations"] = allocations
        cfg["pair_targets"] = targets
        config_changed = True
        action = _log_action("remove_pair", {
            "symbol": sym,
            "severity": removal["severity"],
            "reasons": removal["reasons"],
            "pf": removal.get("pf", 0),
            "wr": removal.get("wr", 0),
            "pnl": removal.get("pnl", 0),
            "trades": removal.get("trades", 0),
            "remaining_pairs": active_pairs,
        })
        actions.append(action)

    # ── 2. Evaluar trailing stop ──
    trailing_adj = _decide_trailing_adjustment(by_reason, current_trailing, gm)
    if trailing_adj:
        cfg["trailing_stop_pct"] = trailing_adj["new_value"]
        config_changed = True
        direction = "widen" if trailing_adj["new_value"] > current_trailing else "tighten"
        _mark_param_adjusted("trailing_stop_pct", direction)
        action = _log_action("adjust_trailing", {
            "old_value": current_trailing,
            "new_value": trailing_adj["new_value"],
            "reason": trailing_adj["reason"],
        })
        actions.append(action)

    # ── 3. Evaluar TP/SL ──
    tp_sl_actions = _decide_tp_sl_adjustment(gm, current_tp, current_sl, by_reason=by_reason)
    for adj in tp_sl_actions:
        cfg[adj["param"]] = adj["new"]
        config_changed = True
        direction = adj["action"].split("_")[0]  # tighten or widen
        _mark_param_adjusted(adj["param"], direction)
        action = _log_action("adjust_param", {
            "param": adj["param"],
            "old_value": adj["old"],
            "new_value": adj["new"],
            "reason": adj["reason"],
        })
        actions.append(action)

    # ── 3b. Guarda de coherencia: TP >= SL * MIN_RR_RATIO ──
    final_tp = float(cfg.get("take_profit_pct", TP_MIN))
    final_sl = float(cfg.get("stop_loss_pct", SL_MIN))
    if final_tp < final_sl * MIN_RR_RATIO:
        # TP demasiado bajo respecto al SL — reducir SL para mantener R:R
        corrected_sl = round(final_tp / MIN_RR_RATIO, 4)
        corrected_sl = min(max(corrected_sl, SL_MIN), SL_MAX)
        if corrected_sl != final_sl:
            cfg["stop_loss_pct"] = corrected_sl
            config_changed = True
            action = _log_action("coherence_guard", {
                "issue": f"TP ({final_tp*100:.2f}%) < SL ({final_sl*100:.2f}%) * {MIN_RR_RATIO}. R:R desfavorable.",
                "corrected": f"SL ajustado de {final_sl*100:.2f}% a {corrected_sl*100:.2f}% para mantener R:R minimo.",
            })
            actions.append(action)

    # ── 3c. Auto-ajuste de exit aggressiveness ──
    # Si la mayoria de cierres son por senal tecnica con perdida, las salidas son demasiado agresivas
    signal_exits = by_reason.get("senal EMA/RSI/MACD", {})
    signal_count = signal_exits.get("count", 0)
    signal_pnl = signal_exits.get("pnl", 0)
    if signal_count >= 10 and gm["trades"] >= MIN_TRADES_GLOBAL:
        signal_pct = signal_count / gm["trades"] * 100
        signal_avg_pnl = signal_pnl / signal_count if signal_count > 0 else 0
        # Si >80% de salidas son por senal y el PnL promedio es negativo -> salidas muy agresivas
        if signal_pct > 80 and signal_avg_pnl < -0.05:
            current_sell_conf = cfg.get("sell_signal_confirmations", 2)
            current_min_hold = cfg.get("min_hold_seconds", 20)
            adj_made = False
            if current_sell_conf < 4 and _can_adjust_param("sell_signal_confirmations", "raise"):
                cfg["sell_signal_confirmations"] = current_sell_conf + 1
                _mark_param_adjusted("sell_signal_confirmations", "raise")
                config_changed = True
                adj_made = True
            if current_min_hold < 300 and _can_adjust_param("min_hold_seconds", "raise"):
                new_hold = min(current_min_hold + 60, 300)
                cfg["min_hold_seconds"] = new_hold
                _mark_param_adjusted("min_hold_seconds", "raise")
                config_changed = True
                adj_made = True
            if adj_made:
                action = _log_action("adjust_exit_aggressiveness", {
                    "signal_exit_pct": round(signal_pct, 1),
                    "signal_avg_pnl": round(signal_avg_pnl, 3),
                    "new_sell_conf": cfg.get("sell_signal_confirmations"),
                    "new_min_hold": cfg.get("min_hold_seconds"),
                    "reason": f"Salidas por senal {signal_pct:.0f}% con PnL promedio ${signal_avg_pnl:.3f}. Reducir agresividad.",
                })
                actions.append(action)

    # ── 3d. Auto-deteccion de trailing catastrofico ──
    # Si hay trades por trailing stop con perdida promedio > 2x la perdida promedio por senal, el trailing es peligroso
    trailing_exits = by_reason.get("trailing stop", {})
    trailing_count = trailing_exits.get("count", 0)
    trailing_pnl = trailing_exits.get("pnl", 0)
    if trailing_count >= 1:
        trailing_avg = trailing_pnl / trailing_count
        signal_avg = signal_pnl / signal_count if signal_count > 0 else -0.05
        # Trailing catastrofico: perdida promedio trailing > 2x the average loss OR > $2 de perdida promedio
        is_catastrophic = (trailing_avg < signal_avg * 2 and trailing_avg < -1.0) or trailing_avg < -2.0
        if is_catastrophic:
            current_trailing = cfg.get("trailing_stop_pct", 0.005)
            current_max_trail_loss = cfg.get("max_trailing_loss_pct", 0.003)
            adj_trail = False
            # 1) Reducir trailing_stop_pct para que el stop siga más cerca
            if current_trailing > TRAILING_MIN and _can_adjust_param("trailing_stop_pct", "tighten"):
                new_trailing = max(round(current_trailing - 0.001, 4), TRAILING_MIN)
                cfg["trailing_stop_pct"] = new_trailing
                _mark_param_adjusted("trailing_stop_pct", "tighten")
                config_changed = True
                adj_trail = True
            # 2) Reducir max_trailing_loss_pct (hard cap) si la pérdida fue extrema
            if trailing_avg < -3.0 and current_max_trail_loss > 0.002:
                new_cap = max(round(current_max_trail_loss - 0.001, 4), 0.002)
                cfg["max_trailing_loss_pct"] = new_cap
                config_changed = True
                adj_trail = True
            if adj_trail:
                action = _log_action("trailing_catastrophic_fix", {
                    "trailing_avg_pnl": round(trailing_avg, 2),
                    "trailing_count": trailing_count,
                    "new_trailing_pct": cfg.get("trailing_stop_pct"),
                    "new_max_trail_loss": cfg.get("max_trailing_loss_pct"),
                    "reason": f"Trailing catastrofico detectado: avg ${trailing_avg:.2f}/trade. Ajustando trailing y cap de perdida.",
                })
                actions.append(action)

    # ── 4. Rebalancear capital ──
    rebalance = _decide_rebalance(by_sym, active_pairs, balance, allocations)
    if rebalance:
        cfg["pair_allocations"] = rebalance["new_allocations"]
        # Actualizar targets proporcionales
        for sym, alloc in rebalance["new_allocations"].items():
            cfg.setdefault("pair_targets", {})[sym] = round(alloc * 1.25, 2)
        config_changed = True
        action = _log_action("rebalance", {
            "changes": rebalance["changes"],
            "reason": rebalance["reason"],
        })
        actions.append(action)

    # ── 4b. Auto-correccion de rebalanceo extremo ──
    # Si algun par tiene menos del 20% de lo que le corresponderia equitativamente, restaurar gradualmente
    if len(active_pairs) >= 2:
        equal_share = balance / len(active_pairs)
        current_allocs = cfg.get("pair_allocations", {})
        starved_pairs = []
        for sym in active_pairs:
            alloc = current_allocs.get(sym, equal_share)
            if alloc < equal_share * 0.20:
                starved_pairs.append(sym)
        if starved_pairs:
            # Restaurar gradualmente hacia equitativo (mover 15% del delta por ciclo)
            restore_speed = 0.15
            new_allocs = dict(current_allocs)
            for sym in starved_pairs:
                delta = equal_share - new_allocs.get(sym, 0)
                new_allocs[sym] = round(new_allocs.get(sym, 0) + delta * restore_speed, 2)
            # Quitar proporcional de los que tienen de mas
            surplus_pairs = [s for s in active_pairs if s not in starved_pairs]
            surplus_total = sum(new_allocs.get(s, 0) for s in surplus_pairs)
            needed = balance - sum(new_allocs.get(s, 0) for s in starved_pairs)
            if surplus_total > 0 and surplus_pairs:
                for s in surplus_pairs:
                    new_allocs[s] = round(new_allocs.get(s, 0) * (needed / surplus_total), 2)
            cfg["pair_allocations"] = new_allocs
            for sym, alloc in new_allocs.items():
                cfg.setdefault("pair_targets", {})[sym] = round(alloc * 1.25, 2)
            config_changed = True
            action = _log_action("rebalance_recovery", {
                "starved_pairs": starved_pairs,
                "new_allocations": new_allocs,
                "reason": f"Pares con capital critico (<20% equitativo) detectados: {starved_pairs}. Restaurando gradualmente.",
            })
            actions.append(action)

    # ── 5. Integrar sugerencias de exit_learning (Inteli Genie) ──
    try:
        import intelligence_engine as _ie
        intel_state = _ie.get_intelligence_state()
        exit_analysis = intel_state.get("exit_analysis", {})
        for sym, edata in exit_analysis.items():
            for sug in edata.get("suggestions", []):
                saction = sug.get("action", "")
                confidence = sug.get("confidence", 0)
                if confidence < 0.5:
                    continue
                if saction == "tighten_tp" and cfg["take_profit_pct"] > TP_MIN:
                    if _can_adjust_param("take_profit_pct", "tighten"):
                        old_tp = cfg["take_profit_pct"]
                        new_tp = max(old_tp - 0.001, TP_MIN)
                        if new_tp != old_tp:
                            cfg["take_profit_pct"] = new_tp
                            config_changed = True
                            _mark_param_adjusted("take_profit_pct", "tighten")
                            action = _log_action("exit_learning_tp", {
                                "symbol": sym, "old": old_tp, "new": new_tp,
                                "reason": f"Exit learning ({sym}): {sug['reason']}",
                            })
                            actions.append(action)
                elif saction == "widen_tp" and cfg["take_profit_pct"] < TP_MAX:
                    if _can_adjust_param("take_profit_pct", "widen"):
                        old_tp = cfg["take_profit_pct"]
                        new_tp = min(old_tp + 0.001, TP_MAX)
                        if new_tp != old_tp:
                            cfg["take_profit_pct"] = new_tp
                            config_changed = True
                            _mark_param_adjusted("take_profit_pct", "widen")
                            action = _log_action("exit_learning_tp", {
                                "symbol": sym, "old": old_tp, "new": new_tp,
                                "reason": f"Exit learning ({sym}): {sug['reason']}",
                            })
                            actions.append(action)
                elif saction == "widen_trailing" and cfg["trailing_stop_pct"] < TRAILING_MAX:
                    if _can_adjust_param("trailing_stop_pct", "widen"):
                        old_trail = cfg["trailing_stop_pct"]
                        new_trail = min(old_trail + 0.001, TRAILING_MAX)
                        if new_trail != old_trail:
                            cfg["trailing_stop_pct"] = new_trail
                            config_changed = True
                            _mark_param_adjusted("trailing_stop_pct", "widen")
                            action = _log_action("exit_learning_trailing", {
                                "symbol": sym, "old": old_trail, "new": new_trail,
                                "reason": f"Exit learning ({sym}): {sug['reason']}",
                            })
                            actions.append(action)
    except Exception:
        pass  # Intelligence engine no disponible, seguir sin

    # ── Guardar config si hubo cambios ──
    if config_changed:
        _save_config(cfg)
        _log_action("config_saved", {"changed_keys": [a.get("type", "") for a in actions]})

    # ── Si no hubo cambios, loguear que todo esta ok ──
    if not actions:
        _log_action("no_changes", {
            "trades": gm.get("trades", 0),
            "pf": round(gm.get("pf", 0), 2),
            "wr": round(gm.get("wr", 0), 1),
            "pnl": round(gm.get("pnl", 0), 2),
        })

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "actions": actions,
        "metrics": {
            "trades": gm.get("trades", 0),
            "pf": round(gm.get("pf", 0), 2),
            "wr": round(gm.get("wr", 0), 1),
            "pnl": round(gm.get("pnl", 0), 2),
            "pairs_active": len(active_pairs),
        },
        "config_changed": config_changed,
    }

    # Actualizar state
    with _lock:
        _state["last_run"] = result["timestamp"]
        _state["last_actions"] = actions
        _state["cycle_count"] += 1
        _state["history"].append(result)
        if len(_state["history"]) > 50:
            _state["history"] = _state["history"][-50:]

    return result


def _auto_retrain_model():
    """Re-entrena el modelo AI si hay suficiente dataset y el modelo es viejo."""
    try:
        import train_model
    except ImportError:
        return
    ai_model_file = "ai_model.json"
    dataset_file = "ml_dataset.csv"
    if not os.path.exists(dataset_file):
        return
    # Solo entrenar si no hay modelo o tiene mas de 2 horas
    if os.path.exists(ai_model_file):
        age_hours = (time.time() - os.path.getmtime(ai_model_file)) / 3600
        if age_hours < 2:
            return
    try:
        result = train_model.train_and_save()
        if result.get("success"):
            test_acc = result.get("test_metrics", {}).get("accuracy")
            test_f1 = result.get("test_metrics", {}).get("f1")
            _log_action("auto_retrain", {
                "accuracy": test_acc,
                "f1": test_f1,
                "samples": result.get("labeled_samples"),
                "model_type": result.get("model_type"),
            })
            # Habilitar AI si estaba apagada (no cambiar mode si ya esta configurado)
            cfg = _read_config()
            if not cfg.get("ai_enabled", False):
                cfg["ai_enabled"] = True
                if cfg.get("ai_mode", "off") == "off":
                    cfg["ai_mode"] = "filter"
                _save_config(cfg)
    except Exception as e:
        _log_action("auto_retrain_error", {"error": str(e)[:100]})


# ─────────────────────────────────────────────
#  BACKGROUND THREAD
# ─────────────────────────────────────────────

def _optimizer_loop():
    """Loop principal que corre cada N minutos."""
    _stop_event.wait(timeout=60)  # Esperar 1 min al arrancar
    retrain_counter = 0
    while not _stop_event.is_set():
        with _lock:
            enabled = _state["enabled"]
            interval = _state["interval_minutes"]

        if enabled:
            try:
                with _lock:
                    _state["running"] = True
                    _state["error"] = None
                run_optimization_cycle()
                # Auto-retrain AI cada 6 ciclos (~1 hora con interval=10)
                retrain_counter += 1
                if retrain_counter >= 6:
                    retrain_counter = 0
                    _auto_retrain_model()
            except Exception as e:
                with _lock:
                    _state["error"] = str(e)
                _log_action("error", {"error": str(e)})
            finally:
                with _lock:
                    _state["running"] = False

        _stop_event.wait(timeout=interval * 60)


def start_optimizer():
    """Inicia el optimizer en background."""
    global _thread
    if _thread and _thread.is_alive():
        return

    # Restaurar estado persistente (hash, cooldowns)
    _load_persistent_state()

    # Leer config para enabled e interval
    cfg = _read_config()
    with _lock:
        _state["enabled"] = bool(cfg.get("smart_optimizer_enabled", False))
        _state["interval_minutes"] = max(1, int(cfg.get("smart_optimizer_interval_minutes", 10)))

    _stop_event.clear()
    _thread = threading.Thread(target=_optimizer_loop, daemon=True, name="AutoOptimizer")
    _thread.start()


def stop_optimizer():
    """Detiene el optimizer."""
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)


def get_optimizer_state() -> Dict[str, Any]:
    """Retorna estado actual (thread-safe)."""
    cfg = _read_config()
    with _lock:
        state = dict(_state)
        # Sincronizar con config
        state["enabled"] = bool(cfg.get("smart_optimizer_enabled", False))
        state["interval_minutes"] = max(1, int(cfg.get("smart_optimizer_interval_minutes", 10)))
        _state["enabled"] = state["enabled"]
        _state["interval_minutes"] = state["interval_minutes"]
    return state


def update_optimizer_config(enabled: bool, interval_minutes: int):
    """Actualiza enabled/interval desde el dashboard."""
    cfg = _read_config()
    cfg["smart_optimizer_enabled"] = enabled
    cfg["smart_optimizer_interval_minutes"] = max(1, min(240, interval_minutes))
    _save_config(cfg)
    with _lock:
        _state["enabled"] = enabled
        _state["interval_minutes"] = cfg["smart_optimizer_interval_minutes"]


def force_optimize() -> Dict[str, Any]:
    """Ejecuta un ciclo de optimizacion inmediato."""
    return run_optimization_cycle()


# ─────────────────────────────────────────────
#  GRID SEARCH VIA BACKTESTING
# ─────────────────────────────────────────────

# Estado global del grid search (thread-safe)
_gs_lock = threading.Lock()
_gs_state: Dict[str, Any] = {
    "running": False,
    "progress": 0,          # 0-100
    "total_combos": 0,
    "evaluated": 0,
    "best": None,            # Mejores params encontrados
    "results": [],           # Top 10 resultados
    "last_run": None,
    "error": None,
}


def get_grid_search_state() -> Dict[str, Any]:
    """Retorna estado actual del grid search."""
    with _gs_lock:
        return dict(_gs_state)


def run_grid_search(
    symbol: str = "BTCUSDT",
    limit: int = 1000,
    interval: str = "1m",
    balance: float = 100.0,
    tp_range: Optional[Tuple[float, float, float]] = None,
    sl_range: Optional[Tuple[float, float, float]] = None,
    trailing_range: Optional[Tuple[float, float, float]] = None,
    fee_pct: float = 0.001,
    metric: str = "profit_factor",
) -> Dict[str, Any]:
    """
    Ejecuta grid search sobre parametros de estrategia usando backtest real.

    Ranges: (min, max, step) — si None, usa defaults.
    metric: metrica a optimizar (profit_factor, sharpe_ratio, total_pnl_pct, expectancy).
    """
    try:
        import backtest as backtest_mod
    except ImportError:
        return {"error": "backtest module not available"}

    # Defaults razonables
    if tp_range is None:
        tp_range = (0.004, 0.014, 0.002)
    if sl_range is None:
        sl_range = (0.003, 0.008, 0.001)
    if trailing_range is None:
        trailing_range = (0.003, 0.008, 0.001)

    # Generar grid
    def _frange(start: float, stop: float, step: float) -> List[float]:
        vals = []
        v = start
        while v <= stop + 1e-9:
            vals.append(round(v, 6))
            v += step
        return vals

    tp_values = _frange(*tp_range)
    sl_values = _frange(*sl_range)
    trail_values = _frange(*trailing_range)

    combos = [
        (tp, sl, tr)
        for tp in tp_values
        for sl in sl_values
        for tr in trail_values
    ]
    total = len(combos)

    with _gs_lock:
        if _gs_state["running"]:
            return {"error": "Grid search ya en ejecucion"}
        _gs_state["running"] = True
        _gs_state["progress"] = 0
        _gs_state["total_combos"] = total
        _gs_state["evaluated"] = 0
        _gs_state["best"] = None
        _gs_state["results"] = []
        _gs_state["error"] = None

    valid_metrics = ("profit_factor", "sharpe_ratio", "total_pnl_pct", "expectancy")
    if metric not in valid_metrics:
        metric = "profit_factor"

    try:
        # Descargar velas una sola vez
        raw_klines = backtest_mod.fetch_klines(symbol, interval, limit)
        candles = backtest_mod.parse_klines(raw_klines)

        all_results = []
        for i, (tp, sl, tr) in enumerate(combos):
            try:
                r = backtest_mod.run_backtest(
                    candles, symbol=symbol,
                    initial_balance=balance,
                    take_profit_pct=tp,
                    stop_loss_pct=sl,
                    trailing_stop_pct=tr,
                    fee_pct=fee_pct,
                )
                score = getattr(r, metric, 0.0)
                entry = {
                    "take_profit_pct": tp,
                    "stop_loss_pct": sl,
                    "trailing_stop_pct": tr,
                    "trades": r.total_trades,
                    "win_rate": round(r.win_rate, 2),
                    "profit_factor": round(r.profit_factor, 3),
                    "sharpe_ratio": round(r.sharpe_ratio, 3),
                    "total_pnl": round(r.total_pnl, 4),
                    "total_pnl_pct": round(r.total_pnl_pct, 2),
                    "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                    "expectancy": round(r.expectancy, 4),
                    "score": round(score, 4),
                }
                all_results.append(entry)
            except Exception:
                pass  # Skip combos que fallan

            with _gs_lock:
                _gs_state["evaluated"] = i + 1
                _gs_state["progress"] = round((i + 1) / total * 100, 1)

        # Ordenar por metrica elegida (descendente)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_10 = all_results[:10]
        best = top_10[0] if top_10 else None

        with _gs_lock:
            _gs_state["running"] = False
            _gs_state["progress"] = 100
            _gs_state["best"] = best
            _gs_state["results"] = top_10
            _gs_state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Loguear resultado
        if best:
            _log_action("grid_search_done", {
                "symbol": symbol,
                "combos": total,
                "metric": metric,
                "best_score": best["score"],
                "best_params": {
                    "tp": best["take_profit_pct"],
                    "sl": best["stop_loss_pct"],
                    "trailing": best["trailing_stop_pct"],
                },
            })

        return {
            "symbol": symbol,
            "candles": len(candles),
            "combos_evaluated": len(all_results),
            "metric": metric,
            "best": best,
            "top_10": top_10,
        }

    except Exception as e:
        with _gs_lock:
            _gs_state["running"] = False
            _gs_state["error"] = str(e)
        return {"error": str(e)}


def apply_grid_search_best() -> Dict[str, Any]:
    """Aplica los mejores parametros del grid search a runtime_config."""
    with _gs_lock:
        best = _gs_state.get("best")
    if not best:
        return {"error": "No hay resultados de grid search"}

    cfg = _read_config()
    old = {
        "take_profit_pct": cfg.get("take_profit_pct"),
        "stop_loss_pct": cfg.get("stop_loss_pct"),
        "trailing_stop_pct": cfg.get("trailing_stop_pct"),
    }
    cfg["take_profit_pct"] = best["take_profit_pct"]
    cfg["stop_loss_pct"] = best["stop_loss_pct"]
    cfg["trailing_stop_pct"] = best["trailing_stop_pct"]
    _save_config(cfg)

    _log_action("grid_search_applied", {
        "old": old,
        "new": {
            "take_profit_pct": best["take_profit_pct"],
            "stop_loss_pct": best["stop_loss_pct"],
            "trailing_stop_pct": best["trailing_stop_pct"],
        },
    })

    return {"ok": True, "applied": best, "old": old}


# ─────────────────────────────────────────────
#  REGIME-BASED PARAMETER PROFILES
# ─────────────────────────────────────────────

REGIME_PARAMS_FILE = "regime_params.json"

# Defaults conservadores por regimen (se afinan via grid search)
_DEFAULT_REGIME_PARAMS = {
    "trending_up": {
        "take_profit_pct": 0.010,
        "stop_loss_pct": 0.005,
        "trailing_stop_pct": 0.005,
        "trade_pct_mult": 1.1,
    },
    "trending_down": {
        "take_profit_pct": 0.006,
        "stop_loss_pct": 0.004,
        "trailing_stop_pct": 0.003,
        "trade_pct_mult": 0.6,
    },
    "ranging": {
        "take_profit_pct": 0.006,
        "stop_loss_pct": 0.004,
        "trailing_stop_pct": 0.004,
        "trade_pct_mult": 0.7,
    },
    "volatile": {
        "take_profit_pct": 0.012,
        "stop_loss_pct": 0.006,
        "trailing_stop_pct": 0.006,
        "trade_pct_mult": 0.5,
    },
}


def load_regime_params() -> Dict[str, Any]:
    """Carga perfiles de params por regimen (o devuelve defaults)."""
    if os.path.exists(REGIME_PARAMS_FILE):
        try:
            with open(REGIME_PARAMS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return dict(_DEFAULT_REGIME_PARAMS)


def save_regime_params(params: Dict[str, Any]):
    """Guarda perfiles de params por regimen."""
    with open(REGIME_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


def get_params_for_regime(regime: str) -> Dict[str, Any]:
    """Retorna params optimizados para el regimen actual."""
    profiles = load_regime_params()
    return profiles.get(regime, profiles.get("ranging", _DEFAULT_REGIME_PARAMS["ranging"]))


def update_regime_from_grid_search(regime: str) -> Dict[str, Any]:
    """Actualiza el perfil de un regimen con los mejores params del ultimo grid search."""
    with _gs_lock:
        best = _gs_state.get("best")
    if not best:
        return {"error": "No hay resultados de grid search"}

    profiles = load_regime_params()
    old = profiles.get(regime, {})
    profiles[regime] = {
        "take_profit_pct": best["take_profit_pct"],
        "stop_loss_pct": best["stop_loss_pct"],
        "trailing_stop_pct": best["trailing_stop_pct"],
        "trade_pct_mult": old.get("trade_pct_mult", 1.0),
    }
    save_regime_params(profiles)

    _log_action("regime_profile_updated", {
        "regime": regime,
        "old": old,
        "new": profiles[regime],
    })

    return {"ok": True, "regime": regime, "params": profiles[regime]}
