"""
Configuracion compartida entre scalping_bot.py y dashboard.py.

Fuente unica de verdad para:
- DEFAULT_RUNTIME_CONFIG: valores por defecto de todos los parametros del bot.
- normalize_runtime_config(): validacion y clamping de valores.
"""

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

DEFAULT_RUNTIME_CONFIG = {
    "symbols": DEFAULT_SYMBOLS.copy(),
    "wallet_balance": 100.0,
    "pair_allocations": {},
    "pair_targets": {},
    "trade_pct": 0.95,
    "take_profit_pct": 0.008,
    "trailing_stop_pct": 0.004,
    "break_even_pct": 0.005,
    "max_initial_loss_pct": 0.003,
    "max_trailing_loss_pct": 0.003,  # Hard cap: max perdida por trailing (0.3%)
    "target_usdt": 120.0,
    "update_mode": "candle",
    "poll_seconds": 60,
    "signal_confirmations": 2,
    "sell_signal_confirmations": 2,
    "min_hold_seconds": 20,
    "max_api_latency_ms": 1200,
    "cooldown_seconds": 8,
    "daily_loss_limit_usdt": 3.0,
    "max_drawdown_pct": 0.05,
    "max_trades_per_hour": 40,
    "optimizer_mode": "auto",
    "optimizer_window_hours": 24,
    "optimizer_pf_keep": 1.10,
    "optimizer_pf_backoff": 1.00,
    "optimizer_auto_enabled": False,
    "optimizer_auto_minutes": 10,
    "smart_optimizer_enabled": False,
    "smart_optimizer_interval_minutes": 10,
    "ai_enabled": False,
    "ai_mode": "filter",
    "ai_min_confidence": 0.60,
    "ai_adaptive_sizing": False,
    "ai_high_confidence": 0.75,
    "ai_low_confidence": 0.62,
    "ai_high_trade_pct_mult": 1.10,
    "ai_low_trade_pct_mult": 0.70,
    "ai_high_tp_mult": 1.30,
    "ai_low_tp_mult": 0.80,
    # Intelligence Engine
    "intel_enabled": True,
    "intel_hours_enabled": True,
    "intel_hours_min_trades": 10,
    "intel_hours_min_pf": 0.5,
    "intel_regime_enabled": True,
    "intel_regime_block_ranging": True,
    "intel_regime_reduce_volatile": True,
    "intel_regime_volatile_reduce_pct": 0.5,
    "intel_volume_enabled": True,
    "intel_volume_spike_mult": 3.0,
    "intel_volume_drought_mult": 0.3,
    "intel_score_enabled": True,
    "intel_score_min_buy": 40,
    "intel_correlation_enabled": True,
    "intel_correlation_max_entries": 3,
    "intel_adaptive_trailing_enabled": True,
    "intel_trailing_atr_mult": 1.5,
    "intel_exit_learning_enabled": True,
    # Multi-timeframe
    "mtf_enabled": False,
    "mtf_timeframe": "5m",
    "mtf_candles": 30,
    # Comisiones
    "fee_pct": 0.001,  # 0.1% taker fee (Binance default)
    # Modo de mercado
    "market_mode": "spot",          # "spot" o "futures"
    "futures_leverage": 3,          # Apalancamiento en futuros (1-20)
    "futures_fee_pct": 0.0004,      # 0.04% taker fee (Binance Futures default)
    "spot_fee_pct": 0.001,          # 0.1% taker fee (Binance Spot default)
    # Trades diarios
    "max_trades_per_day": 0,        # 0 = sin limite
}

DEFAULT_MULTI_SOURCE_CONFIG = {
    "enabled": True,
    "threshold_pct": 0.15,
    "action": "alert",
    "refresh_seconds": 5,
}


def normalize_runtime_config(payload: dict) -> dict:
    cfg = DEFAULT_RUNTIME_CONFIG.copy()
    cfg.update(payload or {})

    raw_symbols = cfg.get("symbols", DEFAULT_SYMBOLS)
    parsed_symbols = []
    if isinstance(raw_symbols, str):
        chunks = [s.strip().upper() for s in raw_symbols.split(",")]
        parsed_symbols = [s for s in chunks if s.endswith("USDT") and len(s) >= 6]
    elif isinstance(raw_symbols, list):
        chunks = [str(s).strip().upper() for s in raw_symbols]
        parsed_symbols = [s for s in chunks if s.endswith("USDT") and len(s) >= 6]
    dedup = []
    seen = set()
    for sym in parsed_symbols:
        if sym in seen:
            continue
        seen.add(sym)
        dedup.append(sym)
    if not dedup:
        dedup = DEFAULT_SYMBOLS.copy()
    cfg["symbols"] = dedup[:12]

    cfg["wallet_balance"] = min(max(float(cfg.get("wallet_balance", 100.0)), 1.0), 1_000_000.0)
    raw_alloc = cfg.get("pair_allocations") or {}
    if not isinstance(raw_alloc, dict):
        raw_alloc = {}
    clean_alloc = {}
    for sym in cfg["symbols"]:
        val = raw_alloc.get(sym)
        if val is not None:
            clean_alloc[sym] = min(max(float(val), 0.0), cfg["wallet_balance"])
    cfg["pair_allocations"] = clean_alloc
    raw_targets = cfg.get("pair_targets") or {}
    if not isinstance(raw_targets, dict):
        raw_targets = {}
    clean_targets = {}
    for sym in cfg["symbols"]:
        val = raw_targets.get(sym)
        if val is not None:
            clean_targets[sym] = max(float(val), 0.0)
    cfg["pair_targets"] = clean_targets

    cfg["trade_pct"] = min(max(float(cfg["trade_pct"]), 0.01), 1.0)
    cfg["take_profit_pct"] = min(max(float(cfg["take_profit_pct"]), 0.0001), 0.5)
    cfg["trailing_stop_pct"] = min(max(float(cfg["trailing_stop_pct"]), 0.0001), 0.5)
    cfg["target_usdt"] = max(float(cfg["target_usdt"]), 1.0)
    cfg["update_mode"] = str(cfg.get("update_mode", "candle")).strip().lower()
    if cfg["update_mode"] not in {"candle", "tick"}:
        cfg["update_mode"] = "candle"
    cfg["poll_seconds"] = min(max(int(float(cfg.get("poll_seconds", 60))), 1), 60)
    cfg["signal_confirmations"] = min(max(int(float(cfg.get("signal_confirmations", 2))), 1), 5)
    cfg["sell_signal_confirmations"] = min(max(int(float(cfg.get("sell_signal_confirmations", 2))), 1), 5)
    cfg["min_hold_seconds"] = min(max(int(float(cfg.get("min_hold_seconds", 20))), 0), 600)
    cfg["max_api_latency_ms"] = min(max(int(float(cfg.get("max_api_latency_ms", 1200))), 200), 10000)
    cfg["cooldown_seconds"] = min(max(int(float(cfg.get("cooldown_seconds", 8))), 0), 300)
    cfg["daily_loss_limit_usdt"] = min(max(float(cfg.get("daily_loss_limit_usdt", 3.0)), 0.1), 1000.0)
    cfg["max_drawdown_pct"] = min(max(float(cfg.get("max_drawdown_pct", 0.05)), 0.01), 0.50)
    cfg["max_trades_per_hour"] = min(max(int(float(cfg.get("max_trades_per_hour", 40))), 1), 300)
    mode = str(cfg.get("optimizer_mode", "auto")).strip().lower()
    cfg["optimizer_mode"] = mode if mode in {"frecuencia", "balanceado", "auto"} else "auto"
    cfg["optimizer_window_hours"] = min(max(int(float(cfg.get("optimizer_window_hours", 24))), 1), 168)
    cfg["optimizer_pf_keep"] = min(max(float(cfg.get("optimizer_pf_keep", 1.10)), 0.50), 3.00)
    cfg["optimizer_pf_backoff"] = min(max(float(cfg.get("optimizer_pf_backoff", 1.00)), 0.50), 3.00)
    cfg["optimizer_auto_enabled"] = bool(cfg.get("optimizer_auto_enabled", False))
    cfg["optimizer_auto_minutes"] = min(max(int(float(cfg.get("optimizer_auto_minutes", 10))), 1), 240)
    cfg["ai_enabled"] = bool(cfg.get("ai_enabled", False))
    ai_mode = str(cfg.get("ai_mode", "filter")).strip().lower()
    cfg["ai_mode"] = ai_mode if ai_mode in {"off", "filter", "advisor"} else "filter"
    cfg["ai_min_confidence"] = min(max(float(cfg.get("ai_min_confidence", 0.60)), 0.50), 0.99)
    cfg["ai_adaptive_sizing"] = bool(cfg.get("ai_adaptive_sizing", False))
    cfg["ai_high_confidence"] = min(max(float(cfg.get("ai_high_confidence", 0.75)), 0.50), 0.99)
    cfg["ai_low_confidence"] = min(max(float(cfg.get("ai_low_confidence", 0.62)), 0.50), 0.99)
    cfg["ai_high_trade_pct_mult"] = min(max(float(cfg.get("ai_high_trade_pct_mult", 1.10)), 0.50), 1.50)
    cfg["ai_low_trade_pct_mult"] = min(max(float(cfg.get("ai_low_trade_pct_mult", 0.70)), 0.30), 1.00)
    cfg["ai_high_tp_mult"] = min(max(float(cfg.get("ai_high_tp_mult", 1.30)), 1.00), 2.00)
    cfg["ai_low_tp_mult"] = min(max(float(cfg.get("ai_low_tp_mult", 0.80)), 0.50), 1.00)
    cfg["fee_pct"] = min(max(float(cfg.get("fee_pct", 0.001)), 0.0), 0.01)
    # Modo de mercado
    mm = str(cfg.get("market_mode", "spot")).strip().lower()
    cfg["market_mode"] = mm if mm in {"spot", "futures"} else "spot"
    cfg["futures_leverage"] = min(max(int(float(cfg.get("futures_leverage", 3))), 1), 20)
    cfg["futures_fee_pct"] = min(max(float(cfg.get("futures_fee_pct", 0.0004)), 0.0), 0.01)
    cfg["spot_fee_pct"] = min(max(float(cfg.get("spot_fee_pct", 0.001)), 0.0), 0.01)
    # Auto-sync fee_pct segun modo
    if cfg["market_mode"] == "futures":
        cfg["fee_pct"] = cfg["futures_fee_pct"]
    else:
        cfg["fee_pct"] = cfg["spot_fee_pct"]
    # Trades diarios
    cfg["max_trades_per_day"] = min(max(int(float(cfg.get("max_trades_per_day", 0))), 0), 1000)
    cfg["mtf_enabled"] = bool(cfg.get("mtf_enabled", False))
    mtf_tf = str(cfg.get("mtf_timeframe", "5m")).strip().lower()
    cfg["mtf_timeframe"] = mtf_tf if mtf_tf in {"3m", "5m", "15m"} else "5m"
    cfg["mtf_candles"] = min(max(int(float(cfg.get("mtf_candles", 30))), 20), 100)
    return cfg
