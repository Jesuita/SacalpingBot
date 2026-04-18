import json
import os
import sys
from typing import List

import requests

ENV_FILE = ".env"
RUNTIME_CONFIG_FILE = "runtime_config.json"
AI_MODEL_FILE = "ai_model.json"


def read_env(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_symbols(symbols: List[str]) -> List[str]:
    errors = []
    if not symbols:
        errors.append("No hay simbolos configurados")
        return errors
    for sym in symbols:
        if not isinstance(sym, str) or not sym.endswith("USDT") or len(sym) < 6:
            errors.append(f"Simbolo invalido: {sym}")
    return errors


def ping_binance() -> str:
    try:
        r = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        if r.status_code != 200:
            return f"Binance ping HTTP {r.status_code}"
        return ""
    except Exception as exc:
        return f"Error de red contra Binance: {exc}"


def validate_symbol_market(symbol: str) -> str:
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=5)
        if r.status_code != 200:
            return f"{symbol}: ticker HTTP {r.status_code}"
        price = float(r.json().get("price", 0))
        if price <= 0:
            return f"{symbol}: precio invalido"
        return ""
    except Exception as exc:
        return f"{symbol}: error API {exc}"


def main() -> int:
    env = read_env(ENV_FILE)
    cfg = read_config(RUNTIME_CONFIG_FILE)

    mode = (env.get("TRADING_MODE") or "paper").strip().lower()
    symbols = cfg.get("symbols") or ["BTCUSDT", "ETHUSDT"]
    ai_enabled = bool(cfg.get("ai_enabled", False))

    errors = []
    warnings = []

    if mode == "real":
        api_key = (env.get("BINANCE_API_KEY") or "").strip()
        api_secret = (env.get("BINANCE_API_SECRET") or "").strip()
        if not api_key or api_key.lower() == "tu_api_key_aqui":
            errors.append("BINANCE_API_KEY ausente o placeholder")
        if not api_secret or api_secret.lower() == "tu_api_secret_aqui":
            errors.append("BINANCE_API_SECRET ausente o placeholder")

    errors.extend(validate_symbols(symbols))

    ping_error = ping_binance()
    if ping_error:
        errors.append(ping_error)

    for sym in symbols:
        e = validate_symbol_market(sym)
        if e:
            errors.append(e)

    if ai_enabled and not os.path.exists(AI_MODEL_FILE):
        warnings.append("IA habilitada pero no existe ai_model.json (fallback tecnico)")

    print("=== PREFLIGHT ===")
    print(f"mode={mode}")
    print(f"symbols={','.join(symbols)}")
    print(f"ai_enabled={ai_enabled}")

    if warnings:
        print("-- WARNINGS --")
        for w in warnings:
            print(f"[WARN] {w}")

    if errors:
        print("-- ERRORS --")
        for e in errors:
            print(f"[ERROR] {e}")
        return 1

    print("Preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
