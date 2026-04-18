"""
Modulo de conexion a Binance.
Inicializa el cliente con soporte para Testnet y produccion.
Provee funciones para verificar credenciales, consultar balances y precios.
"""

import os
import logging
import time
from typing import Optional

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

load_dotenv()

logger = logging.getLogger("binance_client")

# ─── Constantes ───────────────────────────────────────────────
TESTNET_API_URL = "https://testnet.binance.vision/api"
TESTNET_WS_URL = "wss://testnet.binance.vision/ws"

# Rate limit: maximo de requests por minuto (Binance permite 1200/min)
MAX_REQUESTS_PER_MINUTE = 1000
_request_timestamps: list[float] = []


# ─── Rate limiting ────────────────────────────────────────────
def _check_rate_limit():
    """Verifica que no estemos excediendo el rate limit de Binance."""
    now = time.time()
    # Limpiar timestamps mas viejos que 60 segundos
    while _request_timestamps and _request_timestamps[0] < now - 60:
        _request_timestamps.pop(0)
    if len(_request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        wait = 60 - (now - _request_timestamps[0])
        if wait > 0:
            logger.warning("Rate limit alcanzado, esperando %.1fs", wait)
            time.sleep(wait)
    _request_timestamps.append(time.time())


# ─── Inicializacion del cliente ───────────────────────────────
def crear_cliente(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: Optional[bool] = None,
) -> Client:
    """
    Crea e inicializa el cliente de Binance.

    Args:
        api_key: Clave API. Si es None, se lee de BINANCE_API_KEY en .env
        api_secret: Secreto API. Si es None, se lee de BINANCE_API_SECRET en .env
        testnet: Si True, usa Testnet. Si None, lee BINANCE_TESTNET de .env

    Returns:
        Client de python-binance configurado
    """
    key = api_key or os.getenv("BINANCE_API_KEY", "")
    secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
    usar_testnet = testnet if testnet is not None else os.getenv("BINANCE_TESTNET", "True").lower() in ("true", "1", "yes")

    if not key or not secret or key == "tu_api_key_aqui":
        logger.warning("API keys no configuradas. Funcionalidad limitada a endpoints publicos.")

    try:
        if usar_testnet:
            cliente = Client(key, secret, testnet=True)
            cliente.API_URL = TESTNET_API_URL
            logger.info("Cliente Binance inicializado en TESTNET")
        else:
            cliente = Client(key, secret)
            logger.info("Cliente Binance inicializado en PRODUCCION")
        return cliente

    except BinanceAPIException as e:
        logger.error("Error de API Binance al crear cliente: %s (codigo: %s)", e.message, e.code)
        raise
    except BinanceRequestException as e:
        logger.error("Error de conexion con Binance: %s", e)
        raise


# ─── Verificacion de conexion y credenciales ──────────────────
def verificar_conexion(cliente: Client) -> dict:
    """
    Verifica que la conexion y credenciales sean validas.

    Returns:
        dict con status, server_time, y permissions (si auth es valida)
    """
    resultado = {"status": "error", "server_time": None, "permissions": []}

    try:
        # Test 1: ping al servidor (no requiere auth)
        _check_rate_limit()
        cliente.ping()
        logger.info("Ping a Binance OK")

        # Test 2: hora del servidor
        _check_rate_limit()
        server_time = cliente.get_server_time()
        resultado["server_time"] = server_time.get("serverTime")
        logger.info("Server time: %s", resultado["server_time"])

        # Test 3: verificar credenciales con account info
        _check_rate_limit()
        account = cliente.get_account()
        resultado["permissions"] = [
            p for p in ["SPOT", "MARGIN", "FUTURES"]
            if account.get(f"can{p.capitalize()}" if p != "SPOT" else "canTrade")
        ]
        # Permisos reales del endpoint
        resultado["permissions"] = []
        if account.get("canTrade"):
            resultado["permissions"].append("SPOT_TRADE")
        if account.get("canWithdraw"):
            resultado["permissions"].append("WITHDRAW")
        if account.get("canDeposit"):
            resultado["permissions"].append("DEPOSIT")

        resultado["status"] = "ok"
        logger.info("Credenciales validas. Permisos: %s", resultado["permissions"])

    except BinanceAPIException as e:
        if e.code == -2015:
            resultado["status"] = "credenciales_invalidas"
            logger.error("API key o secret invalidos: %s", e.message)
        elif e.code == -1003:
            resultado["status"] = "rate_limited"
            logger.error("Rate limit excedido: %s", e.message)
        else:
            resultado["status"] = f"api_error_{e.code}"
            logger.error("Error API: %s (codigo: %s)", e.message, e.code)
    except BinanceRequestException as e:
        resultado["status"] = "conexion_fallida"
        logger.error("Error de red: %s", e)
    except Exception as e:
        resultado["status"] = "error_desconocido"
        logger.error("Error inesperado verificando conexion: %s", e)

    return resultado


# ─── Balance de cuenta ────────────────────────────────────────
def obtener_balance(cliente: Client, asset: str = "USDT") -> dict:
    """
    Obtiene el balance de un activo especifico.

    Args:
        cliente: Cliente de Binance
        asset: Simbolo del activo (ej: USDT, BTC, ETH)

    Returns:
        dict con free, locked, total
    """
    _check_rate_limit()
    try:
        balance = cliente.get_asset_balance(asset=asset.upper())
        if balance is None:
            return {"asset": asset, "free": 0.0, "locked": 0.0, "total": 0.0}

        free = float(balance.get("free", 0))
        locked = float(balance.get("locked", 0))
        return {
            "asset": asset.upper(),
            "free": free,
            "locked": locked,
            "total": free + locked,
        }
    except BinanceAPIException as e:
        logger.error("Error obteniendo balance de %s: %s (codigo: %s)", asset, e.message, e.code)
        raise
    except BinanceRequestException as e:
        logger.error("Error de red obteniendo balance: %s", e)
        raise


def obtener_todos_los_balances(cliente: Client, min_total: float = 0.0) -> list[dict]:
    """
    Obtiene todos los balances con total mayor a min_total.

    Args:
        cliente: Cliente de Binance
        min_total: Filtro minimo de balance total

    Returns:
        Lista de dicts con asset, free, locked, total
    """
    _check_rate_limit()
    try:
        account = cliente.get_account()
        balances = []
        for b in account.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            total = free + locked
            if total > min_total:
                balances.append({
                    "asset": b["asset"],
                    "free": free,
                    "locked": locked,
                    "total": total,
                })
        return balances
    except BinanceAPIException as e:
        logger.error("Error obteniendo balances: %s (codigo: %s)", e.message, e.code)
        raise


# ─── Precio actual ────────────────────────────────────────────
def obtener_precio(cliente: Client, symbol: str) -> float:
    """
    Obtiene el precio actual de un par.

    Args:
        cliente: Cliente de Binance
        symbol: Par de trading (ej: BTCUSDT)

    Returns:
        Precio como float
    """
    _check_rate_limit()
    try:
        ticker = cliente.get_symbol_ticker(symbol=symbol.upper())
        precio = float(ticker["price"])
        logger.debug("Precio %s: %.8f", symbol, precio)
        return precio
    except BinanceAPIException as e:
        logger.error("Error obteniendo precio de %s: %s (codigo: %s)", symbol, e.message, e.code)
        raise
    except BinanceRequestException as e:
        logger.error("Error de red obteniendo precio: %s", e)
        raise


def obtener_precios_multiples(cliente: Client, symbols: list[str]) -> dict[str, float]:
    """
    Obtiene precios de multiples pares en una sola llamada.

    Args:
        cliente: Cliente de Binance
        symbols: Lista de pares (ej: ["BTCUSDT", "ETHUSDT"])

    Returns:
        Dict {symbol: precio}
    """
    _check_rate_limit()
    try:
        tickers = cliente.get_all_tickers()
        symbol_set = {s.upper() for s in symbols}
        return {
            t["symbol"]: float(t["price"])
            for t in tickers
            if t["symbol"] in symbol_set
        }
    except BinanceAPIException as e:
        logger.error("Error obteniendo precios multiples: %s (codigo: %s)", e.message, e.code)
        raise


# ─── Informacion del par (filtros, precisiones) ──────────────
def obtener_info_par(cliente: Client, symbol: str) -> dict:
    """
    Obtiene informacion del par: filtros de cantidad, precio, notional minimo.

    Returns:
        dict con step_size, tick_size, min_notional, min_qty, max_qty, base_precision, quote_precision
    """
    _check_rate_limit()
    try:
        info = cliente.get_symbol_info(symbol.upper())
        if info is None:
            raise ValueError(f"Par {symbol} no encontrado en Binance")

        resultado = {
            "symbol": info["symbol"],
            "base_asset": info["baseAsset"],
            "quote_asset": info["quoteAsset"],
            "base_precision": info["baseAssetPrecision"],
            "quote_precision": info["quoteAssetPrecision"],
            "step_size": "0.00000001",
            "tick_size": "0.01",
            "min_notional": "10.0",
            "min_qty": "0.00000001",
            "max_qty": "9999999.0",
        }

        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                resultado["step_size"] = f["stepSize"]
                resultado["min_qty"] = f["minQty"]
                resultado["max_qty"] = f["maxQty"]
            elif f["filterType"] == "PRICE_FILTER":
                resultado["tick_size"] = f["tickSize"]
            elif f["filterType"] == "NOTIONAL":
                resultado["min_notional"] = f.get("minNotional", "10.0")
            elif f["filterType"] == "MIN_NOTIONAL":
                resultado["min_notional"] = f.get("minNotional", "10.0")

        logger.debug("Info par %s: step=%s, tick=%s, min_notional=%s",
                      symbol, resultado["step_size"], resultado["tick_size"], resultado["min_notional"])
        return resultado

    except BinanceAPIException as e:
        logger.error("Error obteniendo info de %s: %s (codigo: %s)", symbol, e.message, e.code)
        raise


# ─── Klines (velas) ──────────────────────────────────────────
def obtener_klines(
    cliente: Client,
    symbol: str,
    interval: str = Client.KLINE_INTERVAL_1MINUTE,
    limit: int = 100,
) -> list[list]:
    """
    Obtiene velas historicas.

    Args:
        cliente: Cliente de Binance
        symbol: Par (ej: BTCUSDT)
        interval: Intervalo (ej: Client.KLINE_INTERVAL_1MINUTE)
        limit: Cantidad de velas (max 1000)

    Returns:
        Lista de velas [open_time, open, high, low, close, volume, ...]
    """
    _check_rate_limit()
    try:
        klines = cliente.get_klines(symbol=symbol.upper(), interval=interval, limit=limit)
        logger.debug("Obtenidas %d velas de %s (%s)", len(klines), symbol, interval)
        return klines
    except BinanceAPIException as e:
        logger.error("Error obteniendo klines de %s: %s (codigo: %s)", symbol, e.message, e.code)
        raise
    except BinanceRequestException as e:
        logger.error("Error de red obteniendo klines: %s", e)
        raise
