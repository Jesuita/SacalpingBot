"""
Modulo de ordenes de Binance.
Crear ordenes de mercado/limite, cancelar, consultar estado.
Validacion de cantidad y precision segun filtros del par.
"""

import math
import logging
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from binance_client import obtener_info_par, _check_rate_limit

logger = logging.getLogger("binance_orders")


# ─── Utilidades de precision ──────────────────────────────────
def _calcular_precision(step_size: str) -> int:
    """Calcula la cantidad de decimales a partir del step_size."""
    step = step_size.rstrip("0")
    if "." in step:
        return len(step.split(".")[1])
    return 0


def _ajustar_cantidad(cantidad: float, step_size: str) -> float:
    """
    Ajusta la cantidad al step_size del par.
    Redondea hacia abajo para no exceder el balance.
    """
    precision = _calcular_precision(step_size)
    factor = 10 ** precision
    return math.floor(cantidad * factor) / factor


def _ajustar_precio(precio: float, tick_size: str) -> float:
    """Ajusta el precio al tick_size del par."""
    precision = _calcular_precision(tick_size)
    factor = 10 ** precision
    return math.floor(precio * factor) / factor


# ─── Validacion previa a la orden ─────────────────────────────
def validar_orden(
    cliente: Client,
    symbol: str,
    cantidad: float,
    precio: Optional[float] = None,
) -> dict:
    """
    Valida que una orden cumpla con los filtros de Binance.

    Args:
        cliente: Cliente de Binance
        symbol: Par de trading
        cantidad: Cantidad del activo base
        precio: Precio (para ordenes limite). Si None, se estima con precio actual.

    Returns:
        dict con: valido (bool), cantidad_ajustada, precio_ajustado, errores (lista)
    """
    errores = []
    info = obtener_info_par(cliente, symbol)

    # Ajustar cantidad al step_size
    cantidad_ajustada = _ajustar_cantidad(cantidad, info["step_size"])
    min_qty = float(info["min_qty"])
    max_qty = float(info["max_qty"])

    if cantidad_ajustada < min_qty:
        errores.append(f"Cantidad {cantidad_ajustada} menor al minimo {min_qty}")
    if cantidad_ajustada > max_qty:
        errores.append(f"Cantidad {cantidad_ajustada} mayor al maximo {max_qty}")

    # Validar notional minimo
    precio_ref = precio
    if precio_ref is None:
        try:
            from binance_client import obtener_precio
            precio_ref = obtener_precio(cliente, symbol)
        except Exception:
            precio_ref = 0

    notional = cantidad_ajustada * precio_ref if precio_ref else 0
    min_notional = float(info["min_notional"])
    if notional > 0 and notional < min_notional:
        errores.append(f"Notional ${notional:.2f} menor al minimo ${min_notional:.2f}")

    # Ajustar precio si es orden limite
    precio_ajustado = _ajustar_precio(precio, info["tick_size"]) if precio else None

    return {
        "valido": len(errores) == 0,
        "cantidad_ajustada": cantidad_ajustada,
        "precio_ajustado": precio_ajustado,
        "step_size": info["step_size"],
        "tick_size": info["tick_size"],
        "min_notional": min_notional,
        "errores": errores,
    }


# ─── Orden de mercado ─────────────────────────────────────────
def orden_mercado(
    cliente: Client,
    symbol: str,
    lado: str,
    cantidad: float,
    validar: bool = True,
) -> dict:
    """
    Crea una orden de mercado (compra o venta).

    Args:
        cliente: Cliente de Binance
        symbol: Par (ej: BTCUSDT)
        lado: "BUY" o "SELL"
        cantidad: Cantidad del activo base
        validar: Si True, valida filtros antes de enviar

    Returns:
        dict con la respuesta de Binance (orderId, fills, status, etc.)
    """
    lado = lado.upper()
    if lado not in ("BUY", "SELL"):
        raise ValueError(f"Lado invalido: {lado}. Usar 'BUY' o 'SELL'")

    if validar:
        val = validar_orden(cliente, symbol, cantidad)
        if not val["valido"]:
            raise ValueError(f"Orden invalida para {symbol}: {', '.join(val['errores'])}")
        cantidad = val["cantidad_ajustada"]

    logger.info("Creando orden MARKET %s %s qty=%.8f", lado, symbol, cantidad)
    _check_rate_limit()

    try:
        if lado == "BUY":
            orden = cliente.order_market_buy(symbol=symbol.upper(), quantity=cantidad)
        else:
            orden = cliente.order_market_sell(symbol=symbol.upper(), quantity=cantidad)

        # Extraer precio promedio ponderado de los fills
        fills = orden.get("fills", [])
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
            avg_price = total_cost / total_qty if total_qty > 0 else 0
            orden["_avg_price"] = avg_price
            orden["_total_commission"] = sum(float(f["commission"]) for f in fills)

        logger.info("Orden MARKET %s %s ejecutada: orderId=%s, status=%s",
                     lado, symbol, orden.get("orderId"), orden.get("status"))
        return orden

    except BinanceAPIException as e:
        logger.error("Error creando orden MARKET %s %s: %s (codigo: %s)",
                     lado, symbol, e.message, e.code)
        raise
    except BinanceRequestException as e:
        logger.error("Error de red creando orden: %s", e)
        raise


# ─── Orden de mercado por monto quote (USDT) ──────────────────
def orden_mercado_por_usdt(
    cliente: Client,
    symbol: str,
    lado: str,
    usdt_amount: float,
) -> dict:
    """
    Crea una orden de mercado especificando el monto en USDT (quote).

    Args:
        cliente: Cliente de Binance
        symbol: Par (ej: BTCUSDT)
        lado: "BUY" o "SELL"
        usdt_amount: Monto en USDT a gastar/recibir

    Returns:
        dict con la respuesta de Binance
    """
    lado = lado.upper()
    if lado not in ("BUY", "SELL"):
        raise ValueError(f"Lado invalido: {lado}. Usar 'BUY' o 'SELL'")

    logger.info("Creando orden MARKET %s %s quoteOrderQty=%.2f USDT", lado, symbol, usdt_amount)
    _check_rate_limit()

    try:
        orden = cliente.create_order(
            symbol=symbol.upper(),
            side=lado,
            type="MARKET",
            quoteOrderQty=f"{usdt_amount:.2f}",
        )

        fills = orden.get("fills", [])
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
            avg_price = total_cost / total_qty if total_qty > 0 else 0
            orden["_avg_price"] = avg_price

        logger.info("Orden MARKET (quote) %s %s ejecutada: orderId=%s",
                     lado, symbol, orden.get("orderId"))
        return orden

    except BinanceAPIException as e:
        logger.error("Error creando orden MARKET quote %s %s: %s (codigo: %s)",
                     lado, symbol, e.message, e.code)
        raise


# ─── Orden limite ──────────────────────────────────────────────
def orden_limite(
    cliente: Client,
    symbol: str,
    lado: str,
    cantidad: float,
    precio: float,
    time_in_force: str = "GTC",
    validar: bool = True,
) -> dict:
    """
    Crea una orden limite.

    Args:
        cliente: Cliente de Binance
        symbol: Par (ej: BTCUSDT)
        lado: "BUY" o "SELL"
        cantidad: Cantidad del activo base
        precio: Precio limite
        time_in_force: GTC (Good-Til-Cancelled), IOC (Immediate-Or-Cancel), FOK (Fill-Or-Kill)
        validar: Si True, valida filtros antes de enviar

    Returns:
        dict con la respuesta de Binance
    """
    lado = lado.upper()
    if lado not in ("BUY", "SELL"):
        raise ValueError(f"Lado invalido: {lado}. Usar 'BUY' o 'SELL'")

    if validar:
        val = validar_orden(cliente, symbol, cantidad, precio)
        if not val["valido"]:
            raise ValueError(f"Orden invalida: {', '.join(val['errores'])}")
        cantidad = val["cantidad_ajustada"]
        precio = val["precio_ajustado"]

    logger.info("Creando orden LIMIT %s %s qty=%.8f price=%.8f", lado, symbol, cantidad, precio)
    _check_rate_limit()

    try:
        orden = cliente.create_order(
            symbol=symbol.upper(),
            side=lado,
            type="LIMIT",
            timeInForce=time_in_force,
            quantity=cantidad,
            price=f"{precio:.8f}".rstrip("0").rstrip("."),
        )

        logger.info("Orden LIMIT %s %s creada: orderId=%s, status=%s",
                     lado, symbol, orden.get("orderId"), orden.get("status"))
        return orden

    except BinanceAPIException as e:
        logger.error("Error creando orden LIMIT %s %s: %s (codigo: %s)",
                     lado, symbol, e.message, e.code)
        raise


# ─── Cancelar orden ───────────────────────────────────────────
def cancelar_orden(cliente: Client, symbol: str, order_id: int) -> dict:
    """
    Cancela una orden abierta.

    Args:
        cliente: Cliente de Binance
        symbol: Par de trading
        order_id: ID de la orden a cancelar

    Returns:
        dict con la respuesta de cancelacion
    """
    logger.info("Cancelando orden %d de %s", order_id, symbol)
    _check_rate_limit()

    try:
        resultado = cliente.cancel_order(symbol=symbol.upper(), orderId=order_id)
        logger.info("Orden %d cancelada. Status: %s", order_id, resultado.get("status"))
        return resultado

    except BinanceAPIException as e:
        if e.code == -2011:
            logger.warning("Orden %d no encontrada o ya cancelada: %s", order_id, e.message)
        else:
            logger.error("Error cancelando orden %d: %s (codigo: %s)", order_id, e.message, e.code)
        raise


# ─── Estado de orden ──────────────────────────────────────────
def consultar_orden(cliente: Client, symbol: str, order_id: int) -> dict:
    """
    Consulta el estado de una orden.

    Args:
        cliente: Cliente de Binance
        symbol: Par de trading
        order_id: ID de la orden

    Returns:
        dict con status, executedQty, price, etc.
    """
    _check_rate_limit()

    try:
        orden = cliente.get_order(symbol=symbol.upper(), orderId=order_id)
        logger.debug("Estado orden %d (%s): %s", order_id, symbol, orden.get("status"))
        return orden

    except BinanceAPIException as e:
        logger.error("Error consultando orden %d: %s (codigo: %s)", order_id, e.message, e.code)
        raise


# ─── Ordenes abiertas ─────────────────────────────────────────
def ordenes_abiertas(cliente: Client, symbol: Optional[str] = None) -> list[dict]:
    """
    Lista ordenes abiertas de un par o de todos.

    Args:
        cliente: Cliente de Binance
        symbol: Par especifico. Si None, devuelve todas las abiertas.

    Returns:
        Lista de ordenes abiertas
    """
    _check_rate_limit()

    try:
        if symbol:
            ordenes = cliente.get_open_orders(symbol=symbol.upper())
        else:
            ordenes = cliente.get_open_orders()
        logger.debug("Ordenes abiertas%s: %d",
                      f" ({symbol})" if symbol else "", len(ordenes))
        return ordenes

    except BinanceAPIException as e:
        logger.error("Error listando ordenes abiertas: %s (codigo: %s)", e.message, e.code)
        raise


# ─── Historial de ordenes ─────────────────────────────────────
def historial_ordenes(cliente: Client, symbol: str, limit: int = 50) -> list[dict]:
    """
    Obtiene el historial reciente de ordenes de un par.

    Args:
        cliente: Cliente de Binance
        symbol: Par de trading
        limit: Cantidad maxima de ordenes (default 50, max 1000)

    Returns:
        Lista de ordenes historicas
    """
    _check_rate_limit()

    try:
        ordenes = cliente.get_all_orders(symbol=symbol.upper(), limit=limit)
        logger.debug("Historial de %s: %d ordenes", symbol, len(ordenes))
        return ordenes

    except BinanceAPIException as e:
        logger.error("Error obteniendo historial de %s: %s (codigo: %s)",
                     symbol, e.message, e.code)
        raise
