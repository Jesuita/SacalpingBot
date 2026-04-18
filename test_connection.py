"""
Script de prueba para verificar la conexion con Binance.
Valida credenciales, WebSocket, balance y precio.

Uso:
    py -3 test_connection.py
    py -3 test_connection.py --symbol ETHUSDT
    py -3 test_connection.py --no-ws          # sin test de WebSocket
"""

import sys
import os
import time
import logging
import argparse
import threading

# Configurar logging antes de importar modulos
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)-20s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_connection")

# ─── Imports del proyecto ─────────────────────────────────────
try:
    from binance_client import (
        crear_cliente,
        verificar_conexion,
        obtener_balance,
        obtener_todos_los_balances,
        obtener_precio,
        obtener_precios_multiples,
        obtener_info_par,
        obtener_klines,
    )
    from binance_websocket import BinanceWebSocket
    from binance_orders import validar_orden
except ImportError as e:
    print(f"\nError importando modulos: {e}")
    print("Asegurate de haber instalado las dependencias:")
    print("  pip install python-binance python-dotenv\n")
    sys.exit(1)


def separador(titulo: str):
    print(f"\n{'='*50}")
    print(f"  {titulo}")
    print(f"{'='*50}")


def test_credenciales(cliente) -> bool:
    """Test 1: Verificar conexion y credenciales."""
    separador("TEST 1: Verificar conexion y credenciales")

    resultado = verificar_conexion(cliente)

    if resultado["status"] == "ok":
        print(f"  [OK] Conexion exitosa")
        print(f"  [OK] Server time: {resultado['server_time']}")
        print(f"  [OK] Permisos: {', '.join(resultado['permissions'])}")
        return True
    elif resultado["status"] == "credenciales_invalidas":
        print(f"  [FAIL] Credenciales invalidas")
        print(f"         Revisa BINANCE_API_KEY y BINANCE_API_SECRET en .env")
        return False
    else:
        print(f"  [FAIL] Status: {resultado['status']}")
        # Puede seguir con endpoints publicos
        return False


def test_precio(cliente, symbol: str) -> bool:
    """Test 2: Obtener precio actual."""
    separador(f"TEST 2: Precio actual de {symbol}")

    try:
        precio = obtener_precio(cliente, symbol)
        print(f"  [OK] {symbol}: ${precio:,.2f}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_precios_multiples(cliente, symbols: list) -> bool:
    """Test 3: Precios de multiples pares."""
    separador("TEST 3: Precios multiples")

    try:
        precios = obtener_precios_multiples(cliente, symbols)
        for sym, precio in precios.items():
            print(f"  [OK] {sym}: ${precio:,.4f}")
        print(f"  Total: {len(precios)} pares consultados")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_balance(cliente) -> bool:
    """Test 4: Consultar balance (requiere credenciales validas)."""
    separador("TEST 4: Balance de cuenta")

    try:
        balances = obtener_todos_los_balances(cliente, min_total=0.0001)
        if not balances:
            print(f"  [OK] Cuenta sin balances (normal en testnet nuevo)")
            # Intentar USDT especificamente
            usdt = obtener_balance(cliente, "USDT")
            print(f"  [OK] USDT: free={usdt['free']:.4f}, locked={usdt['locked']:.4f}")
        else:
            for b in balances[:10]:  # Mostrar max 10
                print(f"  [OK] {b['asset']:6s}: free={b['free']:.8f}, locked={b['locked']:.8f}")
            if len(balances) > 10:
                print(f"  ... y {len(balances)-10} activos mas")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        print(f"         (Normal si las credenciales no son validas)")
        return False


def test_info_par(cliente, symbol: str) -> bool:
    """Test 5: Informacion y filtros del par."""
    separador(f"TEST 5: Info del par {symbol}")

    try:
        info = obtener_info_par(cliente, symbol)
        print(f"  [OK] Base: {info['base_asset']}, Quote: {info['quote_asset']}")
        print(f"  [OK] Step size: {info['step_size']}")
        print(f"  [OK] Tick size: {info['tick_size']}")
        print(f"  [OK] Min notional: ${float(info['min_notional']):.2f}")
        print(f"  [OK] Min qty: {info['min_qty']}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_klines(cliente, symbol: str) -> bool:
    """Test 6: Obtener velas historicas."""
    separador(f"TEST 6: Klines de {symbol}")

    try:
        from binance.client import Client
        klines = obtener_klines(cliente, symbol, Client.KLINE_INTERVAL_1MINUTE, limit=5)
        print(f"  [OK] Obtenidas {len(klines)} velas")
        if klines:
            ultima = klines[-1]
            print(f"  [OK] Ultima vela: O={float(ultima[1]):.2f} H={float(ultima[2]):.2f} "
                  f"L={float(ultima[3]):.2f} C={float(ultima[4]):.2f} V={float(ultima[5]):.2f}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_validacion_orden(cliente, symbol: str) -> bool:
    """Test 7: Validar una orden simulada."""
    separador(f"TEST 7: Validar orden simulada ({symbol})")

    try:
        # Simular compra de $10 USDT
        precio = obtener_precio(cliente, symbol)
        cantidad = 10.0 / precio

        val = validar_orden(cliente, symbol, cantidad, precio)
        if val["valido"]:
            print(f"  [OK] Orden valida")
            print(f"  [OK] Cantidad ajustada: {val['cantidad_ajustada']}")
            print(f"  [OK] Precio ajustado: {val['precio_ajustado']}")
        else:
            print(f"  [WARN] Orden no valida: {', '.join(val['errores'])}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_websocket(symbol: str, duracion: int = 8) -> bool:
    """Test 8: WebSocket en tiempo real."""
    separador(f"TEST 8: WebSocket ticker {symbol} ({duracion}s)")

    mensajes = []
    error_event = threading.Event()

    def callback(msg):
        if msg.get("e") == "error":
            error_event.set()
            return
        precio = msg.get("c", "N/A")
        mensajes.append(precio)
        if len(mensajes) <= 3:
            print(f"  [WS] {symbol} precio: ${float(precio):,.2f}")

    ws = BinanceWebSocket()
    try:
        ws.iniciar_ticker(symbol, callback)
        print(f"  Esperando datos por {duracion} segundos...")

        for _ in range(duracion):
            time.sleep(1)
            if error_event.is_set():
                print(f"  [FAIL] Error en WebSocket")
                return False

        ws.detener()

        if mensajes:
            print(f"  [OK] Recibidos {len(mensajes)} mensajes en {duracion}s")
            return True
        else:
            print(f"  [WARN] No se recibieron mensajes (puede ser normal en testnet)")
            return False

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False
    finally:
        ws.detener()


def main():
    parser = argparse.ArgumentParser(description="Test de conexion con Binance")
    parser.add_argument("--symbol", default="BTCUSDT", help="Par a testear (default: BTCUSDT)")
    parser.add_argument("--no-ws", action="store_true", help="Omitir test de WebSocket")
    parser.add_argument("--no-auth", action="store_true", help="Omitir tests que requieren autenticacion")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    multi_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    print("\n" + "=" * 50)
    print("  TEST DE CONEXION CON BINANCE")
    print(f"  Par: {symbol}")
    print(f"  Testnet: {os.getenv('BINANCE_TESTNET', 'True')}")
    print("=" * 50)

    # Crear cliente
    try:
        cliente = crear_cliente()
    except Exception as e:
        print(f"\n[FATAL] No se pudo crear el cliente: {e}")
        sys.exit(1)

    resultados = {}

    # Tests publicos (no requieren auth)
    resultados["credenciales"] = test_credenciales(cliente)
    resultados["precio"] = test_precio(cliente, symbol)
    resultados["precios_multi"] = test_precios_multiples(cliente, multi_symbols)
    resultados["info_par"] = test_info_par(cliente, symbol)
    resultados["klines"] = test_klines(cliente, symbol)

    # Tests autenticados
    if not args.no_auth:
        resultados["balance"] = test_balance(cliente)
        resultados["validacion_orden"] = test_validacion_orden(cliente, symbol)
    else:
        print("\n  [SKIP] Tests de autenticacion omitidos (--no-auth)")

    # Test WebSocket
    if not args.no_ws:
        resultados["websocket"] = test_websocket(symbol)
    else:
        print("\n  [SKIP] Test de WebSocket omitido (--no-ws)")

    # Resumen
    separador("RESUMEN")
    ok = sum(1 for v in resultados.values() if v)
    total = len(resultados)
    for nombre, estado in resultados.items():
        marca = "[OK]  " if estado else "[FAIL]"
        print(f"  {marca} {nombre}")
    print(f"\n  Resultado: {ok}/{total} tests pasaron")

    if ok == total:
        print("  [OK] Conexion con Binance verificada correctamente\n")
    elif ok >= total - 2:
        print("  ~ Conexion parcial. Revisa los tests fallidos.\n")
    else:
        print("  [X] Problemas de conexion. Revisa .env y credenciales.\n")

    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
