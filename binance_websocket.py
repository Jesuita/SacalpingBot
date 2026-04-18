"""
Modulo de WebSocket de Binance.
Streams en tiempo real de precios (ticker) y velas (klines).
Reconexion automatica ante desconexiones.
Callbacks configurables para procesar datos.
"""

import os
import logging
import threading
import time
from typing import Callable, Optional

from dotenv import load_dotenv
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException

load_dotenv()

logger = logging.getLogger("binance_websocket")

# ─── Constantes ───────────────────────────────────────────────
TESTNET_WS_URL = "wss://testnet.binance.vision/ws"
MAX_RECONNECT_INTENTOS = 10
RECONNECT_DELAY_BASE = 2  # segundos, se duplica en cada intento (backoff exponencial)


class BinanceWebSocket:
    """
    Gestor de WebSocket de Binance con reconexion automatica.

    Ejemplo de uso:
        def mi_callback(datos):
            print(f"Precio: {datos['precio']}")

        ws = BinanceWebSocket()
        ws.iniciar_ticker("BTCUSDT", mi_callback)
        # ... mas tarde:
        ws.detener()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
    ):
        self._api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self._api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self._testnet = testnet if testnet is not None else os.getenv("BINANCE_TESTNET", "True").lower() in ("true", "1", "yes")

        self._twm: Optional[ThreadedWebsocketManager] = None
        self._streams: dict[str, str] = {}  # nombre → stream_name
        self._callbacks: dict[str, Callable] = {}  # nombre → callback del usuario
        self._activo = False
        self._lock = threading.Lock()
        self._reconnect_count = 0

    # ─── Iniciar el manager ────────────────────────────────────
    def _iniciar_twm(self):
        """Crea e inicia el ThreadedWebsocketManager."""
        if self._twm is not None:
            try:
                self._twm.stop()
            except Exception:
                pass

        self._twm = ThreadedWebsocketManager(
            api_key=self._api_key,
            api_secret=self._api_secret,
            testnet=self._testnet,
        )
        self._twm.start()
        self._activo = True
        self._reconnect_count = 0
        logger.info("WebSocket manager iniciado (testnet=%s)", self._testnet)

    # ─── Reconexion automatica ─────────────────────────────────
    def _reconectar(self, nombre: str, tipo: str, **kwargs):
        """Intenta reconectar un stream especifico con backoff exponencial."""
        if self._reconnect_count >= MAX_RECONNECT_INTENTOS:
            logger.error("Maximo de intentos de reconexion alcanzado (%d) para %s",
                         MAX_RECONNECT_INTENTOS, nombre)
            return

        self._reconnect_count += 1
        delay = min(RECONNECT_DELAY_BASE * (2 ** (self._reconnect_count - 1)), 60)
        logger.warning("Reconectando %s en %.0fs (intento %d/%d)",
                       nombre, delay, self._reconnect_count, MAX_RECONNECT_INTENTOS)
        time.sleep(delay)

        try:
            with self._lock:
                if not self._activo:
                    return
                if tipo == "ticker":
                    self._registrar_ticker(kwargs["symbol"], self._callbacks.get(nombre))
                elif tipo == "kline":
                    self._registrar_kline(kwargs["symbol"], kwargs["interval"], self._callbacks.get(nombre))
            self._reconnect_count = 0
            logger.info("Reconexion exitosa para %s", nombre)
        except Exception as e:
            logger.error("Fallo reconexion de %s: %s", nombre, e)
            # Reintentar en un thread separado para no bloquear
            threading.Thread(
                target=self._reconectar,
                args=(nombre, tipo),
                kwargs=kwargs,
                daemon=True,
            ).start()

    # ─── Wrapper de callback con deteccion de desconexion ──────
    def _crear_callback_wrapper(self, nombre: str, callback: Callable, tipo: str, **kwargs):
        """Crea un wrapper que detecta errores/cierre y dispara reconexion."""
        def wrapper(msg):
            if msg.get("e") == "error":
                logger.error("Error en stream %s: %s", nombre, msg)
                threading.Thread(
                    target=self._reconectar,
                    args=(nombre, tipo),
                    kwargs=kwargs,
                    daemon=True,
                ).start()
                return
            try:
                callback(msg)
            except Exception as e:
                logger.error("Error en callback de %s: %s", nombre, e)
        return wrapper

    # ─── Stream de ticker (precio en tiempo real) ──────────────
    def _registrar_ticker(self, symbol: str, callback: Callable) -> str:
        """Registra un stream de ticker sin wrapper (uso interno)."""
        nombre = f"ticker_{symbol.lower()}"
        stream = self._twm.start_symbol_ticker_socket(
            callback=self._crear_callback_wrapper(nombre, callback, "ticker", symbol=symbol),
            symbol=symbol.upper(),
        )
        self._streams[nombre] = stream
        return nombre

    def iniciar_ticker(self, symbol: str, callback: Callable) -> str:
        """
        Inicia un stream de precio en tiempo real para un par.

        Args:
            symbol: Par de trading (ej: "BTCUSDT")
            callback: Funcion que recibe un dict con los datos del ticker.
                      Campos principales: symbol, price, priceChange, volume, etc.

        Returns:
            Nombre del stream para referencia

        Ejemplo de datos que recibe el callback:
            {
                "e": "24hrTicker",
                "s": "BTCUSDT",
                "c": "73000.00",  # ultimo precio
                "p": "500.00",    # cambio de precio
                "P": "0.69",      # cambio porcentual
                "v": "12345.678", # volumen base
                "q": "901234567", # volumen quote
            }
        """
        with self._lock:
            if not self._activo:
                self._iniciar_twm()

            nombre = f"ticker_{symbol.lower()}"
            self._callbacks[nombre] = callback
            self._registrar_ticker(symbol, callback)
            logger.info("Ticker stream iniciado para %s", symbol)
            return nombre

    # ─── Stream de klines (velas en tiempo real) ───────────────
    def _registrar_kline(self, symbol: str, interval: str, callback: Callable) -> str:
        """Registra un stream de klines sin wrapper (uso interno)."""
        nombre = f"kline_{symbol.lower()}_{interval}"
        stream = self._twm.start_kline_socket(
            callback=self._crear_callback_wrapper(nombre, callback, "kline", symbol=symbol, interval=interval),
            symbol=symbol.upper(),
            interval=interval,
        )
        self._streams[nombre] = stream
        return nombre

    def iniciar_klines(self, symbol: str, interval: str, callback: Callable) -> str:
        """
        Inicia un stream de velas en tiempo real.

        Args:
            symbol: Par de trading (ej: "BTCUSDT")
            interval: Intervalo de velas (ej: "1m", "5m", "15m", "1h")
            callback: Funcion que recibe un dict con los datos de la vela.

        Returns:
            Nombre del stream para referencia

        Ejemplo de datos que recibe el callback:
            {
                "e": "kline",
                "s": "BTCUSDT",
                "k": {
                    "t": 1618884000000,  # apertura timestamp
                    "T": 1618884059999,  # cierre timestamp
                    "s": "BTCUSDT",
                    "i": "1m",           # intervalo
                    "o": "73000.00",     # open
                    "h": "73100.00",     # high
                    "l": "72900.00",     # low
                    "c": "73050.00",     # close
                    "v": "100.123",      # volumen
                    "x": false,          # vela cerrada?
                }
            }
        """
        with self._lock:
            if not self._activo:
                self._iniciar_twm()

            nombre = f"kline_{symbol.lower()}_{interval}"
            self._callbacks[nombre] = callback
            self._registrar_kline(symbol, interval, callback)
            logger.info("Kline stream iniciado para %s (%s)", symbol, interval)
            return nombre

    # ─── Stream de multiples tickers ───────────────────────────
    def iniciar_tickers_multiples(self, symbols: list[str], callback: Callable) -> str:
        """
        Inicia un stream de tickers para multiples pares.

        Args:
            symbols: Lista de pares (ej: ["BTCUSDT", "ETHUSDT"])
            callback: Funcion que recibe una lista de dicts de ticker

        Returns:
            Nombre del stream
        """
        with self._lock:
            if not self._activo:
                self._iniciar_twm()

            nombre = "tickers_multi"
            self._callbacks[nombre] = callback
            stream = self._twm.start_miniticker_socket(
                callback=self._crear_callback_wrapper(nombre, callback, "ticker_multi"),
            )
            self._streams[nombre] = stream
            logger.info("Multi-ticker stream iniciado para %d pares", len(symbols))
            return nombre

    # ─── Detener un stream especifico ──────────────────────────
    def detener_stream(self, nombre: str):
        """Detiene un stream especifico por nombre."""
        with self._lock:
            stream = self._streams.pop(nombre, None)
            self._callbacks.pop(nombre, None)
            if stream and self._twm:
                try:
                    self._twm.stop_socket(stream)
                    logger.info("Stream %s detenido", nombre)
                except Exception as e:
                    logger.warning("Error deteniendo stream %s: %s", nombre, e)

    # ─── Detener todo ──────────────────────────────────────────
    def detener(self):
        """Detiene todos los streams y el manager."""
        with self._lock:
            self._activo = False
            self._streams.clear()
            self._callbacks.clear()
            if self._twm:
                try:
                    self._twm.stop()
                    logger.info("WebSocket manager detenido")
                except Exception as e:
                    logger.warning("Error deteniendo TWM: %s", e)
                self._twm = None

    # ─── Estado ────────────────────────────────────────────────
    @property
    def activo(self) -> bool:
        return self._activo

    @property
    def streams_activos(self) -> list[str]:
        return list(self._streams.keys())

    def __del__(self):
        self.detener()
