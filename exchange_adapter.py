"""
exchange_adapter.py -- Capa de abstraccion para multiples exchanges.

Interfaz unificada para obtener precios, velas y ejecutar ordenes.
Soporta Binance (produccion y testnet) como implementacion base.
Nuevos exchanges se agregan creando subclases de ExchangeAdapter.

No requiere librerias externas mas alla de 'requests'.
"""

import hashlib
import hmac
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

# ---------------------------------------------------------------------------
#  INTERFAZ BASE
# ---------------------------------------------------------------------------

class ExchangeAdapter:
    """Interfaz abstracta para un exchange."""

    name: str = "base"

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

    def get_price(self, symbol: str) -> Optional[float]:
        """Obtiene precio actual de un par."""
        raise NotImplementedError

    def get_candles(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[List]:
        """Obtiene velas OHLCV. Retorna [[time, open, high, low, close, volume], ...]"""
        raise NotImplementedError

    def get_orderbook(self, symbol: str, limit: int = 5) -> Dict:
        """Obtiene orderbook: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}"""
        raise NotImplementedError

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET") -> Dict:
        """Coloca una orden. Retorna dict con orderId, status, executedQty, etc."""
        raise NotImplementedError

    def get_balance(self, asset: str = "USDT") -> float:
        """Obtiene balance de un activo."""
        raise NotImplementedError

    def get_symbol_info(self, symbol: str) -> Dict:
        """Info del par: min_qty, step_size, min_notional, etc."""
        raise NotImplementedError

    def ping(self) -> bool:
        """Verifica conectividad con el exchange."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  BINANCE
# ---------------------------------------------------------------------------

class BinanceAdapter(ExchangeAdapter):
    """Implementacion para Binance (spot)."""

    name = "binance"

    MAINNET_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL

    def _public_get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _signed_request(self, method: str, endpoint: str, params: dict = None) -> dict:
        if not self.api_key or not self.api_secret:
            raise ValueError("API key/secret required for signed requests")
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}{endpoint}"
        if method == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=10)
        else:
            resp = requests.post(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            data = self._public_get("/api/v3/ticker/price", {"symbol": symbol})
            return float(data.get("price", 0))
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[List]:
        try:
            data = self._public_get("/api/v3/klines", {
                "symbol": symbol, "interval": interval, "limit": limit
            })
            return [
                [int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                for c in data
            ]
        except Exception:
            return []

    def get_orderbook(self, symbol: str, limit: int = 5) -> Dict:
        try:
            data = self._public_get("/api/v3/depth", {"symbol": symbol, "limit": limit})
            return {
                "bids": [[float(b[0]), float(b[1])] for b in data.get("bids", [])],
                "asks": [[float(a[0]), float(a[1])] for a in data.get("asks", [])],
            }
        except Exception:
            return {"bids": [], "asks": []}

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET") -> Dict:
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type,
            "quantity": f"{quantity:.8f}",
        }
        return self._signed_request("POST", "/api/v3/order", params)

    def get_balance(self, asset: str = "USDT") -> float:
        try:
            data = self._signed_request("GET", "/api/v3/account")
            for b in data.get("balances", []):
                if b["asset"] == asset:
                    return float(b["free"])
            return 0.0
        except Exception:
            return 0.0

    def get_symbol_info(self, symbol: str) -> Dict:
        try:
            data = self._public_get("/api/v3/exchangeInfo", {"symbol": symbol})
            for s in data.get("symbols", []):
                if s["symbol"] == symbol:
                    info = {"symbol": symbol, "status": s.get("status")}
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            info["min_qty"] = float(f["minQty"])
                            info["step_size"] = float(f["stepSize"])
                        elif f["filterType"] == "NOTIONAL":
                            info["min_notional"] = float(f.get("minNotional", 0))
                        elif f["filterType"] == "MIN_NOTIONAL":
                            info["min_notional"] = float(f.get("minNotional", 0))
                    return info
            return {}
        except Exception:
            return {}

    def ping(self) -> bool:
        try:
            self._public_get("/api/v3/ping")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
#  BYBIT (placeholder para futuro)
# ---------------------------------------------------------------------------

class BybitAdapter(ExchangeAdapter):
    """Placeholder para Bybit. Metodos publicos basicos."""

    name = "bybit"

    MAINNET_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            # Bybit usa formato diferente: BTCUSDT
            resp = requests.get(
                f"{self.base_url}/v5/market/tickers",
                params={"category": "spot", "symbol": symbol},
                timeout=10,
            )
            data = resp.json()
            tickers = data.get("result", {}).get("list", [])
            if tickers:
                return float(tickers[0].get("lastPrice", 0))
            return None
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[List]:
        # Bybit usa intervalos: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        interval_map = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "1h": "60"}
        bybit_interval = interval_map.get(interval, "1")
        try:
            resp = requests.get(
                f"{self.base_url}/v5/market/kline",
                params={"category": "spot", "symbol": symbol, "interval": bybit_interval, "limit": limit},
                timeout=10,
            )
            data = resp.json()
            candles = []
            for c in data.get("result", {}).get("list", []):
                candles.append([int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])])
            candles.reverse()  # Bybit retorna mas reciente primero
            return candles
        except Exception:
            return []

    def get_orderbook(self, symbol: str, limit: int = 5) -> Dict:
        try:
            resp = requests.get(
                f"{self.base_url}/v5/market/orderbook",
                params={"category": "spot", "symbol": symbol, "limit": limit},
                timeout=10,
            )
            data = resp.json()
            result = data.get("result", {})
            return {
                "bids": [[float(b[0]), float(b[1])] for b in result.get("b", [])],
                "asks": [[float(a[0]), float(a[1])] for a in result.get("a", [])],
            }
        except Exception:
            return {"bids": [], "asks": []}

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET") -> Dict:
        raise NotImplementedError("Bybit order placement not implemented yet")

    def get_balance(self, asset: str = "USDT") -> float:
        raise NotImplementedError("Bybit balance not implemented yet")

    def get_symbol_info(self, symbol: str) -> Dict:
        return {}

    def ping(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/v5/market/time", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
#  FACTORY
# ---------------------------------------------------------------------------

EXCHANGE_REGISTRY = {
    "binance": BinanceAdapter,
    "bybit": BybitAdapter,
}


def create_adapter(exchange: str, api_key: str = "", api_secret: str = "", testnet: bool = False) -> ExchangeAdapter:
    """Crea un adapter por nombre de exchange."""
    cls = EXCHANGE_REGISTRY.get(exchange.lower())
    if cls is None:
        raise ValueError(f"Exchange no soportado: {exchange}. Disponibles: {list(EXCHANGE_REGISTRY.keys())}")
    return cls(api_key=api_key, api_secret=api_secret, testnet=testnet)


def list_exchanges() -> List[str]:
    """Lista exchanges soportados."""
    return list(EXCHANGE_REGISTRY.keys())
