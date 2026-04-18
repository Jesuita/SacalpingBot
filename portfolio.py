"""Modulo de analisis de correlacion entre pares para el portfolio.

Calcula correlacion de Pearson entre retornos de pares activos
usando datos de velas de Binance.
"""
import math
import time
from typing import Dict, List, Optional, Tuple

BASE_URL = "https://api.binance.com"

try:
    import requests
except ImportError:
    requests = None


def _fetch_closes(symbol: str, interval: str = "5m", limit: int = 100) -> List[float]:
    """Obtiene precios de cierre de Binance."""
    if requests is None:
        return []
    try:
        resp = requests.get(
            f"{BASE_URL}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        return [float(k[4]) for k in resp.json()]
    except Exception:
        return []


def calc_returns(closes: List[float]) -> List[float]:
    """Calcula retornos porcentuales desde precios de cierre."""
    if len(closes) < 2:
        return []
    return [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calcula correlacion de Pearson entre dos series.
    Retorna valor entre -1 y 1. Retorna 0 si datos insuficientes.
    """
    n = min(len(x), len(y))
    if n < 5:
        return 0.0
    x, y = x[:n], y[:n]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0

    return cov / (std_x * std_y)


def build_correlation_matrix(symbols: List[str], interval: str = "5m",
                              limit: int = 100) -> Dict:
    """Construye matriz de correlacion NxN para los pares dados.

    Retorna:
        {
            "symbols": ["BTCUSDT", "ETHUSDT", ...],
            "matrix": [[1.0, 0.85, ...], [0.85, 1.0, ...], ...],
            "timestamp": "2026-04-17T12:00:00",
            "interval": "5m",
            "candles": 100
        }
    """
    returns_map = {}
    for sym in symbols:
        closes = _fetch_closes(sym, interval, limit)
        returns_map[sym] = calc_returns(closes)
        time.sleep(0.1)  # Rate limit

    n = len(symbols)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            elif j > i:
                corr = pearson_correlation(
                    returns_map.get(symbols[i], []),
                    returns_map.get(symbols[j], []),
                )
                matrix[i][j] = round(corr, 4)
                matrix[j][i] = round(corr, 4)

    return {
        "symbols": symbols,
        "matrix": matrix,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interval": interval,
        "candles": limit,
    }


def get_high_correlations(matrix_result: Dict, threshold: float = 0.7) -> List[Dict]:
    """Extrae pares con correlacion alta (> threshold) de la matriz."""
    symbols = matrix_result["symbols"]
    matrix = matrix_result["matrix"]
    pairs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            corr = matrix[i][j]
            if abs(corr) >= threshold:
                pairs.append({
                    "pair_a": symbols[i],
                    "pair_b": symbols[j],
                    "correlation": corr,
                    "risk": "high" if corr > 0.85 else "moderate",
                })
    return sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)
