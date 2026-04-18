"""
backtest.py — Motor de backtesting reproducible para el scalping bot.

Usa datos históricos de Binance y aplica la misma lógica de señales
que scalping_bot.py (EMA + RSI + MACD).

Uso:
    py backtest.py                         # Defaults: BTCUSDT, 1000 velas
    py backtest.py --symbol ETHUSDT --limit 2000
    py backtest.py --symbols BTCUSDT,ETHUSDT --trailing 0.006 --tp 0.008
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

# Reusar indicadores y señales del bot
from scalping_bot import (
    calc_ema,
    calc_rsi,
    calc_macd,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)

BASE_URL = "https://api.binance.com"


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> List[dict]:
    """Descarga velas históricas de Binance. Máximo 1000 por request."""
    all_klines = []
    remaining = limit
    end_time = None

    while remaining > 0:
        batch = min(remaining, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if end_time:
            params["endTime"] = end_time - 1

        resp = requests.get(f"{BASE_URL}/api/v3/klines", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        all_klines = data + all_klines
        end_time = data[0][0]  # open_time del primer candle
        remaining -= len(data)

        if len(data) < batch:
            break

    return all_klines


def parse_klines(raw: list) -> List[dict]:
    """Convierte klines crudas a dicts con open/high/low/close/volume."""
    candles = []
    for k in raw:
        candles.append({
            "timestamp": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return candles


# ─────────────────────────────────────────────
#  SIGNAL (idéntica a scalping_bot.get_signal)
# ─────────────────────────────────────────────

def get_signal(closes: list) -> str:
    min_len = max(EMA_SLOW + 5, MACD_SLOW + MACD_SIGNAL + 2)
    if len(closes) < min_len:
        return "HOLD"

    ema_fast_now = calc_ema(closes, EMA_FAST)
    ema_slow_now = calc_ema(closes, EMA_SLOW)
    ema_fast_prev = calc_ema(closes[:-1], EMA_FAST)
    ema_slow_prev = calc_ema(closes[:-1], EMA_SLOW)
    rsi = calc_rsi(closes, RSI_PERIOD)
    macd_now, macd_signal_now, macd_hist_now = calc_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_prev, macd_signal_prev, macd_hist_prev = calc_macd(closes[:-1], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
    bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now
    macd_positive = macd_now > macd_signal_now
    macd_hist_rising = macd_hist_now >= macd_hist_prev
    trend_up = ema_fast_now > ema_slow_now
    macd_negative_cross = macd_prev >= macd_signal_prev and macd_now < macd_signal_now

    continuation_buy = trend_up and macd_positive and macd_hist_rising and (RSI_OVERSOLD < rsi < 68)

    if (bullish_cross and rsi < RSI_OVERBOUGHT and macd_positive) or continuation_buy:
        return "BUY"
    if bearish_cross or rsi > RSI_OVERBOUGHT or macd_negative_cross:
        return "SELL"
    return "HOLD"


# ─────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────

@dataclass
class Trade:
    entry_price: float
    entry_time: int
    exit_price: float = 0.0
    exit_time: int = 0
    qty: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    symbol: str
    total_candles: int
    initial_balance: float
    final_balance: float
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


def run_backtest(
    candles: List[dict],
    symbol: str = "BTCUSDT",
    initial_balance: float = 100.0,
    trade_pct: float = 0.95,
    trailing_stop_pct: float = 0.006,
    take_profit_pct: float = 0.008,
    stop_loss_pct: float = 0.005,
    fee_pct: float = 0.001,
) -> BacktestResult:
    """Ejecuta backtest vela por vela con la estrategia EMA+RSI+MACD."""

    balance = initial_balance
    in_position = False
    entry_price = 0.0
    peak_price = 0.0
    trailing_stop_price = 0.0
    qty = 0.0
    entry_time = 0
    trades: List[Trade] = []

    # Equity tracking para drawdown
    peak_balance = initial_balance
    max_dd = 0.0

    # Returns por trade para Sharpe
    trade_returns: List[float] = []

    # Equity curve (valor de cartera en cada vela)
    equity_curve: List[float] = []

    closes: List[float] = []

    for candle in candles:
        price = candle["close"]
        closes.append(price)

        signal = get_signal(closes)

        # --- Gestión de posición abierta ---
        if in_position:
            # Update trailing
            if price > peak_price:
                peak_price = price
                new_stop = peak_price * (1 - trailing_stop_pct)
                if new_stop > trailing_stop_price:
                    trailing_stop_price = new_stop

            # Check exits
            pnl_pct = (price - entry_price) / entry_price
            reason = ""

            if price <= trailing_stop_price:
                reason = "trailing_stop"
            elif pnl_pct >= take_profit_pct:
                reason = "take_profit"
            elif pnl_pct <= -stop_loss_pct:
                reason = "stop_loss"
            elif signal == "SELL":
                reason = "signal_sell"

            if reason:
                gross_proceeds = qty * price
                proceeds = gross_proceeds * (1 - fee_pct)
                cost_basis = qty * entry_price
                pnl = proceeds - cost_basis
                balance += proceeds
                trade = Trade(
                    entry_price=entry_price,
                    entry_time=entry_time,
                    exit_price=price,
                    exit_time=candle["timestamp"],
                    qty=qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason,
                )
                trades.append(trade)
                trade_returns.append(pnl_pct)

                in_position = False
                qty = 0.0
                entry_price = 0.0

        # --- Entrada ---
        elif signal == "BUY" and balance > 0:
            amount_usdt = balance * trade_pct
            raw_qty = amount_usdt / price
            qty = raw_qty * (1 - fee_pct)  # Descontar comisión de compra
            balance -= amount_usdt
            entry_price = price
            peak_price = price
            trailing_stop_price = price * (1 - trailing_stop_pct)
            entry_time = candle["timestamp"]
            in_position = True

        # --- Drawdown tracking ---
        current_equity = balance + (qty * price if in_position else 0)
        if current_equity > peak_balance:
            peak_balance = current_equity
        dd = (peak_balance - current_equity) / peak_balance
        if dd > max_dd:
            max_dd = dd
        equity_curve.append(current_equity)

    # Cerrar posición abierta al final
    if in_position:
        final_price = closes[-1]
        gross_proceeds = qty * final_price
        proceeds = gross_proceeds * (1 - fee_pct)
        pnl = proceeds - (qty * entry_price)
        pnl_pct = (final_price - entry_price) / entry_price
        balance += proceeds
        trades.append(Trade(
            entry_price=entry_price,
            entry_time=entry_time,
            exit_price=final_price,
            exit_time=candles[-1]["timestamp"],
            qty=qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason="end_of_data",
        ))
        trade_returns.append(pnl_pct)

    # --- Métricas ---
    total_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses_list = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses_list))

    result = BacktestResult(
        symbol=symbol,
        total_candles=len(candles),
        initial_balance=initial_balance,
        final_balance=round(balance, 4),
        total_trades=total_trades,
        wins=len(wins),
        losses=len(losses_list),
        win_rate=round(len(wins) / max(total_trades, 1) * 100, 2),
        total_pnl=round(balance - initial_balance, 4),
        total_pnl_pct=round((balance - initial_balance) / initial_balance * 100, 2),
        profit_factor=round(gross_profit / max(gross_loss, 0.0001), 3),
        max_drawdown=round(max_dd * initial_balance, 4),
        max_drawdown_pct=round(max_dd * 100, 2),
        expectancy=round(sum(t.pnl for t in trades) / max(total_trades, 1), 4),
        avg_win=round(gross_profit / max(len(wins), 1), 4),
        avg_loss=round(-gross_loss / max(len(losses_list), 1), 4),
        trades=trades,
        equity_curve=equity_curve,
    )

    # Sharpe ratio (asumiendo risk-free=0)
    if len(trade_returns) >= 2:
        mean_r = sum(trade_returns) / len(trade_returns)
        var_r = sum((r - mean_r) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
        std_r = var_r ** 0.5
        result.sharpe_ratio = round(mean_r / max(std_r, 1e-9), 3)

    return result


# ─────────────────────────────────────────────
#  WALK-FORWARD
# ─────────────────────────────────────────────

def walk_forward(
    candles: List[dict],
    symbol: str = "BTCUSDT",
    window_size: int = 500,
    step: int = 250,
    **kwargs,
) -> List[BacktestResult]:
    """Ejecuta backtests en ventanas deslizantes (walk-forward)."""
    results = []
    i = 0
    while i + window_size <= len(candles):
        window = candles[i : i + window_size]
        r = run_backtest(window, symbol=symbol, **kwargs)
        results.append(r)
        i += step

    # Última ventana si queda data
    if i < len(candles) and len(candles) - i >= 50:
        window = candles[i:]
        r = run_backtest(window, symbol=symbol, **kwargs)
        results.append(r)

    return results


# ─────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────

def print_result(r: BacktestResult, verbose: bool = False):
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST: {r.symbol} | {r.total_candles} velas")
    print(f"{'=' * 60}")
    print(f"  Balance inicial : ${r.initial_balance:.2f}")
    print(f"  Balance final   : ${r.final_balance:.2f}")
    print(f"  PnL total       : ${r.total_pnl:+.2f} ({r.total_pnl_pct:+.2f}%)")
    print(f"  Trades          : {r.total_trades}")
    print(f"  Win rate        : {r.win_rate:.1f}%")
    print(f"  Profit factor   : {r.profit_factor:.3f}")
    print(f"  Expectancy      : ${r.expectancy:+.4f}")
    print(f"  Avg win         : ${r.avg_win:+.4f}")
    print(f"  Avg loss        : ${r.avg_loss:+.4f}")
    print(f"  Max drawdown    : ${r.max_drawdown:.4f} ({r.max_drawdown_pct:.2f}%)")
    print(f"  Sharpe ratio    : {r.sharpe_ratio:.3f}")
    print(f"{'=' * 60}")

    if verbose and r.trades:
        print(f"\n  {'#':>3}  {'Entrada':>10}  {'Salida':>10}  {'PnL':>10}  {'%':>8}  Razón")
        print(f"  {'-' * 60}")
        for i, t in enumerate(r.trades, 1):
            print(f"  {i:3d}  ${t.entry_price:>9.2f}  ${t.exit_price:>9.2f}  ${t.pnl:>+9.4f}  {t.pnl_pct:>+7.3f}%  {t.reason}")


def print_walk_forward_summary(results: List[BacktestResult]):
    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD ANALYSIS | {len(results)} ventanas")
    print(f"{'=' * 60}")
    if not results:
        print("  Sin resultados")
        return

    pnls = [r.total_pnl for r in results]
    wrs = [r.win_rate for r in results]
    pfs = [r.profit_factor for r in results]
    dds = [r.max_drawdown_pct for r in results]

    profitable_windows = sum(1 for p in pnls if p > 0)
    print(f"  Ventanas rentables : {profitable_windows}/{len(results)} ({profitable_windows / len(results) * 100:.0f}%)")
    print(f"  PnL promedio       : ${sum(pnls) / len(pnls):+.4f}")
    print(f"  PnL total          : ${sum(pnls):+.4f}")
    print(f"  Win rate promedio  : {sum(wrs) / len(wrs):.1f}%")
    print(f"  PF promedio        : {sum(pfs) / len(pfs):.3f}")
    print(f"  Max DD promedio    : {sum(dds) / len(dds):.2f}%")
    print(f"  Peor DD            : {max(dds):.2f}%")
    print(f"{'=' * 60}")


def save_result_json(r: BacktestResult, filepath: str = "backtest_result.json"):
    """Guarda resultado en JSON para análisis posterior."""
    data = {
        "symbol": r.symbol,
        "total_candles": r.total_candles,
        "initial_balance": r.initial_balance,
        "final_balance": r.final_balance,
        "total_trades": r.total_trades,
        "wins": r.wins,
        "losses": r.losses,
        "win_rate": r.win_rate,
        "total_pnl": r.total_pnl,
        "total_pnl_pct": r.total_pnl_pct,
        "profit_factor": r.profit_factor,
        "max_drawdown": r.max_drawdown,
        "max_drawdown_pct": r.max_drawdown_pct,
        "expectancy": r.expectancy,
        "sharpe_ratio": r.sharpe_ratio,
        "trades": [
            {
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
            }
            for t in r.trades
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nResultado guardado en {filepath}")


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────

def generate_charts(r: BacktestResult, output_dir: str = "."):
    """Genera gráficos de equity curve, drawdown y distribución de PnL."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado — saltando gráficos. pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(f"Backtest: {r.symbol} | {r.total_candles} velas | PnL: ${r.total_pnl:+.2f} ({r.total_pnl_pct:+.1f}%)", fontsize=13)

    # 1. Equity Curve
    ax1 = axes[0]
    if r.equity_curve:
        ax1.plot(r.equity_curve, linewidth=0.8, color="#2196F3")
        ax1.axhline(y=r.initial_balance, linestyle="--", color="gray", alpha=0.6, linewidth=0.7)
        # Marcar trades
        for t in r.trades:
            if t.entry_time and t.exit_time:
                idx_exit = min(t.exit_time, len(r.equity_curve) - 1) if isinstance(t.exit_time, int) and t.exit_time < len(r.equity_curve) else None
        ax1.fill_between(range(len(r.equity_curve)), r.initial_balance, r.equity_curve,
                         where=[e >= r.initial_balance for e in r.equity_curve], alpha=0.15, color="green")
        ax1.fill_between(range(len(r.equity_curve)), r.initial_balance, r.equity_curve,
                         where=[e < r.initial_balance for e in r.equity_curve], alpha=0.15, color="red")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    if r.equity_curve:
        peak = r.equity_curve[0]
        drawdowns = []
        for eq in r.equity_curve:
            if eq > peak:
                peak = eq
            dd_pct = (peak - eq) / peak * 100
            drawdowns.append(-dd_pct)
        ax2.fill_between(range(len(drawdowns)), 0, drawdowns, color="#F44336", alpha=0.4)
        ax2.plot(drawdowns, linewidth=0.6, color="#D32F2F")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title(f"Drawdown (máx: {r.max_drawdown_pct:.2f}%)")
    ax2.grid(True, alpha=0.3)

    # 3. PnL Distribution
    ax3 = axes[2]
    if r.trades:
        pnls = [t.pnl for t in r.trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, width=0.8)
        ax3.axhline(y=0, color="gray", linewidth=0.5)
        if r.expectancy != 0:
            ax3.axhline(y=r.expectancy, linestyle="--", color="#FF9800", alpha=0.7, linewidth=0.7, label=f"Expectancy: ${r.expectancy:+.4f}")
            ax3.legend(fontsize=8)
    ax3.set_ylabel("PnL ($)")
    ax3.set_xlabel("Trade #")
    ax3.set_title(f"PnL por Trade | WR: {r.win_rate:.1f}% | PF: {r.profit_factor:.2f} | Sharpe: {r.sharpe_ratio:.3f}")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"backtest_{r.symbol}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado en {filepath}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtester para estrategia EMA+RSI+MACD")
    parser.add_argument("--symbol", default="BTCUSDT", help="Par a testear (default: BTCUSDT)")
    parser.add_argument("--symbols", default=None, help="Múltiples pares separados por coma")
    parser.add_argument("--limit", type=int, default=1000, help="Número de velas (default: 1000)")
    parser.add_argument("--interval", default="1m", help="Temporalidad (default: 1m)")
    parser.add_argument("--balance", type=float, default=100.0, help="Balance inicial (default: 100)")
    parser.add_argument("--trailing", type=float, default=0.006, help="Trailing stop (default: 0.006)")
    parser.add_argument("--tp", type=float, default=0.008, help="Take profit (default: 0.008)")
    parser.add_argument("--sl", type=float, default=0.005, help="Stop loss (default: 0.005)")
    parser.add_argument("--trade-pct", type=float, default=0.95, help="Fraccion balance por trade (default: 0.95)")
    parser.add_argument("--walk-forward", action="store_true", help="Ejecutar walk-forward analysis")
    parser.add_argument("--window", type=int, default=500, help="Tamaño ventana walk-forward (default: 500)")
    parser.add_argument("--step", type=int, default=250, help="Paso walk-forward (default: 250)")
    parser.add_argument("--verbose", action="store_true", help="Mostrar trades individuales")
    parser.add_argument("--save", action="store_true", help="Guardar resultado en JSON")
    parser.add_argument("--chart", action="store_true", help="Generar gráficos PNG")
    parser.add_argument("--fee", type=float, default=0.001, help="Comisión por trade (default: 0.001 = 0.1%%)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else [args.symbol]

    bt_kwargs = {
        "initial_balance": args.balance,
        "trade_pct": args.trade_pct,
        "trailing_stop_pct": args.trailing,
        "take_profit_pct": args.tp,
        "stop_loss_pct": args.sl,
        "fee_pct": args.fee,
    }

    for sym in symbols:
        sym = sym.strip().upper()
        print(f"\nDescargando {args.limit} velas de {sym} ({args.interval})...")
        raw = fetch_klines(sym, args.interval, args.limit)
        candles = parse_klines(raw)
        print(f"  Recibidas: {len(candles)} velas")

        if args.walk_forward:
            results = walk_forward(
                candles, symbol=sym,
                window_size=args.window, step=args.step,
                **bt_kwargs,
            )
            for i, r in enumerate(results, 1):
                print(f"\n  --- Ventana {i} ({r.total_candles} velas) ---")
                print_result(r, verbose=args.verbose)
            print_walk_forward_summary(results)
            if args.chart and results:
                generate_charts(results[-1])
        else:
            result = run_backtest(candles, symbol=sym, **bt_kwargs)
            print_result(result, verbose=args.verbose)

            if args.save:
                save_result_json(result, f"backtest_{sym}.json")
            if args.chart:
                generate_charts(result)


if __name__ == "__main__":
    main()
