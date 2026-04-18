"""
telegram_notifier.py — Notificaciones por Telegram para el bot de trading.

Configuración via variables de entorno:
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=987654321

O via runtime_config.json:
    "telegram_enabled": true,
    "telegram_bot_token": "...",
    "telegram_chat_id": "..."
"""

import json
import os
import threading
import time
from typing import Optional

import requests

RUNTIME_CONFIG_FILE = "runtime_config.json"

_lock = threading.Lock()
_last_sent: dict = {}  # rate limiting por tipo de mensaje
MIN_INTERVAL_SECONDS = 10  # mínimo entre mensajes del mismo tipo


def _get_config() -> dict:
    """Obtiene config de Telegram desde env o runtime_config."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        try:
            with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not token:
                token = str(cfg.get("telegram_bot_token", "")).strip()
            if not chat_id:
                chat_id = str(cfg.get("telegram_chat_id", "")).strip()
        except Exception:
            pass

    return {
        "enabled": bool(token and chat_id),
        "token": token,
        "chat_id": chat_id,
    }


def send_message(text: str, msg_type: str = "general", parse_mode: str = "HTML") -> bool:
    """Envía mensaje a Telegram con rate limiting."""
    cfg = _get_config()
    if not cfg["enabled"]:
        return False

    now = time.time()
    with _lock:
        last = _last_sent.get(msg_type, 0)
        if now - last < MIN_INTERVAL_SECONDS:
            return False
        _last_sent[msg_type] = now

    try:
        url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": cfg["chat_id"], "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def notify_trade(symbol: str, trade_type: str, price: float, pnl: float = 0.0, reason: str = ""):
    """Notifica un trade ejecutado."""
    if trade_type == "BUY":
        emoji = "🟢"
        msg = f"{emoji} <b>COMPRA {symbol}</b>\nPrecio: ${price:,.2f}"
    else:
        emoji = "🔴" if pnl < 0 else "🟢"
        msg = f"{emoji} <b>VENTA {symbol}</b>\nPrecio: ${price:,.2f}\nPnL: ${pnl:+.4f}\nRazón: {reason}"
    send_message(msg, msg_type=f"trade_{symbol}")


def notify_kill_switch(symbol: str, reason: str, detail: str = ""):
    """Notifica activación de kill switch."""
    msg = f"🚨 <b>KILL SWITCH ACTIVO</b>\nSímbolo: {symbol}\nRazón: {reason}"
    if detail:
        msg += f"\nDetalle: {detail}"
    send_message(msg, msg_type="kill_switch")


def notify_summary(total_pnl: float, win_rate: float, trades: int, balance: float):
    """Notifica resumen periódico."""
    emoji = "📈" if total_pnl >= 0 else "📉"
    msg = (
        f"{emoji} <b>RESUMEN</b>\n"
        f"Balance: ${balance:,.2f}\n"
        f"PnL: ${total_pnl:+.2f}\n"
        f"Trades: {trades} | WR: {win_rate:.1f}%"
    )
    send_message(msg, msg_type="summary")


def notify_error(error: str):
    """Notifica un error crítico."""
    msg = f"⚠️ <b>ERROR</b>\n{error[:500]}"
    send_message(msg, msg_type="error")


# ---------------------------------------------------------------------------
# RESÚMENES PROGRAMADOS (diario / semanal)
# ---------------------------------------------------------------------------

_scheduler_started = False
_bot_ref = None  # referencia al objeto bot para obtener datos


def set_bot_reference(bot):
    """Registra referencia al bot para consultar estado en resúmenes."""
    global _bot_ref
    _bot_ref = bot


def _build_daily_summary() -> Optional[str]:
    """Construye resumen diario desde datos del bot."""
    if _bot_ref is None:
        return None
    try:
        wallets = getattr(_bot_ref, "wallets", {})
        if not wallets:
            return None
        lines = ["📊 <b>RESUMEN DIARIO</b>\n"]
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        for sym, w in wallets.items():
            trades = getattr(w, "trades", [])
            day_trades = [t for t in trades if time.time() - t.get("timestamp", 0) < 86400]
            pnl = sum(t.get("pnl", 0) for t in day_trades)
            wins = sum(1 for t in day_trades if t.get("pnl", 0) > 0)
            total_pnl += pnl
            total_trades += len(day_trades)
            total_wins += wins
            if day_trades:
                lines.append(f"  {sym}: {len(day_trades)} trades, PnL ${pnl:+.4f}")
        wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        lines.append(f"\n<b>Total:</b> {total_trades} trades | WR: {wr:.1f}% | PnL: ${total_pnl:+.4f}")
        return "\n".join(lines)
    except Exception:
        return None


def _build_weekly_summary() -> Optional[str]:
    """Construye resumen semanal."""
    if _bot_ref is None:
        return None
    try:
        wallets = getattr(_bot_ref, "wallets", {})
        if not wallets:
            return None
        lines = ["📈 <b>RESUMEN SEMANAL</b>\n"]
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        for sym, w in wallets.items():
            trades = getattr(w, "trades", [])
            week_trades = [t for t in trades if time.time() - t.get("timestamp", 0) < 604800]
            pnl = sum(t.get("pnl", 0) for t in week_trades)
            wins = sum(1 for t in week_trades if t.get("pnl", 0) > 0)
            total_pnl += pnl
            total_trades += len(week_trades)
            total_wins += wins
            if week_trades:
                lines.append(f"  {sym}: {len(week_trades)} trades, PnL ${pnl:+.4f}")
        wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        lines.append(f"\n<b>Total:</b> {total_trades} trades | WR: {wr:.1f}% | PnL: ${total_pnl:+.4f}")
        return "\n".join(lines)
    except Exception:
        return None


def _schedule_daily():
    """Envía resumen diario y reprograma."""
    msg = _build_daily_summary()
    if msg:
        send_message(msg, msg_type="daily_summary")
    # Reprogramar en 24h
    t = threading.Timer(86400, _schedule_daily)
    t.daemon = True
    t.start()


def _schedule_weekly():
    """Envía resumen semanal y reprograma."""
    msg = _build_weekly_summary()
    if msg:
        send_message(msg, msg_type="weekly_summary")
    t = threading.Timer(604800, _schedule_weekly)
    t.daemon = True
    t.start()


def start_scheduled_summaries(bot=None):
    """Inicia los resúmenes programados (llamar una vez al arrancar)."""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True
    if bot:
        set_bot_reference(bot)
    # Primer resumen diario en 24h, semanal en 7d
    td = threading.Timer(86400, _schedule_daily)
    td.daemon = True
    td.start()
    tw = threading.Timer(604800, _schedule_weekly)
    tw.daemon = True
    tw.start()


# ---------------------------------------------------------------------------
# COMANDOS TELEGRAM (polling)
# ---------------------------------------------------------------------------

_command_handlers = {}
_polling_thread = None
_polling_active = False
_last_update_id = 0


def register_command(command: str, handler):
    """Registra un handler para un comando Telegram (ej: /status)."""
    _command_handlers[command.lstrip("/")] = handler


def _default_status_handler() -> str:
    """Handler por defecto para /status."""
    if _bot_ref is None:
        return "Bot no disponible."
    try:
        wallets = getattr(_bot_ref, "wallets", {})
        paused = getattr(_bot_ref, "paused_symbols", set())
        lines = ["<b>Estado del Bot</b>\n"]
        for sym, w in wallets.items():
            bal = getattr(w, "usdt_balance", 0)
            pos = "EN POSICION" if getattr(w, "in_position", False) else "sin posicion"
            status = "PAUSADO" if sym in paused else "activo"
            lines.append(f"{sym}: ${bal:.2f} | {pos} | {status}")
        return "\n".join(lines)
    except Exception:
        return "Error obteniendo estado."


def _default_balance_handler() -> str:
    """Handler por defecto para /balance."""
    if _bot_ref is None:
        return "Bot no disponible."
    try:
        wallets = getattr(_bot_ref, "wallets", {})
        total = 0.0
        lines = ["<b>Balance</b>\n"]
        for sym, w in wallets.items():
            bal = getattr(w, "usdt_balance", 0)
            total += bal
            lines.append(f"{sym}: ${bal:.2f}")
        lines.append(f"\n<b>Total: ${total:.2f}</b>")
        return "\n".join(lines)
    except Exception:
        return "Error obteniendo balance."


def _default_pause_handler(args: str = "") -> str:
    """Handler por defecto para /pause [symbol]."""
    if _bot_ref is None:
        return "Bot no disponible."
    sym = args.strip().upper()
    if not sym:
        return "Uso: /pause BTCUSDT"
    paused = getattr(_bot_ref, "paused_symbols", None)
    if paused is None:
        _bot_ref.paused_symbols = set()
        paused = _bot_ref.paused_symbols
    paused.add(sym)
    return f"⏸ {sym} pausado."


def _default_resume_handler(args: str = "") -> str:
    """Handler por defecto para /resume [symbol]."""
    if _bot_ref is None:
        return "Bot no disponible."
    sym = args.strip().upper()
    if not sym:
        return "Uso: /resume BTCUSDT"
    paused = getattr(_bot_ref, "paused_symbols", set())
    paused.discard(sym)
    return f"▶ {sym} reanudado."


def _poll_updates():
    """Polling loop para recibir comandos Telegram."""
    global _last_update_id, _polling_active
    cfg = _get_config()
    if not cfg["enabled"]:
        return

    while _polling_active:
        try:
            url = f"https://api.telegram.org/bot{cfg['token']}/getUpdates"
            params = {"offset": _last_update_id + 1, "timeout": 30}
            resp = requests.get(url, params=params, timeout=35)
            if resp.status_code != 200:
                time.sleep(5)
                continue
            data = resp.json()
            for update in data.get("result", []):
                _last_update_id = update["update_id"]
                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))
                # Solo responder al chat configurado
                if chat_id != cfg["chat_id"]:
                    continue
                if not text.startswith("/"):
                    continue
                parts = text.split(maxsplit=1)
                cmd = parts[0].lstrip("/").split("@")[0]
                args = parts[1] if len(parts) > 1 else ""
                handler = _command_handlers.get(cmd)
                if handler:
                    try:
                        import inspect
                        sig = inspect.signature(handler)
                        if len(sig.parameters) > 0:
                            reply = handler(args)
                        else:
                            reply = handler()
                    except Exception as e:
                        reply = f"Error: {str(e)[:200]}"
                    if reply:
                        send_message(str(reply), msg_type=f"cmd_{cmd}")
        except Exception:
            time.sleep(5)


def start_command_polling(bot=None):
    """Inicia el polling de comandos Telegram en background."""
    global _polling_thread, _polling_active
    if _polling_thread and _polling_thread.is_alive():
        return
    if bot:
        set_bot_reference(bot)

    # Registrar handlers por defecto
    if "status" not in _command_handlers:
        register_command("status", _default_status_handler)
    if "balance" not in _command_handlers:
        register_command("balance", _default_balance_handler)
    if "pause" not in _command_handlers:
        register_command("pause", _default_pause_handler)
    if "resume" not in _command_handlers:
        register_command("resume", _default_resume_handler)

    _polling_active = True
    _polling_thread = threading.Thread(target=_poll_updates, daemon=True)
    _polling_thread.start()


def stop_command_polling():
    """Detiene el polling de comandos."""
    global _polling_active
    _polling_active = False
