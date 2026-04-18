"""
Watchdog — Mantiene bot y dashboard corriendo de forma autonoma.

Uso:
    py watchdog.py

Monitorea cada 30 segundos:
- Si el bot (scalping_bot.py) no esta corriendo, lo reinicia
- Si el dashboard (dashboard.py) no esta corriendo, lo reinicia
- Registra todos los reinicios en watchdog.log
"""

import subprocess
import time
import os
import sys
import signal
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────
CHECK_INTERVAL = 30        # Segundos entre chequeos
MAX_RESTARTS = 10          # Max reinicios por proceso por hora
STARTUP_DELAY = 5          # Segundos entre arranque de bot y dashboard
WORKDIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

LOG_FILE = os.path.join(WORKDIR, "watchdog.log")
logger = logging.getLogger("watchdog")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(sh)

# ─────────────────────────────────────────────
#  ESTADO
# ─────────────────────────────────────────────
bot_process = None
dash_process = None
restart_history = {"bot": [], "dash": []}


def _clean_old_restarts(name: str):
    """Mantener solo reinicios de la ultima hora."""
    cutoff = time.time() - 3600
    restart_history[name] = [t for t in restart_history[name] if t > cutoff]


def _can_restart(name: str) -> bool:
    _clean_old_restarts(name)
    return len(restart_history[name]) < MAX_RESTARTS


def is_alive(proc) -> bool:
    if proc is None:
        return False
    return proc.poll() is None


def start_bot():
    global bot_process
    if not _can_restart("bot"):
        logger.warning("BOT: Demasiados reinicios en la ultima hora. Pausando.")
        return False
    logger.info("BOT: Iniciando scalping_bot.py ...")
    bot_process = subprocess.Popen(
        [PYTHON, "scalping_bot.py"],
        cwd=WORKDIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    restart_history["bot"].append(time.time())
    logger.info(f"BOT: PID={bot_process.pid}")
    return True


def start_dash():
    global dash_process
    if not _can_restart("dash"):
        logger.warning("DASH: Demasiados reinicios en la ultima hora. Pausando.")
        return False
    logger.info("DASH: Iniciando dashboard.py ...")
    dash_process = subprocess.Popen(
        [PYTHON, "dashboard.py"],
        cwd=WORKDIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    restart_history["dash"].append(time.time())
    logger.info(f"DASH: PID={dash_process.pid}")
    return True


def stop_all():
    global bot_process, dash_process
    for name, proc in [("BOT", bot_process), ("DASH", dash_process)]:
        if proc and is_alive(proc):
            logger.info(f"{name}: Deteniendo PID={proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    bot_process = None
    dash_process = None


def run():
    logger.info("=" * 50)
    logger.info("WATCHDOG INICIADO")
    logger.info(f"  Python: {PYTHON}")
    logger.info(f"  Workdir: {WORKDIR}")
    logger.info(f"  Check interval: {CHECK_INTERVAL}s")
    logger.info("=" * 50)

    # Arranque inicial
    start_bot()
    time.sleep(STARTUP_DELAY)
    start_dash()

    try:
        while True:
            time.sleep(CHECK_INTERVAL)

            # Chequear bot
            if not is_alive(bot_process):
                logger.warning("BOT: Proceso caido. Reiniciando...")
                start_bot()
                time.sleep(STARTUP_DELAY)

            # Chequear dashboard
            if not is_alive(dash_process):
                logger.warning("DASH: Proceso caido. Reiniciando...")
                start_dash()

            # Health check: verificar que el dashboard responde
            try:
                import requests
                r = requests.get("http://localhost:9000/config", timeout=5)
                if r.status_code != 200:
                    logger.warning(f"DASH: Health check fallo (status={r.status_code}). Reiniciando...")
                    if is_alive(dash_process):
                        dash_process.terminate()
                        dash_process.wait(timeout=5)
                    start_dash()
            except Exception:
                if is_alive(dash_process):
                    # El proceso corre pero no responde — darle mas tiempo
                    pass
                else:
                    logger.warning("DASH: No responde y proceso caido. Reiniciando...")
                    start_dash()

    except KeyboardInterrupt:
        logger.info("WATCHDOG: Ctrl+C recibido. Deteniendo todo...")
        stop_all()
        logger.info("WATCHDOG: Finalizado.")


if __name__ == "__main__":
    run()
