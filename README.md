# 🤖 Bot de Scalping Crypto — Paper & Real Trading con IA

## Descripción del proyecto

Bot de trading automatizado en Python que opera **múltiples pares** en Binance usando una estrategia de **scalping con indicadores técnicos (EMA + RSI + MACD)** y un **filtro de IA** opcional (clasificador binario). Soporta modo **paper trading** (simulado) y **real trading** (órdenes firmadas vía API de Binance).

Incluye **dashboard web en tiempo real** (FastAPI), **notificaciones Telegram**, **backtesting**, **entrenamiento de modelo IA** y **kill switches** de seguridad.

---

## Estructura del proyecto

```
├── scalping_bot.py          # Bot principal (paper + real wallet, estrategia, loop)
├── config_defaults.py       # Config compartida (defaults, normalización)
├── dashboard.py             # Dashboard web (FastAPI, puerto 9000)
├── templates/index.html     # Frontend del dashboard (HTML/CSS/JS)
├── backtest.py              # Backtesting reproducible por par y modo
├── train_model.py           # Entrenamiento offline del modelo IA
├── train_ai_model.py        # Script alternativo de entrenamiento
├── intelligence_engine.py   # Motor IA: scoring, parámetros adaptativos
├── market_scanner.py        # Scanner de mercado multi-par
├── binance_client.py        # Cliente REST de Binance (público)
├── binance_orders.py        # Órdenes firmadas (real trading)
├── binance_websocket.py     # WebSocket de Binance (modo tick)
├── telegram_notifier.py     # Notificaciones Telegram
├── watchdog.py              # Watchdog de procesos
├── auto_optimizer.py        # Optimizador automático de parámetros
├── preflight_real.py        # Verificaciones pre-real trading
├── requirements.txt         # Dependencias Python
├── .env.example             # Variables de entorno de ejemplo
├── levantar_todo.bat        # Script para levantar bot + dashboard
├── tests/                   # 87 tests (pytest)
│   ├── conftest.py
│   ├── test_backtest.py
│   ├── test_dashboard.py
│   ├── test_indicators.py
│   ├── test_intelligence.py
│   ├── test_signal.py
│   ├── test_train_model.py
│   └── test_wallet.py
└── .github/workflows/tests.yml  # CI con GitHub Actions
```

---

## Arquitectura

### Módulos principales

| Módulo | Responsabilidad |
|---|---|
| `scalping_bot.py` | Bot core: BaseWallet → PaperWallet / RealWallet, indicadores, señales, loop |
| `config_defaults.py` | Fuente única de verdad para config runtime y multi-source |
| `dashboard.py` | API REST + frontend para monitoreo y configuración en vivo |
| `intelligence_engine.py` | Scoring IA, ajuste adaptativo de trailing stop / trade size |
| `backtest.py` | Walk-forward backtesting con métricas (Sharpe, drawdown, etc.) |

### Jerarquía de wallets

```
BaseWallet (abstracta)
├── PaperWallet   — simulación sin dinero real
└── RealWallet    — órdenes firmadas vía Binance API
```

---

## Estrategia de trading

### Indicadores usados

- **EMA rápida (9)** + **EMA lenta (21):** Cruce para detectar tendencia
- **RSI (14):** Sobrecompra / sobreventa
- **MACD (12, 26, 9):** Confirmación de momentum
- **Filtro IA (opcional):** Clasificador binario que puntúa probabilidad de retorno positivo

### Reglas de entrada (BUY)

1. EMA(9) cruza por encima de EMA(21)
2. RSI < 65
3. *(Si IA habilitada)* Score IA ≥ umbral de confianza

### Reglas de salida (SELL)

- Cruce EMA bajista, RSI > 65, take profit, stop loss, trailing stop, o kill switch activado

---

## Configuración

Todos los parámetros se gestionan en `config_defaults.py` y pueden modificarse en caliente desde el dashboard (`/config`).

---

## Dependencias

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:
```
requests
python-dotenv
fastapi
uvicorn[standard]
python-binance
```

Para tests: `pip install pytest httpx`

---

## Cómo ejecutar

```bash
# Bot (paper trading por defecto)
python scalping_bot.py

# Dashboard web (puerto 9000)
python dashboard.py

# Ambos (Windows)
levantar_todo.bat

# Tests
python -m pytest tests/ -v

# Backtesting
python backtest.py
```

---

## Dashboard

Accesible en `http://localhost:9000`. Permite:
- Ver estado del bot en tiempo real (balance, posiciones, señales)
- Modificar configuración runtime (indicadores, riesgo, IA)
- Ver análisis de trades y eventos
- Control de kill switches

---

## IA Integrada

| Modo | Comportamiento |
|---|---|
| `off` | Solo estrategia técnica |
| `filter` | BUY solo si regla técnica + score IA ≥ umbral |
| `advisor` | IA ajusta parámetros adaptativos (trailing stop, trade size) |

Entrenamiento: `python train_model.py` sobre los datos de `ml_dataset.csv`.

---

## Tests

87 tests automatizados con pytest:

| Suite | Cobertura |
|---|---|
| `test_wallet.py` | PaperWallet: buy/sell/trailing/total_value |
| `test_indicators.py` | EMA, RSI, MACD |
| `test_signal.py` | Lógica de señales |
| `test_backtest.py` | Backtesting engine |
| `test_intelligence.py` | Motor IA |
| `test_train_model.py` | Pipeline de entrenamiento |
| `test_dashboard.py` | Endpoints del dashboard |

CI: GitHub Actions ejecuta tests automáticamente en cada push/PR.

---

## Seguridad

- **Kill switches:** Pérdida diaria máxima y drawdown máximo configurables
- **Modo real** requiere confirmación explícita + preflight checks (`preflight_real.py`)
- Sin reintentos automáticos en errores de API de órdenes
- Fallback a estrategia técnica si IA falla

