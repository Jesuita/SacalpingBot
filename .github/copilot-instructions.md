# 🤖 Bot de Scalping BTC/USDT — Paper Trading

## Descripción del proyecto

Bot de trading automatizado en Python que opera el par **BTC/USDT** en Binance usando una estrategia de **scalping con indicadores técnicos (EMA + RSI)**. Funciona en modo **paper trading** (simulado, sin dinero real), consumiendo precios reales de la API pública de Binance.

El objetivo es simular una cuenta con **$100 USDT** y operar automáticamente hasta alcanzar un **objetivo de ganancia configurable** (por defecto, $120 USDT).

---

## Estructura del proyecto

```
/
├── scalping_bot.py      # Archivo principal del bot
└── README.md            # Este archivo
```

---

## Archivo principal: `scalping_bot.py`

### Secciones del código

| Sección | Descripción |
|---|---|
| `CONFIGURACIÓN` | Variables globales: par, balance, objetivos, parámetros de riesgo |
| `CLIENTE BINANCE` | Funciones para consumir la API pública de Binance (sin API key) |
| `INDICADORES TÉCNICOS` | Cálculo de EMA y RSI desde cero, sin librerías externas |
| `SEÑALES DE TRADING` | Lógica de decisión: cuándo comprar o vender |
| `PaperWallet` | Clase que simula el balance, ejecuta trades y lleva registro |
| `LOOP PRINCIPAL` | Ciclo de 60 segundos que consulta precios y ejecuta la estrategia |

---

## Estrategia de trading

### Indicadores usados

- **EMA rápida (9 períodos):** Captura movimientos de corto plazo
- **EMA lenta (21 períodos):** Referencia de tendencia
- **RSI (14 períodos):** Mide sobrecompra/sobreventa

### Reglas de entrada (BUY)

Se abre una posición cuando se cumplen **ambas condiciones**:
1. La EMA(9) cruza **por encima** de la EMA(21) → señal de impulso alcista
2. El RSI está **por debajo de 65** → no está sobrecomprado

### Reglas de salida (SELL)

Se cierra la posición si se da **alguna** de estas condiciones:
- La EMA(9) cruza **por debajo** de la EMA(21) → reversión bajista
- El RSI supera **65** → sobrecompra
- Se alcanza el **take profit (+0.8%)**
- Se alcanza el **stop loss (-0.5%)**
- Se alcanza el **objetivo global de balance**

---

## Variables de configuración

Todas se encuentran al inicio de `scalping_bot.py` bajo el bloque `CONFIGURACIÓN`:

```python
SYMBOL           = "BTCUSDT"   # Par a operar
INTERVAL         = "1m"        # Temporalidad de las velas
PAPER_BALANCE    = 100.0       # Saldo inicial simulado en USDT
TARGET_USDT      = 120.0       # Objetivo de ganancia para detener el bot
STOP_LOSS_PCT    = 0.005       # Stop loss por operación (0.5%)
TAKE_PROFIT_PCT  = 0.008       # Take profit por operación (0.8%)
TRADE_PCT        = 0.95        # Porcentaje del balance usado por trade

EMA_FAST         = 9           # Período EMA rápida
EMA_SLOW         = 21          # Período EMA lenta
RSI_PERIOD       = 14          # Período RSI
RSI_OVERSOLD     = 35          # Nivel de sobreventa
RSI_OVERBOUGHT   = 65          # Nivel de sobrecompra
```

---

## Dependencias

Solo necesita la librería estándar de Python más `requests`:

```bash
pip install requests
```

No requiere `pandas`, `numpy`, ni ninguna librería de trading. Todo el cálculo de indicadores está implementado manualmente.

---

## Cómo ejecutar

```bash
python scalping_bot.py
```

Para detener el bot manualmente: `Ctrl + C`

Al finalizar (por objetivo alcanzado o interrupción), imprime un **resumen** con:
- Total de trades ejecutados
- Win rate
- Balance final
- Ganancia/pérdida total

---

## API de Binance

En modo paper trading, el bot solo usa **endpoints públicos** (sin autenticación):

| Endpoint | Uso |
|---|---|
| `GET /api/v3/klines` | Obtener historial de velas OHLCV |
| `GET /api/v3/ticker/price` | Obtener precio actual del par |

Las variables `API_KEY` y `API_SECRET` están definidas pero **no se usan** en este modo.

---

## Clase `PaperWallet`

Simula una billetera con USDT y BTC:

| Método | Descripción |
|---|---|
| `buy(price)` | Compra BTC con el 95% del USDT disponible |
| `sell(price, reason)` | Vende todo el BTC al precio actual |
| `total_value(price)` | Calcula el valor total de la cartera (USDT + BTC valorizado) |
| `summary(price)` | Imprime el resumen final de la sesión |
| `in_position` | Propiedad booleana: indica si hay una posición abierta |

---

## Posibles mejoras futuras

- [ ] Pasar a modo real agregando firma HMAC a las órdenes POST de Binance
- [x] Agregar logging a archivo `.log` para análisis posterior
- [x] Soporte para múltiples pares simultáneos
- [x] Notificaciones por Telegram al ejecutar cada trade
- [x] Backtesting sobre datos históricos antes de operar en vivo
- [x] Dashboard web en tiempo real con Flask o FastAPI

---

## Notas para el agente de IA

- El bot está diseñado para ser **modificado y extendido**. Cada sección está claramente delimitada con comentarios.
- La lógica de señales está aislada en la función `get_signal()` — es el lugar indicado para cambiar o agregar indicadores.
- Para agregar un nuevo indicador: implementarlo como función independiente y llamarlo dentro de `get_signal()`.
- El `PaperWallet` puede reemplazarse por una clase `RealWallet` que haga llamadas firmadas a la API de Binance, manteniendo la misma interfaz (`buy`, `sell`, `in_position`).
- El loop principal en `run()` no debe modificarse salvo para cambiar el intervalo de polling (`time.sleep(60)`).

---

## Plan IA Híbrida (Contexto Activo)

Este plan define como dotar al bot de "inteligencia" sin perder control de riesgo.
La estrategia base (EMA + RSI + MACD) sigue siendo el motor principal y la IA actua como filtro y optimizador.

### Fase 0 - Baseline y observabilidad (hecho)

- [x] Bot paper y real con wallet abstraida.
- [x] Dashboard en tiempo real.
- [x] Logging de trades (`trades.log`) y eventos (`bot_events.log`).
- [x] Configuracion runtime desde UI.
- [x] Modo candle y modo tick configurable.

### Fase 1 - Dataset para entrenamiento (hecho)

- [x] Guardar features por decision en `ml_dataset.csv`.
- [x] Features minimas:
	- timestamp, symbol, price
	- ema_fast, ema_slow, rsi, macd, macd_signal, macd_hist
	- signal_regla, in_position, balance
	- future_return_n (label offline)
- [x] Registrar cada evaluacion en candle y tick.
- [x] Agregar endpoint de dashboard para resumen de dataset.

### Fase 2 - IA V1 (filtro de señal) (hecho)

- [x] Entrenamiento offline (script separado) con clasificador binario.
- [x] Objetivo: probabilidad de retorno positivo en horizonte corto.
- [x] Integracion online:
	- `ai_enabled` (bool)
	- `ai_min_confidence` (umbral)
	- `ai_mode` (`off`, `filter`, `advisor`)
- [x] Regla final:
	- BUY solo si regla tecnica BUY y score IA >= umbral.
	- SELL se mantiene conservador (riesgo primero).

### Fase 3 - IA V2 (parametros adaptativos) (hecho)

- [x] Ajustar dinamicamente `trailing_stop_pct` segun volatilidad (ATR adaptativo via intelligence_engine).
- [x] Ajustar dinamicamente `trade_pct`, `take_profit_pct` segun score IA.
- [x] Limites duros para no sobreexponer riesgo.

### Fase 4 - Validacion fuerte antes de real (hecho)

- [x] Backtesting reproducible por par y modo (`backtest.py`).
- [x] Walk-forward por ventanas temporales.
- [x] Metricas objetivo:
	- win rate
	- profit factor
	- max drawdown
	- expectancy por trade
	- Sharpe ratio
- [x] Kill switch por perdida diaria y drawdown maximo.
- [x] Notificaciones Telegram en trades y kill switches.

### Reglas de seguridad permanentes

- Nunca operar en real sin confirmacion explicita de usuario.
- Sin reintentos automaticos ante error de API de orden.
- Riesgo por trade limitado por balance real disponible.
- Si IA falla o no responde: fallback a estrategia tecnica con modo seguro.

### Protocolo de ejecucion en este repo

1. Ejecutar fase por fase, sin saltos.
2. Cada fase debe terminar con:
	 - codigo funcionando
	 - validacion de sintaxis
	 - cambio visible en dashboard
3. No mover logica central sin mantener compatibilidad con PaperWallet/RealWallet.
4. Actualizar esta seccion al cerrar cada fase para mantener contexto historico.
