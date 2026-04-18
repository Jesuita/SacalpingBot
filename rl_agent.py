"""
rl_agent.py -- Agente de Reinforcement Learning (Q-Learning tabular) para el bot.

Aprende politica de trading (BUY/HOLD/SELL) a partir de recompensas reales.
Discretiza el estado del mercado en bins y mantiene una Q-table en JSON.

No requiere numpy, pandas ni librerias externas. Todo manual.

Uso:
    import rl_agent
    agent = rl_agent.RLAgent()          # carga Q-table si existe
    action = agent.choose_action(state)  # 0=HOLD, 1=BUY, 2=SELL
    agent.update(state, action, reward, next_state)
    agent.save()
"""

import json
import math
import os
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

RL_MODEL_FILE = "rl_model.json"

# ---------------------------------------------------------------------------
#  DISCRETIZACION DEL ESTADO
# ---------------------------------------------------------------------------

# Bins para cada feature del estado
# ema_diff_pct: (ema_fast - ema_slow) / price * 100
EMA_DIFF_BINS = [-0.3, -0.1, 0.0, 0.1, 0.3]  # 6 bins

# rsi: 0-100
RSI_BINS = [30, 40, 50, 60, 70]  # 6 bins

# macd_hist: normalizado
MACD_BINS = [-0.5, -0.1, 0.0, 0.1, 0.5]  # 6 bins

# regime: trending_up=0, trending_down=1, ranging=2, volatile=3
REGIME_MAP = {"trending_up": 0, "trending_down": 1, "ranging": 2, "volatile": 3}

# position: 0=sin posicion, 1=en posicion
# hour_bucket: 0-5 (4h buckets: 0-3, 4-7, 8-11, 12-15, 16-19, 20-23)

ACTIONS = ["HOLD", "BUY", "SELL"]
N_ACTIONS = len(ACTIONS)


def _digitize(value: float, bins: list) -> int:
    """Discretiza un valor continuo en un bin index."""
    for i, edge in enumerate(bins):
        if value < edge:
            return i
    return len(bins)


def discretize_state(
    ema_diff_pct: float,
    rsi: float,
    macd_hist: float,
    regime: str,
    in_position: bool,
    hour: int,
) -> str:
    """Convierte features continuas en un estado discreto (string key)."""
    e = _digitize(ema_diff_pct, EMA_DIFF_BINS)
    r = _digitize(rsi, RSI_BINS)
    m = _digitize(macd_hist, MACD_BINS)
    reg = REGIME_MAP.get(regime, 2)
    pos = 1 if in_position else 0
    hb = min(hour // 4, 5)
    return f"{e}_{r}_{m}_{reg}_{pos}_{hb}"


# ---------------------------------------------------------------------------
#  AGENTE Q-LEARNING
# ---------------------------------------------------------------------------

class RLAgent:
    """Agente Q-Learning tabular para trading."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        model_file: str = RL_MODEL_FILE,
    ):
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model_file = model_file
        self.q_table: Dict[str, List[float]] = {}
        self.stats = {
            "total_updates": 0,
            "total_explorations": 0,
            "total_exploitations": 0,
            "states_visited": 0,
            "avg_reward": 0.0,
            "reward_history": [],
        }
        self._lock = threading.Lock()
        self.load()

    def _get_q(self, state: str) -> List[float]:
        """Obtiene Q-values para un estado, inicializando si no existe."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * N_ACTIONS
        return self.q_table[state]

    def choose_action(self, state: str) -> int:
        """Epsilon-greedy action selection."""
        with self._lock:
            if random.random() < self.epsilon:
                self.stats["total_explorations"] += 1
                return random.randint(0, N_ACTIONS - 1)
            else:
                self.stats["total_exploitations"] += 1
                q_values = self._get_q(state)
                max_q = max(q_values)
                # Romper empates aleatoriamente
                best = [i for i, q in enumerate(q_values) if q == max_q]
                return random.choice(best)

    def choose_action_greedy(self, state: str) -> int:
        """Greedy action (sin exploracion, para produccion)."""
        with self._lock:
            q_values = self._get_q(state)
            max_q = max(q_values)
            best = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best)

    def update(self, state: str, action: int, reward: float, next_state: str):
        """Q-Learning update: Q(s,a) = Q(s,a) + alpha * (r + gamma * max Q(s') - Q(s,a))"""
        with self._lock:
            q_sa = self._get_q(state)
            q_next = self._get_q(next_state)
            old_val = q_sa[action]
            best_next = max(q_next)
            new_val = old_val + self.alpha * (reward + self.gamma * best_next - old_val)
            q_sa[action] = new_val

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Stats
            self.stats["total_updates"] += 1
            self.stats["states_visited"] = len(self.q_table)
            self.stats["reward_history"].append(reward)
            # Mantener solo ultimas 1000 recompensas
            if len(self.stats["reward_history"]) > 1000:
                self.stats["reward_history"] = self.stats["reward_history"][-1000:]
            if self.stats["reward_history"]:
                self.stats["avg_reward"] = sum(self.stats["reward_history"]) / len(
                    self.stats["reward_history"]
                )

    def get_action_name(self, action: int) -> str:
        """Convierte indice a nombre de accion."""
        if 0 <= action < N_ACTIONS:
            return ACTIONS[action]
        return "HOLD"

    def get_q_summary(self, state: str) -> Dict:
        """Resumen de Q-values para un estado."""
        with self._lock:
            q_values = self._get_q(state)
            return {
                "state": state,
                "q_values": {ACTIONS[i]: round(q_values[i], 6) for i in range(N_ACTIONS)},
                "best_action": ACTIONS[q_values.index(max(q_values))],
                "epsilon": round(self.epsilon, 4),
            }

    def get_stats(self) -> Dict:
        """Estadisticas del agente."""
        with self._lock:
            return {
                "total_updates": self.stats["total_updates"],
                "total_explorations": self.stats["total_explorations"],
                "total_exploitations": self.stats["total_exploitations"],
                "states_visited": self.stats["states_visited"],
                "epsilon": round(self.epsilon, 4),
                "alpha": self.alpha,
                "gamma": self.gamma,
                "avg_reward": round(self.stats["avg_reward"], 6),
                "q_table_size": len(self.q_table),
            }

    def save(self):
        """Guarda Q-table y stats a JSON."""
        with self._lock:
            data = {
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "stats": {
                    "total_updates": self.stats["total_updates"],
                    "total_explorations": self.stats["total_explorations"],
                    "total_exploitations": self.stats["total_exploitations"],
                    "states_visited": self.stats["states_visited"],
                    "avg_reward": self.stats["avg_reward"],
                },
            }
        try:
            with open(self.model_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load(self):
        """Carga Q-table desde JSON si existe."""
        if not os.path.exists(self.model_file):
            return
        try:
            with open(self.model_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self.q_table = data.get("q_table", {})
                self.epsilon = data.get("epsilon", self.epsilon)
                saved_stats = data.get("stats", {})
                self.stats["total_updates"] = saved_stats.get("total_updates", 0)
                self.stats["total_explorations"] = saved_stats.get("total_explorations", 0)
                self.stats["total_exploitations"] = saved_stats.get("total_exploitations", 0)
                self.stats["states_visited"] = len(self.q_table)
                self.stats["avg_reward"] = saved_stats.get("avg_reward", 0.0)
        except Exception:
            pass

    def reset(self):
        """Resetea la Q-table y stats."""
        with self._lock:
            self.q_table = {}
            self.epsilon = 0.15
            self.stats = {
                "total_updates": 0,
                "total_explorations": 0,
                "total_exploitations": 0,
                "states_visited": 0,
                "avg_reward": 0.0,
                "reward_history": [],
            }


# ---------------------------------------------------------------------------
#  REWARD CALCULATOR
# ---------------------------------------------------------------------------

def calculate_reward(pnl_pct: float, action: int, was_correct: bool) -> float:
    """
    Calcula recompensa para el agente RL.

    pnl_pct: retorno porcentual del trade (0.005 = 0.5%)
    action: accion tomada (0=HOLD, 1=BUY, 2=SELL)
    was_correct: si la decision fue correcta post-facto
    """
    # Recompensa base proporcional al PnL
    reward = pnl_pct * 100  # escalar para mejor convergencia

    # Penalizar overtrading (BUY/SELL innecesarios)
    if not was_correct:
        if action == 1:  # BUY malo
            reward -= 0.5
        elif action == 2:  # SELL prematuro
            reward -= 0.3

    # Bonus por HOLD correcto (no entrar en trade perdedor)
    if action == 0 and was_correct:
        reward += 0.1

    return reward


# ---------------------------------------------------------------------------
#  SINGLETON GLOBAL
# ---------------------------------------------------------------------------

_agent: Optional[RLAgent] = None
_agent_lock = threading.Lock()


def get_agent() -> RLAgent:
    """Obtiene o crea el agente singleton."""
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = RLAgent()
        return _agent


def get_rl_suggestion(
    ema_diff_pct: float,
    rsi: float,
    macd_hist: float,
    regime: str,
    in_position: bool,
    hour: int,
    greedy: bool = True,
) -> Dict:
    """
    Obtiene sugerencia del agente RL.

    Retorna: {"action": "BUY/HOLD/SELL", "q_values": {...}, "state": "..."}
    """
    agent = get_agent()
    state = discretize_state(ema_diff_pct, rsi, macd_hist, regime, in_position, hour)
    if greedy:
        action = agent.choose_action_greedy(state)
    else:
        action = agent.choose_action(state)
    summary = agent.get_q_summary(state)
    summary["action"] = agent.get_action_name(action)
    summary["action_idx"] = action
    return summary


def record_experience(
    ema_diff_pct: float,
    rsi: float,
    macd_hist: float,
    regime: str,
    in_position: bool,
    hour: int,
    action: int,
    reward: float,
    next_ema_diff_pct: float,
    next_rsi: float,
    next_macd_hist: float,
    next_regime: str,
    next_in_position: bool,
    next_hour: int,
):
    """Registra experiencia y actualiza Q-table."""
    agent = get_agent()
    state = discretize_state(ema_diff_pct, rsi, macd_hist, regime, in_position, hour)
    next_state = discretize_state(
        next_ema_diff_pct, next_rsi, next_macd_hist, next_regime, next_in_position, next_hour
    )
    agent.update(state, action, reward, next_state)
    # Auto-save cada 100 updates
    if agent.stats["total_updates"] % 100 == 0:
        agent.save()
