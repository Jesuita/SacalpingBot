"""Tests para rl_agent.py -- Q-Learning tabular."""

import json
import os
import tempfile
import unittest

import rl_agent


class TestDiscretize(unittest.TestCase):
    def test_digitize_basic(self):
        self.assertEqual(rl_agent._digitize(-1.0, [0, 1, 2]), 0)
        self.assertEqual(rl_agent._digitize(0.5, [0, 1, 2]), 1)
        self.assertEqual(rl_agent._digitize(5.0, [0, 1, 2]), 3)

    def test_discretize_state_format(self):
        s = rl_agent.discretize_state(0.0, 50.0, 0.0, "ranging", False, 12)
        parts = s.split("_")
        self.assertEqual(len(parts), 6)

    def test_different_states_different_keys(self):
        s1 = rl_agent.discretize_state(0.5, 70, 0.2, "trending_up", True, 10)
        s2 = rl_agent.discretize_state(-0.5, 30, -0.2, "trending_down", False, 2)
        self.assertNotEqual(s1, s2)

    def test_same_inputs_same_key(self):
        s1 = rl_agent.discretize_state(0.05, 45, 0.05, "volatile", False, 8)
        s2 = rl_agent.discretize_state(0.05, 45, 0.05, "volatile", False, 8)
        self.assertEqual(s1, s2)

    def test_unknown_regime_defaults(self):
        s = rl_agent.discretize_state(0, 50, 0, "unknown_regime", False, 0)
        # unknown defaults to 2 (ranging)
        self.assertIn("_2_", s)


class TestRLAgent(unittest.TestCase):
    def setUp(self):
        self.tmpfile = tempfile.mktemp(suffix=".json")
        self.agent = rl_agent.RLAgent(model_file=self.tmpfile, epsilon=0.0)

    def tearDown(self):
        if os.path.exists(self.tmpfile):
            os.remove(self.tmpfile)

    def test_initial_q_values_zero(self):
        q = self.agent._get_q("new_state")
        self.assertEqual(q, [0.0, 0.0, 0.0])

    def test_greedy_with_zero_q(self):
        # With all zeros, any action is valid
        action = self.agent.choose_action_greedy("some_state")
        self.assertIn(action, [0, 1, 2])

    def test_update_changes_q(self):
        state = "test_state"
        self.agent.update(state, 1, 1.0, "next_state")
        q = self.agent._get_q(state)
        self.assertGreater(q[1], 0.0)

    def test_update_stats(self):
        self.agent.update("s1", 0, 0.5, "s2")
        self.assertEqual(self.agent.stats["total_updates"], 1)

    def test_save_and_load(self):
        self.agent.update("s1", 1, 2.0, "s2")
        self.agent.save()
        agent2 = rl_agent.RLAgent(model_file=self.tmpfile, epsilon=0.0)
        q = agent2._get_q("s1")
        self.assertGreater(q[1], 0.0)

    def test_reset(self):
        self.agent.update("x", 0, 1.0, "y")
        self.agent.reset()
        self.assertEqual(len(self.agent.q_table), 0)
        self.assertEqual(self.agent.stats["total_updates"], 0)

    def test_get_action_name(self):
        self.assertEqual(self.agent.get_action_name(0), "HOLD")
        self.assertEqual(self.agent.get_action_name(1), "BUY")
        self.assertEqual(self.agent.get_action_name(2), "SELL")
        self.assertEqual(self.agent.get_action_name(99), "HOLD")

    def test_get_q_summary(self):
        self.agent.update("s", 1, 5.0, "s2")
        summary = self.agent.get_q_summary("s")
        self.assertIn("q_values", summary)
        self.assertIn("best_action", summary)
        self.assertEqual(summary["best_action"], "BUY")

    def test_epsilon_decay(self):
        agent = rl_agent.RLAgent(model_file=self.tmpfile, epsilon=0.5, epsilon_decay=0.9)
        agent.update("a", 0, 1.0, "b")
        self.assertLess(agent.epsilon, 0.5)

    def test_exploration_with_high_epsilon(self):
        agent = rl_agent.RLAgent(model_file=self.tmpfile, epsilon=1.0)
        actions = set()
        for _ in range(50):
            actions.add(agent.choose_action("explore_state"))
        # Con epsilon=1.0, deberia explorar multiples acciones
        self.assertGreater(len(actions), 1)


class TestReward(unittest.TestCase):
    def test_positive_pnl(self):
        r = rl_agent.calculate_reward(0.005, 1, True)
        self.assertGreater(r, 0)

    def test_negative_pnl(self):
        r = rl_agent.calculate_reward(-0.005, 1, False)
        self.assertLess(r, 0)

    def test_hold_correct_bonus(self):
        r = rl_agent.calculate_reward(0.0, 0, True)
        self.assertGreater(r, 0)

    def test_bad_buy_penalty(self):
        r_bad = rl_agent.calculate_reward(0.0, 1, False)
        r_ok = rl_agent.calculate_reward(0.0, 0, True)
        self.assertLess(r_bad, r_ok)


class TestGlobalFunctions(unittest.TestCase):
    def test_get_rl_suggestion(self):
        result = rl_agent.get_rl_suggestion(
            ema_diff_pct=0.1, rsi=55, macd_hist=0.05,
            regime="trending_up", in_position=False, hour=14,
        )
        self.assertIn("action", result)
        self.assertIn(result["action"], ["HOLD", "BUY", "SELL"])
        self.assertIn("q_values", result)

    def test_record_experience(self):
        agent = rl_agent.get_agent()
        initial = agent.stats["total_updates"]
        rl_agent.record_experience(
            0.1, 55, 0.05, "trending_up", False, 14, 1, 0.5,
            0.15, 60, 0.08, "trending_up", True, 14,
        )
        self.assertEqual(agent.stats["total_updates"], initial + 1)


if __name__ == "__main__":
    unittest.main()
