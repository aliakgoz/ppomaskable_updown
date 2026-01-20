from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from config import Config
from price_series import PolymarketSeries, make_series, sample_outcome
from utils import price_pair


@dataclass
class TradeState:
    t: int
    cash: float
    q_up: float
    q_down: float
    bullets_left: int


class UpDownEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._poly_source: Optional[PolymarketSeries] = None
        self.amounts = self._build_amount_buckets()
        self.action_dim = 1 + 2 * len(self.amounts)
        self.action_space = gym.spaces.Discrete(self.action_dim)

        obs_dim = 6 + int(cfg.env.n_returns)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.prices: np.ndarray = np.zeros(cfg.env.episode_length, dtype=np.float32)
        self.state = TradeState(0, cfg.env.initial_cash, 0.0, 0.0, cfg.env.max_bullets)
        self.prev_worst = float(cfg.env.initial_cash)
        self._reset_episode_stats()

    def _build_amount_buckets(self) -> List[float]:
        cfg = self.cfg.env
        steps = int(round((cfg.amount_max - cfg.amount_min) / cfg.amount_step)) + 1
        return [round(cfg.amount_min + i * cfg.amount_step, 2) for i in range(steps)]

    def _reset_episode_stats(self) -> None:
        self._p_min = 1.0
        self._p_max = 0.0
        self._p_sum = 0.0
        self._invalid_actions = 0
        self._action_counts = {"hold": 0, "buy_up": 0, "buy_down": 0}

    def _update_price_stats(self, p_up: float) -> None:
        self._p_min = min(self._p_min, p_up)
        self._p_max = max(self._p_max, p_up)
        self._p_sum += p_up

    def _current_price(self) -> Tuple[float, float]:
        p_up = float(self.prices[self.state.t])
        return price_pair(p_up)

    def _build_obs(self) -> np.ndarray:
        t_idx = min(self.state.t, self.cfg.env.episode_length - 1)
        p_up = float(self.prices[t_idx])
        cash_norm = self.state.cash / float(self.cfg.env.initial_cash)
        q_up_norm = self.state.q_up / float(self.cfg.env.q_scale)
        q_down_norm = self.state.q_down / float(self.cfg.env.q_scale)
        bullets_norm = self.state.bullets_left / float(self.cfg.env.max_bullets)
        time_norm = self.state.t / float(self.cfg.env.episode_length - 1)

        n_ret = int(self.cfg.env.n_returns)
        if t_idx == 0:
            returns = np.zeros(n_ret, dtype=np.float32)
        else:
            start = max(1, t_idx - n_ret + 1)
            deltas = self.prices[start : t_idx + 1] - self.prices[start - 1 : t_idx]
            deltas = np.clip(deltas, -0.1, 0.1)
            if deltas.shape[0] < n_ret:
                pad = np.zeros((n_ret - deltas.shape[0],), dtype=np.float32)
                returns = np.concatenate([pad, deltas], axis=0)
            else:
                returns = deltas[-n_ret:]

        obs = np.concatenate(
            [
                np.asarray(
                    [p_up, cash_norm, q_up_norm, q_down_norm, bullets_norm, time_norm],
                    dtype=np.float32,
                ),
                returns.astype(np.float32),
            ],
            axis=0,
        )
        return obs

    def _action_to_trade(self, action: int) -> Tuple[str, float]:
        if action == 0:
            return "HOLD", 0.0
        idx = action - 1
        if idx < len(self.amounts):
            return "BUY_UP", self.amounts[idx]
        idx -= len(self.amounts)
        return "BUY_DOWN", self.amounts[idx]

    def get_action_mask(self) -> np.ndarray:
        p_up, p_down = self._current_price()
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[0] = True

        if self.state.bullets_left <= 0:
            return mask
        min_amount = self.amounts[0]
        if self.state.cash < min_amount - 1e-9:
            return mask

        for i, amt in enumerate(self.amounts):
            if self.state.cash >= amt - 1e-9 and p_up >= self.cfg.env.min_price_for_trade:
                mask[1 + i] = True
            if self.state.cash >= amt - 1e-9 and p_down >= self.cfg.env.min_price_for_trade:
                mask[1 + len(self.amounts) + i] = True
        return mask

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.prices, self._poly_source = make_series(self.cfg, self._rng, self._poly_source)
        self.state = TradeState(
            t=0,
            cash=float(self.cfg.env.initial_cash),
            q_up=0.0,
            q_down=0.0,
            bullets_left=int(self.cfg.env.max_bullets),
        )
        self.prev_worst = float(self.cfg.env.initial_cash)
        self._reset_episode_stats()
        p_up, _ = self._current_price()
        self._update_price_stats(p_up)
        obs = self._build_obs()
        info = self._build_info(p_up)
        return obs, info

    def step(self, action: int):
        if self.state.t >= self.cfg.env.episode_length:
            raise RuntimeError("step called after episode end")

        p_up, p_down = self._current_price()
        self._update_price_stats(p_up)
        action_mask = self.get_action_mask()

        invalid = False
        if not action_mask[action]:
            invalid = True
            action = 0

        action_type, amount = self._action_to_trade(action)
        cash = self.state.cash
        q_up = self.state.q_up
        q_down = self.state.q_down
        bullets_left = self.state.bullets_left

        if action_type == "BUY_UP":
            shares = amount / max(p_up, self.cfg.env.price_eps)
            cash -= amount
            q_up += shares
            bullets_left -= 1
            self._action_counts["buy_up"] += 1
        elif action_type == "BUY_DOWN":
            shares = amount / max(p_down, self.cfg.env.price_eps)
            cash -= amount
            q_down += shares
            bullets_left -= 1
            self._action_counts["buy_down"] += 1
        else:
            self._action_counts["hold"] += 1

        if invalid:
            self._invalid_actions += 1

        cash = max(cash, 0.0)

        self.state.cash = cash
        self.state.q_up = q_up
        self.state.q_down = q_down
        self.state.bullets_left = max(0, bullets_left)

        V_up = cash + q_up
        V_down = cash + q_down
        worst = min(V_up, V_down)
        delta = (worst - self.prev_worst) / float(self.cfg.env.initial_cash)
        reward = delta if delta >= 0 else self.cfg.env.loss_factor * delta

        if self.cfg.env.pacing_enabled:
            time_progress = self.state.t / float(self.cfg.env.episode_length - 1)
            bullets_used = self.cfg.env.max_bullets - self.state.bullets_left
            ideal_used = self.cfg.env.max_bullets * time_progress
            pacing_error = abs(bullets_used - ideal_used) / float(self.cfg.env.max_bullets)
            reward -= self.cfg.env.pacing_k * pacing_error

        self.prev_worst = worst

        terminated = self.state.t >= self.cfg.env.episode_length - 1
        truncated = False

        info = self._build_info(p_up)
        info["action_mask"] = action_mask
        info["invalid_action"] = 1 if invalid else 0
        info["V_up"] = V_up
        info["V_down"] = V_down
        info["worst"] = worst

        if terminated:
            outcome = sample_outcome(self.cfg, self.prices, self._rng)
            info["outcome"] = int(outcome)
            info["terminal_info"] = self._terminal_info(outcome)

        self.state.t += 1
        obs = self._build_obs()
        return obs, float(reward), terminated, truncated, info

    def _terminal_info(self, outcome: int) -> Dict[str, float]:
        cash = self.state.cash
        q_up = self.state.q_up
        q_down = self.state.q_down
        v_up = cash + q_up
        v_down = cash + q_down
        worst = min(v_up, v_down)
        bullets_used = self.cfg.env.max_bullets - self.state.bullets_left
        p_mean = self._p_sum / float(self.cfg.env.episode_length)
        return {
            "cash_final": cash,
            "q_up": q_up,
            "q_down": q_down,
            "V_up_final": v_up,
            "V_down_final": v_down,
            "worst_final": worst,
            "bullets_used": bullets_used,
            "p_min": self._p_min,
            "p_mean": p_mean,
            "p_max": self._p_max,
            "invalid_actions": self._invalid_actions,
            "action_counts": dict(self._action_counts),
            "outcome": outcome,
        }

    def _build_info(self, p_up: float) -> Dict[str, float]:
        p_mean = self._p_sum / float(max(1, self.state.t + 1))
        bullets_used = self.cfg.env.max_bullets - self.state.bullets_left
        return {
            "p_up": p_up,
            "p_min": self._p_min,
            "p_mean": p_mean,
            "p_max": self._p_max,
            "cash": self.state.cash,
            "q_up": self.state.q_up,
            "q_down": self.state.q_down,
            "bullets_left": self.state.bullets_left,
            "bullets_used": bullets_used,
            "invalid_actions": self._invalid_actions,
            "action_counts": dict(self._action_counts),
        }
