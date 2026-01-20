import numpy as np

from config import Config
from env_updown import UpDownEnv


def test_env_no_fee():
    cfg = Config()
    env = UpDownEnv(cfg)
    env.prices = np.full(cfg.env.episode_length, 0.5, dtype=np.float32)
    obs, info = env.reset()
    buy_up_idx = 1 + (10 - 4)
    obs, reward, terminated, truncated, info = env.step(buy_up_idx)
    assert abs(env.state.cash - (cfg.env.initial_cash - 10.0)) < 1e-6
