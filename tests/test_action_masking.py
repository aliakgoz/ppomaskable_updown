import numpy as np

from config import Config
from env_updown import TradeState, UpDownEnv


def test_action_masking_bullets_and_cash():
    cfg = Config()
    env = UpDownEnv(cfg)
    env.prices = np.full(cfg.env.episode_length, 0.5, dtype=np.float32)

    env.state = TradeState(t=0, cash=cfg.env.initial_cash, q_up=0.0, q_down=0.0, bullets_left=0)
    mask = env.get_action_mask()
    assert mask[0]
    assert mask[1:].sum() == 0

    env.state = TradeState(t=0, cash=3.0, q_up=0.0, q_down=0.0, bullets_left=cfg.env.max_bullets)
    mask = env.get_action_mask()
    assert mask[0]
    assert mask[1:].sum() == 0


def test_action_masking_price_limits():
    cfg = Config()
    env = UpDownEnv(cfg)
    env.prices = np.full(cfg.env.episode_length, 0.01, dtype=np.float32)
    env.state = TradeState(t=0, cash=cfg.env.initial_cash, q_up=0.0, q_down=0.0, bullets_left=cfg.env.max_bullets)
    mask = env.get_action_mask()
    assert mask[0]
    assert mask[1 : 1 + len(env.amounts)].sum() == 0
    assert mask[1 + len(env.amounts) :].sum() > 0
