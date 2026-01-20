import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config, apply_overrides, load_config
from env_updown import UpDownEnv
from masking import action_mask_fn
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--model", type=str, default="checkpoints/ppo_maskable_latest.zip")
    parser.add_argument("--episodes", type=int, default=50)
    return parser.parse_args()


def make_env(cfg: Config):
    env = UpDownEnv(cfg, seed=cfg.seed + 1234)
    env = ActionMasker(env, action_mask_fn)
    return DummyVecEnv([lambda: env])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else Config()
    if args.override:
        apply_overrides(cfg, args.override)
    set_seed(cfg.seed)

    env = make_env(cfg)
    model = MaskablePPO.load(args.model, env=env, device=cfg.train.device)

    rewards = []
    worsts = []
    bullets = []
    for _ in range(args.episodes):
        obs = env.reset()
        done = [False]
        ep_reward = 0.0
        terminal = None
        while not done[0]:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, done, infos = env.step(action)
            ep_reward += float(reward[0])
            terminal = infos[0].get("terminal_info")
        rewards.append(ep_reward)
        if terminal:
            worsts.append(float(terminal["worst_final"]))
            bullets.append(int(terminal["bullets_used"]))

    print(f"mean_reward: {np.mean(rewards):.4f}")
    if worsts:
        print(f"mean_worst_final: {np.mean(worsts):.2f}")
    if bullets:
        print(f"mean_bullets_used: {np.mean(bullets):.2f}")


if __name__ == "__main__":
    main()
