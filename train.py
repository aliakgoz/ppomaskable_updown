import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config import Config, apply_overrides, load_config, save_config
from env_updown import UpDownEnv
from masking import action_mask_fn
from utils import set_seed


class TerminalInfoCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            terminal = info.get("terminal_info")
            if not terminal:
                continue
            self.logger.record("episode/worst_final", terminal["worst_final"])
            self.logger.record("episode/cash_final", terminal["cash_final"])
            self.logger.record("episode/q_up", terminal["q_up"])
            self.logger.record("episode/q_down", terminal["q_down"])
            self.logger.record("episode/bullets_used", terminal["bullets_used"])
            self.logger.record("episode/outcome", terminal["outcome"])
            self.logger.record("episode/p_min", terminal["p_min"])
            self.logger.record("episode/p_mean", terminal["p_mean"])
            self.logger.record("episode/p_max", terminal["p_max"])
            counts = terminal["action_counts"]
            self.logger.record("episode/action_hold", counts.get("hold", 0))
            self.logger.record("episode/action_buy_up", counts.get("buy_up", 0))
            self.logger.record("episode/action_buy_down", counts.get("buy_down", 0))
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-timesteps", action="store_true")
    return parser.parse_args()


def make_env_fn(cfg: Config, seed: int, rank: int) -> Callable[[], UpDownEnv]:
    def _init():
        env = UpDownEnv(cfg, seed=seed + rank)
        env = ActionMasker(env, action_mask_fn)
        env = RecordEpisodeStatistics(env)
        return env

    return _init


def build_vec_env(cfg: Config) -> DummyVecEnv | SubprocVecEnv:
    env_fns = [make_env_fn(cfg, cfg.seed, i) for i in range(cfg.ppo.n_envs)]
    if cfg.ppo.n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def run_eval(model: MaskablePPO, cfg: Config) -> None:
    eval_cfg = Config(
        seed=cfg.seed + 10_000,
        env=cfg.env,
        series=cfg.series,
        policy=cfg.policy,
        ppo=cfg.ppo,
        train=cfg.train,
    )
    eval_cfg.ppo.n_envs = 1
    eval_env = build_vec_env(eval_cfg)
    rewards = []
    worsts = []
    bullets = []
    for _ in range(cfg.train.eval_episodes):
        obs = eval_env.reset()
        done = [False]
        ep_reward = 0.0
        terminal = None
        while not done[0]:
            masks = get_action_masks(eval_env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, done, infos = eval_env.step(action)
            ep_reward += float(reward[0])
            terminal = infos[0].get("terminal_info")
        rewards.append(ep_reward)
        if terminal:
            worsts.append(float(terminal["worst_final"]))
            bullets.append(int(terminal["bullets_used"]))
    if rewards:
        print(f"Eval mean reward: {np.mean(rewards):.4f}")
    if worsts:
        print(f"Eval mean worst_final: {np.mean(worsts):.2f}")
    if bullets:
        print(f"Eval mean bullets_used: {np.mean(bullets):.2f}")


def apply_smoke(cfg: Config) -> None:
    cfg.ppo.n_envs = 4
    cfg.ppo.n_steps = 256
    cfg.ppo.batch_size = 1024
    cfg.ppo.n_epochs = 5
    cfg.ppo.total_timesteps = 10_000
    cfg.train.eval_every = 10_000
    cfg.train.eval_episodes = 4


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else Config()
    if args.override:
        apply_overrides(cfg, args.override)
    if args.smoke:
        apply_smoke(cfg)

    set_seed(cfg.seed)
    if cfg.train.matmul_precision:
        torch.set_float32_matmul_precision(cfg.train.matmul_precision)
    if cfg.train.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    log_dir = Path(cfg.train.log_dir)
    model_dir = Path(cfg.train.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(model_dir / "config.json"))

    env = build_vec_env(cfg)

    policy_kwargs = {
        "net_arch": {"pi": cfg.policy.pi, "vf": cfg.policy.vf},
        "activation_fn": torch.nn.SiLU,
    }

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device=cfg.train.device)
        # Force LR override on resume so new config takes effect.
        model.learning_rate = cfg.ppo.learning_rate
        model.lr_schedule = lambda _: cfg.ppo.learning_rate
        if getattr(model.policy, "optimizer", None) is not None:
            for group in model.policy.optimizer.param_groups:
                group["lr"] = cfg.ppo.learning_rate
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=cfg.ppo.n_steps,
            batch_size=cfg.ppo.batch_size,
            n_epochs=cfg.ppo.n_epochs,
            learning_rate=cfg.ppo.learning_rate,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            clip_range=cfg.ppo.clip_range,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            target_kl=cfg.ppo.target_kl,
            normalize_advantage=cfg.ppo.normalize_advantage,
            policy_kwargs=policy_kwargs,
            device=cfg.train.device,
            verbose=1,
        )

    new_logger = configure(str(log_dir), ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.train.save_freq,
        save_path=str(model_dir),
        name_prefix="ppo_maskable",
    )
    terminal_cb = TerminalInfoCallback()
    eval_cfg = Config(
        seed=cfg.seed + 10_000,
        env=cfg.env,
        series=cfg.series,
        policy=cfg.policy,
        ppo=cfg.ppo,
        train=cfg.train,
    )
    eval_cfg.ppo.n_envs = 1
    eval_env = build_vec_env(eval_cfg)
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=cfg.train.eval_every,
        n_eval_episodes=cfg.train.eval_episodes,
        deterministic=True,
    )

    model.learn(
        total_timesteps=cfg.ppo.total_timesteps,
        callback=[checkpoint_cb, terminal_cb, eval_cb],
        reset_num_timesteps=args.reset_timesteps,
    )

    model.save(str(model_dir / "ppo_maskable_latest"))
    run_eval(model, cfg)


if __name__ == "__main__":
    main()
