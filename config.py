import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EnvConfig:
    episode_length: int = 900
    initial_cash: float = 200.0
    max_bullets: int = 20
    min_price_for_trade: float = 0.02
    price_eps: float = 0.01
    n_returns: int = 10
    q_scale: float = 200.0
    outcome_mode: str = "threshold"
    loss_factor: float = 4.0
    pacing_enabled: bool = True
    pacing_k: float = 0.005
    min_bullets: int = 5
    min_bullets_penalty: float = 10.0
    amount_min: float = 4.0
    amount_max: float = 20.0
    amount_step: float = 1.0


@dataclass
class SeriesConfig:
    mode: str = "polymarket"  # polymarket, synthetic, file
    series_dir: str = "data/series"
    polymarket_batch_size: int = 64
    synthetic_drift: float = 0.0
    synthetic_sigma_low: float = 0.02
    synthetic_sigma_high: float = 0.08
    synthetic_switch_prob: float = 0.02
    synthetic_mean_reversion: float = 0.02


@dataclass
class PolicyConfig:
    pi: List[int] = field(default_factory=lambda: [4096, 4096, 2048, 2048, 1024])
    vf: List[int] = field(default_factory=lambda: [4096, 4096, 2048, 2048, 1024])
    activation: str = "silu"


@dataclass
class PPOConfig:
    n_envs: int = 64
    n_steps: int = 2048
    batch_size: int = 16384
    n_epochs: int = 10
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    normalize_advantage: bool = True
    use_sde: bool = False


@dataclass
class TrainConfig:
    device: str = "cuda"
    log_dir: str = "runs"
    model_dir: str = "checkpoints"
    save_freq: int = 200_000
    eval_every: int = 500_000
    eval_episodes: int = 20
    use_cudnn_benchmark: bool = True
    matmul_precision: str = "high"
    compile: bool = False


@dataclass
class Config:
    seed: int = 42
    env: EnvConfig = field(default_factory=EnvConfig)
    series: SeriesConfig = field(default_factory=SeriesConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def to_dict(cfg: Config) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: Config, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")


def load_config(path: str) -> Config:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = Config()
    _update_dataclass(cfg, data)
    return cfg


def apply_overrides(cfg: Config, overrides: List[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        _set_by_path(cfg, key, raw)


def _update_dataclass(obj: Any, data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _set_by_path(cfg: Config, key: str, raw: str) -> None:
    parts = key.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"unknown config section: {part}")
        obj = getattr(obj, part)
    field_name = parts[-1]
    if not hasattr(obj, field_name):
        raise KeyError(f"unknown config key: {key}")
    current = getattr(obj, field_name)
    value = _parse_value(raw, current)
    setattr(obj, field_name, value)


def _parse_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "y")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        if raw.strip().startswith("["):
            return json.loads(raw)
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return raw
