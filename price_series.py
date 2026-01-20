import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config import Config
from data_generator import PolymarketDataGenerator
from utils import quantize_price


class PolymarketSeries:
    def __init__(self, batch_size: int):
        self._gen = PolymarketDataGenerator().yield_batches(batch_size=batch_size)
        self._buffer = None
        self._index = 0

    def next_series(self) -> np.ndarray:
        if self._buffer is None or self._index >= len(self._buffer):
            batch, _ = next(self._gen)
            self._buffer = np.asarray(batch, dtype=np.float32)
            self._index = 0
        series = self._buffer[self._index]
        self._index += 1
        return np.asarray(series, dtype=np.float32)


def generate_synthetic_series(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(loc=0.0, scale=0.5)
    regime = 0
    prices = np.zeros(cfg.env.episode_length, dtype=np.float32)
    for t in range(cfg.env.episode_length):
        if rng.random() < cfg.series.synthetic_switch_prob:
            regime = 1 - regime
        sigma = cfg.series.synthetic_sigma_high if regime else cfg.series.synthetic_sigma_low
        x = x + cfg.series.synthetic_drift + sigma * rng.normal() - cfg.series.synthetic_mean_reversion * x
        x = float(np.clip(x, -20.0, 20.0))
        raw = 1.0 / (1.0 + np.exp(-x))
        prices[t] = float(quantize_price(raw))
    return prices


def load_file_series(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    series_dir = Path(cfg.series.series_dir)
    if not series_dir.exists():
        raise FileNotFoundError(f"series_dir not found: {series_dir}")
    files = [p for p in series_dir.iterdir() if p.suffix.lower() in (".json", ".npy", ".csv")]
    if not files:
        raise FileNotFoundError(f"no series files found in {series_dir}")
    path = files[int(rng.integers(0, len(files)))]
    if path.suffix.lower() == ".npy":
        data = np.load(path)
    elif path.suffix.lower() == ".csv":
        data = np.loadtxt(path, delimiter=",")
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if arr.shape[0] != cfg.env.episode_length:
        raise ValueError(f"series length {arr.shape[0]} != {cfg.env.episode_length} in {path}")
    arr = np.clip(arr, 0.0, 1.0)
    out = np.zeros_like(arr, dtype=np.float32)
    for i, v in enumerate(arr.tolist()):
        out[i] = float(quantize_price(v))
    return out


def make_series(
    cfg: Config, rng: np.random.Generator, poly_source: Optional[PolymarketSeries]
) -> Tuple[np.ndarray, Optional[PolymarketSeries]]:
    if cfg.series.mode == "synthetic":
        return generate_synthetic_series(cfg, rng), poly_source
    if cfg.series.mode == "file":
        return load_file_series(cfg, rng), poly_source
    if cfg.series.mode == "polymarket":
        if poly_source is None:
            poly_source = PolymarketSeries(cfg.series.polymarket_batch_size)
        series = poly_source.next_series()
        series = np.clip(series, 0.0, 1.0)
        out = np.zeros_like(series, dtype=np.float32)
        for i, v in enumerate(series.tolist()):
            out[i] = float(quantize_price(v))
        if out.shape[0] != cfg.env.episode_length:
            raise ValueError(f"polymarket series length {out.shape[0]} != {cfg.env.episode_length}")
        return out, poly_source
    raise ValueError(f"unknown series mode: {cfg.series.mode}")


def sample_outcome(cfg: Config, prices: np.ndarray, rng: np.random.Generator) -> int:
    if cfg.env.outcome_mode == "threshold":
        return 1 if float(prices[-1]) >= 0.5 else 0
    if cfg.env.outcome_mode == "random":
        return int(rng.random() < 0.5)
    raise ValueError(f"unknown outcome_mode: {cfg.env.outcome_mode}")
