# MaskablePPO Up/Down Market

Train a large MaskablePPO agent on a 900-step UP/DOWN market with strict action
masking and a polymarket-style data generator.

## Quickstart (Windows 11, Python 3.11)

1) Create and activate a venv.
```
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install PyTorch with CUDA (pick the CUDA version that matches your driver).
```
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

RTX PRO 6000 (Blackwell, sm_120) note:
- Use CUDA 12.8 wheels (cu128) and Torch 2.9.x.
```
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

3) Install project requirements.
```
pip install -r requirements.txt
```

4) Train (full run) or smoke test.
```
python train.py
python train.py --smoke
```

5) Evaluate a saved model.
```
python eval.py --model checkpoints/ppo_maskable_latest.zip
```

## What this does

- Episode length is 900 steps, initial cash is 200, max 20 bullets.
- Actions are discrete: HOLD, BUY_UP amount, BUY_DOWN amount.
- Amount buckets are 4.00 to 20.00 by 1.00 (17 buckets).
- Prices are always 2-decimal and p_up + p_down == 1.00 after rounding.
- Reward is worst-case delta with loss aversion, plus optional pacing penalty.
- Minimum bullets: by default `min_bullets=5`. If fewer are used, a terminal
  penalty is applied (`min_bullets_penalty=10.0` by default). Set
  `min_bullets=0` to disable.
- Action masking prevents invalid buys (cash, bullets, min price).

## Data generation

Default is polymarket-style synthetic data (logit random walk with regime
switching).

Config knobs:
- `series.mode=polymarket` (default)
- `series.mode=synthetic`
- `series.mode=file` and `series.series_dir=data/series` (json, npy, csv)

Example override:
```
python train.py --override series.mode=synthetic
```

## Config overrides

You can override any config key:
```
python train.py --override env.initial_cash=200 ppo.n_envs=32 ppo.n_steps=2048
```

## Tests

```
python -m pytest
```

## Performance notes

- Defaults use a very large policy network and large batches. If you hit OOM,
  reduce `ppo.n_envs`, `ppo.n_steps`, or `ppo.batch_size`.
- GPU is used by default (`train.device=cuda`). To force CPU:
```
python train.py --override train.device=cpu
```

## Files

- `config.py` - dataclass config and overrides
- `env_updown.py` - environment (gymnasium)
- `price_series.py` - polymarket/synthetic/file series
- `masking.py` - action mask helper
- `train.py` - training entry point (MaskablePPO)
- `eval.py` - evaluation entry point
- `tests/` - minimal invariants
