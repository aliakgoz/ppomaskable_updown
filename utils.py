import random
from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple

import numpy as np
import torch


TWO_PLACES = Decimal("0.01")
ONE = Decimal("1.00")


def quantize_2(value: Decimal) -> Decimal:
    return value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def quantize_price(raw_value: float) -> float:
    return float(quantize_2(Decimal(str(raw_value))))


def price_pair(p_up_raw: float) -> Tuple[float, float]:
    p_up = quantize_2(Decimal(str(p_up_raw)))
    p_down = quantize_2(ONE - p_up)
    p_up = min(max(p_up, Decimal("0.00")), ONE)
    p_down = min(max(p_down, Decimal("0.00")), ONE)
    return float(p_up), float(p_down)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
