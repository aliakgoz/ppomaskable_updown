import numpy as np


def action_mask_fn(env) -> np.ndarray:
    return env.get_action_mask()
