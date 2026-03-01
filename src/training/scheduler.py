"""
scheduler.py — LR scheduler for MISM training.

Implements a cosine schedule with linear warm-up.  During warm-up, the LR
rises linearly from 0 to peak.  After warm-up, it decays following a cosine
curve down to min_lr_ratio × peak.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer:           Optimizer,
    num_warmup_steps:    int,
    num_training_steps:  int,
    min_lr_ratio:        float = 0.0,
) -> LambdaLR:
    """Build a cosine LR schedule with linear warm-up.

    Parameters
    ----------
    optimizer          : wrapped optimiser.
    num_warmup_steps   : steps over which LR rises from 0 → peak.
    num_training_steps : total number of optimiser update steps.
    min_lr_ratio       : minimum LR expressed as fraction of peak (default 0.0).
                         E.g. 0.1 means the floor is 10 % of the initial LR.

    Returns
    -------
    LambdaLR scheduler.  Call ``scheduler.step()`` after each optimiser step.
    """
    if num_warmup_steps < 0:
        raise ValueError(f"num_warmup_steps must be ≥ 0, got {num_warmup_steps}")
    if num_training_steps < 1:
        raise ValueError(f"num_training_steps must be ≥ 1, got {num_training_steps}")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(current_step: int) -> float:
        # Linear warm-up phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        decay_steps = num_training_steps - num_warmup_steps
        progress = float(current_step - num_warmup_steps) / float(max(1, decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)
