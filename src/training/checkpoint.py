"""
checkpoint.py — Save / load training checkpoints for MISM.

A checkpoint payload contains:
    model_state_dict     : model weights (DDP-unwrapped)
    optimizer_state_dict : optimiser state
    scheduler_state_dict : LR scheduler state
    epoch                : int (current epoch, 0-indexed)
    step                 : int (global optimiser step)
    stage                : int (1 or 2)
    metrics              : dict[str, float] (latest evaluation metrics)
    config               : dict (serialised MISMConfig for reproducibility)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path:        Union[str, Path],
    model:       torch.nn.Module,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    epoch:       int,
    step:        int,
    stage:       int,
    metrics:     Optional[Dict[str, float]] = None,
    config_dict: Optional[Dict[str, Any]]   = None,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    path       : file path (.pt) for the checkpoint.
    model      : model instance (DDP-wrapped or plain).
    optimizer  : AdamW (or other) optimiser.
    scheduler  : LambdaLR scheduler.
    epoch      : current epoch (0-indexed).
    step       : global optimiser step count.
    stage      : training stage (1 or 2).
    metrics    : optional latest evaluation metrics.
    config_dict: optional serialised training config dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if necessary
    raw_model = model.module if hasattr(model, "module") else model

    payload: Dict[str, Any] = {
        "model_state_dict":     raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch":                epoch,
        "step":                 step,
        "stage":                stage,
        "metrics":              metrics or {},
        "config":               config_dict or {},
    }
    torch.save(payload, path)
    logger.info(
        "Checkpoint saved → %s  (epoch=%d, step=%d, stage=%d)",
        path, epoch, step, stage,
    )


def load_checkpoint(
    path:      Union[str, Path],
    model:     Optional[torch.nn.Module]           = None,
    optimizer: Optional[torch.optim.Optimizer]     = None,
    scheduler=None,
    strict:    bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore model / optimiser / scheduler.

    Parameters
    ----------
    path      : file path (.pt) of the checkpoint.
    model     : optional — restored in-place if provided.
    optimizer : optional — restored in-place if provided.
    scheduler : optional — restored in-place if provided.
    strict    : strict model state_dict loading (default True).

    Returns
    -------
    Dict with keys: epoch, step, stage, metrics, config.

    Raises
    ------
    FileNotFoundError if the checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)

    if model is not None:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(payload["model_state_dict"], strict=strict)
        logger.info("Model state restored from %s", path)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return {
        "epoch":   payload.get("epoch",   0),
        "step":    payload.get("step",    0),
        "stage":   payload.get("stage",   1),
        "metrics": payload.get("metrics", {}),
        "config":  payload.get("config",  {}),
    }
