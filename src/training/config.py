"""
config.py — MISMConfig dataclass and YAML loader.

The config is a single flat dataclass.  YAML files may organise keys into
named sections (model:, training:, loss:, data:, logging:, output:) — these
sections are merged automatically when loading.

Usage
-----
    cfg = load_config("configs/gazeta_2stage.yaml")
    cfg = load_config("configs/base.yaml", overrides={"batch_size": 2})
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class MISMConfig:
    # ── Model architecture ────────────────────────────────────────────
    model_name:           str   = "IlyaGusev/rut5_base_sum_gazeta"
    hidden_size:          int   = 768
    num_heads:            int   = 12
    ffn_dim:              int   = 3072
    dropout:              float = 0.1
    window_size:          int   = 512
    window_overlap:       int   = 128
    max_windows:          int   = 32
    max_kw:               int   = 20
    kw_max_len:           int   = 32
    max_summary_tokens:   int   = 256
    max_src_len:          int   = 4096

    # ── Stage 1: Fusion + KAL only ────────────────────────────────────
    stage1_epochs:        int   = 5
    stage1_lr:            float = 1e-4

    # ── Stage 2: Fusion + KAL + Decoder ──────────────────────────────
    stage2_epochs:        int   = 15
    stage2_lr:            float = 5e-5

    # ── Common training ───────────────────────────────────────────────
    batch_size:           int   = 4       # per GPU
    grad_accum_steps:     int   = 8
    max_grad_norm:        float = 1.0
    warmup_ratio:         float = 0.05
    weight_decay:         float = 0.01
    bf16:                 bool  = True
    gradient_checkpointing: bool = True
    seed:                 int   = 42

    # ── Loss weights ──────────────────────────────────────────────────
    lambda_gen:           float = 0.65
    lambda_cover:         float = 0.15
    lambda_bert:          float = 0.15
    lambda_gate:          float = 0.05
    label_smoothing:      float = 0.1
    gate_threshold_low:   float = 0.2
    gate_threshold_high:  float = 0.5

    # ── Data paths ────────────────────────────────────────────────────
    train_path:           str   = "dataset/splits/train.json"
    val_path:             str   = "dataset/splits/val.json"
    test_path:            str   = "dataset/splits/test.json"

    # ── Logging ───────────────────────────────────────────────────────
    wandb_project:        str   = "mism-summarization"
    wandb_run_name:       Optional[str] = None
    use_wandb:            bool  = True
    use_tensorboard:      bool  = True
    log_every_steps:      int   = 50
    eval_every_steps:     int   = 500
    save_every_steps:     int   = 1000

    # ── Output ────────────────────────────────────────────────────────
    checkpoint_dir:       str   = "checkpoints"

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MISMConfig":
        """Construct from a flat dict; unknown keys are silently ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def __post_init__(self) -> None:
        # Light validation
        if self.stage1_lr <= 0:
            raise ValueError(f"stage1_lr must be > 0, got {self.stage1_lr}")
        if self.stage2_lr <= 0:
            raise ValueError(f"stage2_lr must be > 0, got {self.stage2_lr}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {self.batch_size}")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")


def load_config(
    path:      Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> MISMConfig:
    """Load a YAML config file into a MISMConfig.

    YAML may be flat or sectioned (model:, training:, …).  Nested sections
    are flattened by merging their contents into a single dict.

    Parameters
    ----------
    path      : path to the YAML config file.
    overrides : optional dict of key → value applied after YAML loading.

    Returns
    -------
    MISMConfig instance.
    """
    path = Path(path)
    with open(path) as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Flatten nested sections
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value

    if overrides:
        flat.update(overrides)

    return MISMConfig.from_dict(flat)
