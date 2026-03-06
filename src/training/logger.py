"""
logger.py — MetricsLogger: unified logging backend for MISM training.

Backends (all optional except console)
---------------------------------------
Console      : always on — Python logging (INFO level)
JSON         : metrics.jsonl  — one JSON record per step, always flushed
TensorBoard  : requires ``pip install tensorboard``
W&B          : requires ``pip install wandb`` and prior ``wandb login``

Only the main DDP process (is_main_process=True) writes to any backend;
other ranks are silently no-ops.

Usage
-----
    logger = MetricsLogger(
        log_dir=Path("checkpoints/run1/logs"),
        config_dict=cfg.to_dict(),
        use_wandb=cfg.use_wandb,
        use_tensorboard=cfg.use_tensorboard,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
        is_main_process=True,
    )
    logger.log({"train/loss": 2.3, "train/lr": 1e-4}, step=100)
    logger.close()
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

# Optional backends
try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _TB_OK = True
except ImportError:
    _TB_OK = False


class MetricsLogger:
    """Unified multi-backend metrics logger.

    Parameters
    ----------
    log_dir          : directory for JSON + TensorBoard files.
    config_dict      : serialised training config (for W&B / TensorBoard hparams).
    use_wandb        : enable W&B backend (default True).
    use_tensorboard  : enable TensorBoard backend (default True).
    wandb_project    : W&B project name.
    wandb_run_name   : W&B run name (None → auto-generated).
    is_main_process  : if False, all calls are no-ops (default True).
    """

    def __init__(
        self,
        log_dir:         Path,
        config_dict:     Dict[str, Any],
        use_wandb:       bool = True,
        use_tensorboard: bool = True,
        wandb_project:   str  = "mism-summarization",
        wandb_run_name:  Optional[str] = None,
        is_main_process: bool = True,
    ) -> None:
        self.is_main = is_main_process
        self._wb_run   = None
        self._tb       = None
        self._jsonf    = None

        if not is_main_process:
            return

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # ── JSON log ──────────────────────────────────────────────────
        self._jsonf = open(log_dir / "metrics.jsonl", "a", encoding="utf-8")
        _log.info("JSON metrics → %s", log_dir / "metrics.jsonl")

        # ── TensorBoard ───────────────────────────────────────────────
        if use_tensorboard:
            if _TB_OK:
                tb_dir = log_dir / "tensorboard"
                self._tb = _SummaryWriter(log_dir=str(tb_dir))
                _log.info("TensorBoard → %s", tb_dir)
            else:
                _log.warning(
                    "use_tensorboard=True but 'tensorboard' is not installed. "
                    "Run: pip install tensorboard"
                )

        # ── W&B ───────────────────────────────────────────────────────
        if use_wandb:
            if _WANDB_OK:
                # In DDP training, an interactive wandb login prompt blocks
                # the main process and causes NCCL timeouts on other ranks.
                # Check for a valid API key *before* calling wandb.init().
                _api_key = os.environ.get("WANDB_API_KEY", "")
                if not _api_key:
                    try:
                        _api_key = _wandb.api.api_key or ""
                    except Exception:
                        _api_key = ""
                if _api_key:
                    self._wb_run = _wandb.init(
                        project=wandb_project,
                        name=wandb_run_name,
                        config=config_dict,
                        resume="allow",
                    )
                    _log.info("W&B run → %s",
                              self._wb_run.url if self._wb_run else "?")
                else:
                    _log.warning(
                        "use_wandb=True but W&B is not authenticated. "
                        "Run `wandb login` on the server or set the "
                        "WANDB_API_KEY env var.  Continuing without W&B."
                    )
            else:
                _log.warning(
                    "use_wandb=True but 'wandb' is not installed. "
                    "Run: pip install wandb"
                )

    # ------------------------------------------------------------------

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Write metrics to all enabled backends.

        Parameters
        ----------
        metrics : dict mapping metric name → float value.
        step    : global optimiser step (used as x-axis in all backends).
        """
        if not self.is_main:
            return

        # Console
        parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        _log.info("step=%-7d  %s", step, parts)

        # JSON — always flushed so the file can be read while training
        if self._jsonf is not None:
            record = {"step": step, "ts": time.time(), **metrics}
            self._jsonf.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonf.flush()

        # TensorBoard
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, global_step=step)

        # W&B
        if self._wb_run is not None:
            self._wb_run.log(metrics, step=step)

    def close(self) -> None:
        """Flush and close all backends. Call once at end of training."""
        if not self.is_main:
            return
        if self._jsonf is not None:
            self._jsonf.close()
            self._jsonf = None
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._wb_run is not None:
            self._wb_run.finish()
            self._wb_run = None
