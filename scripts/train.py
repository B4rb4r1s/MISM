#!/usr/bin/env python
"""
train.py — MISM training entry point.

Single-GPU:
    python scripts/train.py --config configs/gazeta_2stage.yaml

Multi-GPU (8×V100, DDP via torchrun):
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/gazeta_2stage.yaml

Run only Stage 1:
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/gazeta_2stage.yaml --stage 1

Run only Stage 2 (load model from Stage 1 checkpoint):
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/gazeta_2stage.yaml --stage 2 \
        --resume checkpoints/gazeta_2stage/best.pt

Resume interrupted training (restores optimizer + scheduler state):
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/gazeta_2stage.yaml \
        --resume checkpoints/gazeta_2stage/step_0005000.pt

Override individual config values:
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/gazeta_2stage.yaml \
        --set batch_size=2 stage1_epochs=2
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import torch

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer

from src.data.dataset import SummarizationDataset
from src.losses.composite_loss import CompositeLoss
from src.models.dual_encoder_summarizer import DualEncoderSummarizer
from src.training.config import load_config
from src.training.trainer import MISMTrainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MISM Dual-Encoder Summarizer")
    p.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. configs/gazeta_2stage.yaml)",
    )
    p.add_argument(
        "--resume", default=None,
        help="Path to checkpoint (.pt) to resume from",
    )
    p.add_argument(
        "--stage", type=int, default=0, choices=[0, 1, 2],
        help="Run specific stage only: 1 or 2. Default 0 = both sequentially.",
    )
    p.add_argument(
        "--set", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override individual config values, e.g. --set batch_size=2 seed=1",
    )
    return p.parse_args()


def _parse_overrides(pairs: list[str]) -> dict:
    """Parse ['key=value', ...] into a dict with auto-typed values."""
    overrides: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got: {pair!r}")
        key, raw_val = pair.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        # Auto-type: bool, int, float, str
        if raw_val.lower() in ("true", "false"):
            overrides[key] = raw_val.lower() == "true"
        else:
            try:
                overrides[key] = int(raw_val)
            except ValueError:
                try:
                    overrides[key] = float(raw_val)
                except ValueError:
                    overrides[key] = raw_val  # keep as string
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── DDP environment variables (set by torchrun) ───────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    rank        = int(os.environ.get("RANK",        0))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        if world_size > 1:
            logger.warning("DDP requested but CUDA not available — using CPU (debug only)")

    # ── Config ────────────────────────────────────────────────────────────
    overrides = _parse_overrides(args.set)
    cfg = load_config(args.config, overrides=overrides)

    torch.manual_seed(cfg.seed + rank)

    # ── File logging (rank 0 only) ────────────────────────────────────────
    if local_rank == 0:
        _log_dir = Path(cfg.checkpoint_dir) / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_path = _log_dir / f"train_{_ts}_rank{rank}.log"
        _fh  = logging.FileHandler(_log_path, mode="w", encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(_fh)
        logger.info("Log file: %s", _log_path)

    if local_rank == 0:
        logger.info("Config:\n%s", cfg)

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = SummarizationDataset.from_json(cfg.train_path)
    val_dataset   = SummarizationDataset.from_json(cfg.val_path)

    if local_rank == 0:
        logger.info("Train: %d samples | Val: %d samples",
                    len(train_dataset), len(val_dataset))

    # ── Model ─────────────────────────────────────────────────────────────
    model = DualEncoderSummarizer.from_pretrained(
        cfg.model_name,
        hidden_size=cfg.hidden_size,
        window_overlap=cfg.window_overlap,
        max_src_len=cfg.max_src_len,
        dropout=cfg.dropout,
    )

    if cfg.gradient_checkpointing:
        # Enable gradient checkpointing ONLY on the T5 decoder stack.
        # The T5 encoders are always frozen in GAZETA_2STAGE and receive
        # integer token IDs (no requires_grad), so checkpointing them is
        # wasteful and triggers "None of the inputs have requires_grad"
        # warnings.  The decoder, however, receives encoder_hidden_states
        # from the trainable FusionLayer, so gradients DO flow through it.
        #
        # use_reentrant=False is REQUIRED for compatibility with DDP
        # find_unused_parameters=True.  Reentrant checkpointing triggers
        # nested backward passes that fire DDP autograd hooks twice for
        # the same parameter, causing "marked as ready twice" errors
        # when the decoder is unfrozen (Stage 2).
        model.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_fn = CompositeLoss(
        lambda_gen=cfg.lambda_gen,
        lambda_cover=cfg.lambda_cover,
        lambda_bert=cfg.lambda_bert,
        lambda_gate=cfg.lambda_gate,
        lambda_kw=cfg.lambda_kw,
        label_smoothing=cfg.label_smoothing,
        gate_threshold_low=cfg.gate_threshold_low,
        gate_threshold_high=cfg.gate_threshold_high,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = MISMTrainer(
        model=model,
        config=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        device=device,
        tokenizer=tokenizer,
        local_rank=local_rank,
        world_size=world_size,
    )

    # ── Determine which stages to run ─────────────────────────────────────
    stages = (1, 2) if args.stage == 0 else (args.stage,)

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume:
        # When running a specific stage (--stage 2 --resume best.pt),
        # load model weights only — optimizer and scheduler will be
        # created fresh by setup_stage().
        # When resuming the full pipeline (--stage 0), also restore
        # optimizer + scheduler state for seamless continuation.
        weights_only = (args.stage != 0)
        meta = trainer.load(args.resume, weights_only=weights_only)
        if weights_only:
            logger.info(
                "Loaded model weights from %s for Stage %d",
                args.resume, args.stage,
            )
        else:
            logger.info(
                "Resumed from %s  (epoch=%d, step=%d, stage=%d)",
                args.resume, meta["epoch"], meta["step"], meta["stage"],
            )

    # ── Train ─────────────────────────────────────────────────────────────
    result = trainer.train(stages=stages)

    if local_rank == 0:
        logger.info(
            "Training complete.  best_val_loss=%.4f  global_step=%d",
            result["best_val_loss"],
            result["global_step"],
        )

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
