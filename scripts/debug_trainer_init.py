#!/usr/bin/env python
"""
debug_trainer_init.py — Diagnose hang in MISMTrainer initialization.

Replicates the exact flow of train.py step-by-step with flush prints
to pinpoint which line causes the hang.

Usage (on the DGX2 server, single GPU):
    python -u scripts/debug_trainer_init.py --config configs/gazeta_2stage.yaml

If a specific step hangs, check the last [STEP N] printed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def ts() -> str:
    return time.strftime("%H:%M:%S")


def step(n: int, msg: str) -> None:
    print(f"[{ts()}] [STEP {n:02d}] {msg}", flush=True)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()

    # ── Step 1: imports ───────────────────────────────────────────────
    step(1, "Importing torch ...")
    import torch
    step(1, f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    # ── Step 2: device ────────────────────────────────────────────────
    step(2, "Setting device ...")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    step(2, f"  device = {device}")

    # ── Step 3: config ────────────────────────────────────────────────
    step(3, "Loading config ...")
    from src.training.config import load_config
    overrides = {}
    for pair in args.set:
        if "=" in pair:
            k, v = pair.split("=", 1)
            overrides[k.strip()] = v.strip()
    cfg = load_config(args.config, overrides=overrides)
    step(3, f"  model_name = {cfg.model_name}")
    step(3, f"  batch_size = {cfg.batch_size}")
    step(3, f"  bf16 = {cfg.bf16}")
    step(3, f"  num_workers would be = {min(4, os.cpu_count() or 1)}")

    torch.manual_seed(cfg.seed)

    # ── Step 4: tokenizer ─────────────────────────────────────────────
    step(4, f"Loading tokenizer from {cfg.model_name} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    step(4, f"  tokenizer loaded, vocab_size = {tokenizer.vocab_size}")

    # ── Step 5: datasets ──────────────────────────────────────────────
    step(5, f"Loading train dataset from {cfg.train_path} ...")
    from src.data.dataset import SummarizationDataset
    train_dataset = SummarizationDataset.from_json(cfg.train_path)
    step(5, f"  train: {len(train_dataset)} samples")
    step(5, f"Loading val dataset from {cfg.val_path} ...")
    val_dataset = SummarizationDataset.from_json(cfg.val_path)
    step(5, f"  val: {len(val_dataset)} samples")

    # ── Step 6: model ─────────────────────────────────────────────────
    step(6, "Creating DualEncoderSummarizer.from_pretrained ...")
    from src.models.dual_encoder_summarizer import DualEncoderSummarizer
    model = DualEncoderSummarizer.from_pretrained(
        cfg.model_name,
        hidden_size=cfg.hidden_size,
        window_overlap=cfg.window_overlap,
        max_src_len=cfg.max_src_len,
        dropout=cfg.dropout,
    )
    step(6, "  model created OK")

    # ── Step 7: gradient checkpointing ────────────────────────────────
    step(7, f"Gradient checkpointing = {cfg.gradient_checkpointing}")
    if cfg.gradient_checkpointing:
        model.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        step(7, "  gradient checkpointing enabled on decoder")
    else:
        step(7, "  skipped")

    # ── Step 8: CompositeLoss ─────────────────────────────────────────
    step(8, "Creating CompositeLoss ...")
    from src.losses.composite_loss import CompositeLoss
    loss_fn = CompositeLoss(
        lambda_gen=cfg.lambda_gen,
        lambda_cover=cfg.lambda_cover,
        lambda_bert=cfg.lambda_bert,
        lambda_gate=cfg.lambda_gate,
        label_smoothing=cfg.label_smoothing,
        gate_threshold_low=cfg.gate_threshold_low,
        gate_threshold_high=cfg.gate_threshold_high,
    )
    step(8, "  CompositeLoss created OK")

    # ══════════════════════════════════════════════════════════════════
    # Now replicate MISMTrainer.__init__ step by step
    # ══════════════════════════════════════════════════════════════════

    print(f"\n[{ts()}] ═══ Replicating MISMTrainer.__init__ ═══\n", flush=True)

    # ── Step 9: DataCollatorForSummarization ──────────────────────────
    step(9, "Creating DataCollatorForSummarization ...")
    from src.data.collator import DataCollatorForSummarization
    collator = DataCollatorForSummarization(
        tokenizer=tokenizer,
        max_kw=cfg.max_kw,
        kw_max_len=cfg.kw_max_len,
        window_size=cfg.window_size,
        window_overlap=cfg.window_overlap,
        max_windows=cfg.max_windows,
        max_summary_tokens=cfg.max_summary_tokens,
    )
    step(9, "  collator created OK")

    # ── Step 10: Test collator on a single sample ─────────────────────
    step(10, "Testing collator on first train sample ...")
    sample = train_dataset[0]
    step(10, f"  sample keys: {list(sample.keys())}")
    batch = collator([sample])
    step(10, f"  batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            step(10, f"    {k}: {v.shape} {v.dtype}")

    # ── Step 11: DataLoader creation (train) ──────────────────────────
    num_workers_val = min(4, os.cpu_count() or 1)
    step(11, f"Creating train DataLoader (num_workers={num_workers_val}) ...")
    from torch.utils.data import DataLoader

    t0 = time.time()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers_val,
        pin_memory=True,
    )
    step(11, f"  train DataLoader created in {time.time()-t0:.2f}s")

    # ── Step 12: DataLoader creation (val) ────────────────────────────
    step(12, f"Creating val DataLoader (num_workers={num_workers_val}) ...")
    t0 = time.time()
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers_val,
        pin_memory=True,
    )
    step(12, f"  val DataLoader created in {time.time()-t0:.2f}s")

    # ── Step 13: model.to(device) ─────────────────────────────────────
    step(13, f"Moving model to {device} ...")
    t0 = time.time()
    model.to(device)
    step(13, f"  model.to({device}) done in {time.time()-t0:.2f}s")
    if device.type == "cuda":
        mem = torch.cuda.memory_allocated(device) / 1024**2
        step(13, f"  GPU memory allocated: {mem:.0f} MB")

    # ── Step 14: MetricsLogger ────────────────────────────────────────
    step(14, "Creating MetricsLogger ...")
    from src.training.logger import MetricsLogger

    log_dir = Path(cfg.checkpoint_dir) / "logs"
    t0 = time.time()
    metrics_logger = MetricsLogger(
        log_dir=log_dir,
        config_dict=cfg.to_dict(),
        use_wandb=cfg.use_wandb,
        use_tensorboard=cfg.use_tensorboard,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
        is_main_process=True,
    )
    step(14, f"  MetricsLogger created in {time.time()-t0:.2f}s")

    # ── Step 15: Iterate first batch from DataLoader ──────────────────
    step(15, "Fetching first batch from train DataLoader ...")
    t0 = time.time()
    for first_batch in train_loader:
        elapsed = time.time() - t0
        step(15, f"  first batch fetched in {elapsed:.2f}s")
        for k, v in first_batch.items():
            if hasattr(v, 'shape'):
                step(15, f"    {k}: {v.shape} {v.dtype}")
        break

    # ── Step 16: Move batch to device ─────────────────────────────────
    step(16, "Moving batch to device ...")
    dev_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in first_batch.items()
    }
    step(16, "  batch on device OK")

    # ── Step 17: Forward pass ─────────────────────────────────────────
    step(17, "Running forward pass ...")
    model.eval()
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if cfg.bf16 else None
        use_amp = amp_dtype is not None
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            output = model(
                input_windows=dev_batch["input_windows"],
                window_attention_mask=dev_batch["window_attention_mask"],
                kw_input_ids=dev_batch["kw_input_ids"],
                kw_attention_mask=dev_batch["kw_attention_mask"],
                kw_scores=dev_batch["kw_scores"],
                kw_mask=dev_batch["kw_mask"],
                labels=dev_batch["labels"],
            )
    step(17, f"  forward pass OK, logits shape = {output.logits.shape}")

    # ── Step 18: Cleanup ──────────────────────────────────────────────
    step(18, "Cleaning up MetricsLogger ...")
    metrics_logger.close()
    step(18, "  done")

    # ── Step 19: Now test with num_workers=0 ──────────────────────────
    step(19, "Re-testing DataLoader with num_workers=0 ...")
    train_loader_0 = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    t0 = time.time()
    for batch0 in train_loader_0:
        step(19, f"  num_workers=0: first batch in {time.time()-t0:.2f}s")
        break
    step(19, "  num_workers=0 test OK")

    print(f"\n[{ts()}] ═══ ALL STEPS PASSED — trainer init is healthy ═══", flush=True)
    print(f"\nIf train.py still hangs, the issue is likely:", flush=True)
    print(f"  1. DDP-specific (try --stage 1 with single GPU: python scripts/train.py ...)", flush=True)
    print(f"  2. Zombie processes from prior torchrun (check: ps aux | grep train.py)", flush=True)
    print(f"  3. NCCL timeout during DDP init (check NCCL_DEBUG=INFO)", flush=True)


if __name__ == "__main__":
    main()
