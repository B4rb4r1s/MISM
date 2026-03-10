"""
trainer.py — MISMTrainer: orchestrates GAZETA_2STAGE training.

GAZETA_2STAGE
─────────────
Stage 1 (stage1_epochs):
    Frozen  : T5 encoder backbones (doc + kw), T5 decoder backbone
    Trainable: FusionLayer, KeywordAttentionLayer (KAL), shared embeddings
    LR      : stage1_lr  (default 1e-4)

Stage 2 (stage2_epochs):
    Frozen  : T5 encoder backbones only
    Trainable: T5 decoder backbone, FusionLayer, KAL, shared embeddings
    LR      : stage2_lr  (default 5e-5)

Both stages use cosine LR with linear warmup, AdamW with weight_decay,
optional bf16 autocast, and gradient accumulation.

Logged per optimiser step
─────────────────────────
train/total       — composite loss
train/l_gen       — generative (CE) component
train/l_cover     — keyword coverage component
train/l_bert      — soft BERTScore component
train/l_gate      — gate hinge component
train/grad_norm   — gradient norm *before* clipping (diagnostic)
train/lr          — current learning rate
train/gate_fusion — mean Fusion gate value (tracks keyword usage)
train/gate_kal    — mean KAL gate value

NaN / Inf detection
────────────────────
If the total loss becomes non-finite, an emergency checkpoint is saved
immediately and a RuntimeError is raised.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.collator import DataCollatorForSummarization
from src.data.dataset import SummarizationDataset
from src.losses.composite_loss import CompositeLoss
from src.models.dual_encoder_summarizer import DualEncoderSummarizer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.config import MISMConfig
from src.training.logger import MetricsLogger
from src.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


class MISMTrainer:
    """Training orchestrator for DualEncoderSummarizer.

    Parameters
    ----------
    model        : DualEncoderSummarizer instance (not yet DDP-wrapped).
    config       : MISMConfig with all hyperparameters.
    train_dataset: SummarizationDataset for training.
    val_dataset  : SummarizationDataset for validation (optional).
    loss_fn      : CompositeLoss instance.
    device       : torch.device (cuda:N or cpu).
    tokenizer    : HuggingFace tokeniser for DataCollator.
    local_rank   : DDP local rank (default 0).
    world_size   : Number of DDP processes (default 1 = single-process).
    """

    def __init__(
        self,
        model:         DualEncoderSummarizer,
        config:        MISMConfig,
        train_dataset: SummarizationDataset,
        val_dataset:   Optional[SummarizationDataset],
        loss_fn:       CompositeLoss,
        device:        torch.device,
        tokenizer,
        local_rank:    int = 0,
        world_size:    int = 1,
    ) -> None:
        self.config       = config
        self.device       = device
        self.local_rank   = local_rank
        self.world_size   = world_size
        self.is_main      = (local_rank == 0)
        self.loss_fn      = loss_fn

        self.global_step   = 0
        self.best_val_loss = float("inf")
        self.current_stage = 0
        self.optimizer:  Optional[torch.optim.AdamW] = None
        self.scheduler   = None

        # ── Collator and DataLoaders ───────────────────────────────────
        self.collator = DataCollatorForSummarization(
            tokenizer=tokenizer,
            max_kw=config.max_kw,
            kw_max_len=config.kw_max_len,
            window_size=config.window_size,
            window_overlap=config.window_overlap,
            max_windows=config.max_windows,
            max_summary_tokens=config.max_summary_tokens,
        )
        self.train_loader = self._build_dataloader(train_dataset, shuffle=True)
        self.val_loader   = (
            self._build_dataloader(val_dataset, shuffle=False)
            if val_dataset is not None else None
        )

        # ── Move model to device ──────────────────────────────────────
        model.to(device)

        # ── DDP wrapping ──────────────────────────────────────────────
        # find_unused_parameters=True is required because some parameters
        # are frozen during stage 1 (they produce no gradients).
        if world_size > 1:
            self.model: nn.Module = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        else:
            self.model = model

        # ── Metrics logger (main process only) ────────────────────────
        log_dir = Path(config.checkpoint_dir) / "logs"
        self.metrics_logger = MetricsLogger(
            log_dir=log_dir,
            config_dict=config.to_dict(),
            use_wandb=config.use_wandb,
            use_tensorboard=config.use_tensorboard,
            wandb_project=config.wandb_project,
            wandb_run_name=config.wandb_run_name,
            is_main_process=self.is_main,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_stage(self, stage: int) -> None:
        """Configure freeze/unfreeze, optimiser and scheduler for a stage.

        Must be called before the first training step of each stage.

        Parameters
        ----------
        stage : 1 = freeze encoders + decoder; 2 = freeze encoders only.
        """
        raw = self._unwrap()

        if stage == 1:
            raw.freeze_encoders()
            raw.freeze_decoder()
            lr     = self.config.stage1_lr
            epochs = self.config.stage1_epochs
        elif stage == 2:
            raw.freeze_encoders()
            raw.unfreeze_decoder()
            lr     = self.config.stage2_lr
            epochs = self.config.stage2_epochs
        else:
            raise ValueError(f"stage must be 1 or 2, got {stage}")

        self.current_stage = stage

        if self.is_main:
            counts = raw.get_trainable_param_count()
            logger.info(
                "Stage %d — trainable params: %d / %d",
                stage,
                counts["TOTAL"]["trainable"],
                counts["TOTAL"]["total"],
            )

        # Build optimiser over trainable parameters only
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=lr,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer.zero_grad()

        # Build scheduler
        steps_per_epoch = max(1, len(self.train_loader) // self.config.grad_accum_steps)
        total_steps     = epochs * steps_per_epoch
        warmup_steps    = max(1, int(total_steps * self.config.warmup_ratio))
        self.scheduler  = build_scheduler(self.optimizer, warmup_steps, total_steps)

        logger.info(
            "Stage %d scheduler: %d total steps, %d warmup steps, lr=%.2e",
            stage, total_steps, warmup_steps, lr,
        )

    def train(
        self,
        stages: tuple = (1, 2),
    ) -> Dict[str, Any]:
        """Run the GAZETA_2STAGE training procedure.

        Parameters
        ----------
        stages : tuple of ints, default (1, 2).
            Which stages to execute.  Examples:
              (1, 2) — full pipeline (default)
              (1,)   — Stage 1 only
              (2,)   — Stage 2 only (requires model weights from a Stage 1
                       checkpoint loaded via ``trainer.load(..., weights_only=True)``)

        Returns
        -------
        Dict with 'best_val_loss' and 'global_step'.
        """
        try:
            # ── Stage 1 ───────────────────────────────────────────────
            if 1 in stages:
                logger.info("═══ STAGE 1  (%d epochs) ═══", self.config.stage1_epochs)
                self.setup_stage(1)
                for epoch in range(self.config.stage1_epochs):
                    train_metrics = self.train_epoch(epoch)
                    val_metrics   = self.evaluate()
                    self._maybe_save(epoch, val_metrics)
                    # Epoch summary log
                    self.metrics_logger.log(
                        {**{"epoch_train/" + k: v for k, v in train_metrics.items()},
                         **val_metrics,
                         "epoch": float(epoch),
                         "stage": 1.0},
                        step=self.global_step,
                    )

            # ── Stage 2 ───────────────────────────────────────────────
            if 2 in stages:
                logger.info("═══ STAGE 2  (%d epochs) ═══", self.config.stage2_epochs)
                self.setup_stage(2)
                start_epoch = self.config.stage1_epochs
                for epoch in range(self.config.stage2_epochs):
                    train_metrics = self.train_epoch(start_epoch + epoch)
                    val_metrics   = self.evaluate()
                    self._maybe_save(start_epoch + epoch, val_metrics)
                    self.metrics_logger.log(
                        {**{"epoch_train/" + k: v for k, v in train_metrics.items()},
                         **val_metrics,
                         "epoch": float(start_epoch + epoch),
                         "stage": 2.0},
                        step=self.global_step,
                    )

        finally:
            # Always close logger — even on error / KeyboardInterrupt
            self.metrics_logger.close()

        return {
            "best_val_loss": self.best_val_loss,
            "global_step":   self.global_step,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Parameters
        ----------
        epoch : zero-indexed epoch number (used only for logging).

        Returns
        -------
        Dict of averaged training metrics for this epoch.
        """
        self.model.train()

        # Set epoch on DistributedSampler so each rank shuffles differently
        if self.world_size > 1:
            sampler = self.train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        accum:    Dict[str, float] = defaultdict(float)
        n_updates = 0
        latest_grad_norm = 0.0     # captures the most-recent pre-clip norm
        latest_lr        = 0.0

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        amp_dtype   = torch.bfloat16 if (self.config.bf16 and device_type == "cuda") else None
        use_amp     = amp_dtype is not None

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            # ── Input NaN check — ALL ranks, first batch only ─────────
            # Catches bad data (NaN/Inf kw_scores, etc.) before the forward pass.
            if batch_idx == 0:
                for _bkey, _bval in batch.items():
                    if isinstance(_bval, torch.Tensor) and _bval.is_floating_point():
                        if not _bval.isfinite().all():
                            logger.error(
                                "[rank=%d epoch=%d] Non-finite input data: "
                                "batch['%s'] has_nan=%s has_inf=%s",
                                self.local_rank, epoch, _bkey,
                                _bval.isnan().any().item(),
                                _bval.isinf().any().item(),
                            )

            # ── Forward ──────────────────────────────────────────────
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype,
                                    enabled=use_amp):
                output = self.model(
                    input_windows=batch["input_windows"],
                    window_attention_mask=batch["window_attention_mask"],
                    kw_input_ids=batch["kw_input_ids"],
                    kw_attention_mask=batch["kw_attention_mask"],
                    kw_scores=batch["kw_scores"],
                    kw_mask=batch["kw_mask"],
                    labels=batch["labels"],
                )

            # ── Loss in float32 (outside autocast for numerical stability) ──
            emb = self._unwrap().shared.weight  # [V, D]
            total_loss, components = self.loss_fn(
                logits=output.logits.float() if use_amp else output.logits,
                labels=batch["labels"],
                embedding_matrix=emb.float() if use_amp else emb,
                kw_attn_weights=output.kw_attn_weights,
                kw_scores=batch["kw_scores"],
                kw_mask=batch["kw_mask"],
                fusion_gate_values=output.fusion_gate_values,
                kal_gate_values=output.kal_gate_values,
                kw_input_ids=batch.get("kw_input_ids"),
            )

            # ── NaN / Inf guard ───────────────────────────────────────
            if not total_loss.isfinite():
                logger.error(
                    "Non-finite loss (%.6g) at step %d epoch %d batch %d — "
                    "components: l_gen=%.4g l_cover=%.4g l_bert=%.4g "
                    "l_gate=%.4g l_kw=%.4g",
                    total_loss.item(), self.global_step, epoch, batch_idx,
                    components.get("l_gen",   float("nan")),
                    components.get("l_cover", float("nan")),
                    components.get("l_bert",  float("nan")),
                    components.get("l_gate",  float("nan")),
                    components.get("l_kw",    float("nan")),
                )
                for _name, _t in [
                    ("logits",             output.logits),
                    ("kw_attn_weights",    output.kw_attn_weights),
                    ("fusion_gate_values", output.fusion_gate_values),
                    ("kal_gate_values",    output.kal_gate_values),
                ]:
                    if not _t.isfinite().all():
                        logger.error(
                            "  output.%s: has_nan=%s has_inf=%s",
                            _name,
                            _t.isnan().any().item(),
                            _t.isinf().any().item(),
                        )
                if self.is_main:
                    self.save(
                        Path(self.config.checkpoint_dir) / "emergency.pt",
                        epoch=epoch, metrics={},
                    )
                raise RuntimeError(
                    f"Non-finite loss: {total_loss.item():.6g} "
                    f"at step {self.global_step}"
                )

            # ── Scaled backward ───────────────────────────────────────
            scaled = total_loss / self.config.grad_accum_steps
            scaled.backward()

            # Accumulate loss and gate stats (per micro-step)
            for k, v in components.items():
                accum[k] += v
            accum["total"]      += total_loss.item()
            accum["gate_fusion"] += output.fusion_gate_values.detach().mean().item()
            accum["gate_kal"]    += output.kal_gate_values.detach().mean().item()

            # ── Optimiser step (every grad_accum_steps micro-steps) ───
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                trainable_params = [p for p in self.model.parameters()
                                    if p.requires_grad]
                # clip_grad_norm_ returns the norm *before* clipping
                raw_norm = nn.utils.clip_grad_norm_(
                    trainable_params, self.config.max_grad_norm,
                )
                latest_grad_norm = (
                    raw_norm.item() if hasattr(raw_norm, "item") else float(raw_norm)
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                n_updates        += 1
                latest_lr = self.optimizer.param_groups[0]["lr"]

                if self.is_main and self.global_step % self.config.log_every_steps == 0:
                    denom     = max(1, n_updates)
                    step_metrics: Dict[str, float] = {
                        "train/" + k: v / denom for k, v in accum.items()
                    }
                    # Non-averaged diagnostics
                    step_metrics["train/grad_norm"] = latest_grad_norm
                    step_metrics["train/lr"]        = latest_lr
                    self.metrics_logger.log(step_metrics, step=self.global_step)

        denom = max(1, n_updates)
        return {k: v / denom for k, v in accum.items()}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation loop.

        Returns
        -------
        Dict with val/* metrics (loss + gate stats) or {} if no val data.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        accum:    Dict[str, float] = defaultdict(float)
        n_batches = 0

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        amp_dtype   = torch.bfloat16 if (self.config.bf16 and device_type == "cuda") else None
        use_amp     = amp_dtype is not None

        for batch in self.val_loader:
            batch = self._to_device(batch)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype,
                                    enabled=use_amp):
                output = self.model(
                    input_windows=batch["input_windows"],
                    window_attention_mask=batch["window_attention_mask"],
                    kw_input_ids=batch["kw_input_ids"],
                    kw_attention_mask=batch["kw_attention_mask"],
                    kw_scores=batch["kw_scores"],
                    kw_mask=batch["kw_mask"],
                    labels=batch["labels"],
                )
            emb = self._unwrap().shared.weight
            total_loss, components = self.loss_fn(
                logits=output.logits.float() if use_amp else output.logits,
                labels=batch["labels"],
                embedding_matrix=emb.float() if use_amp else emb,
                kw_attn_weights=output.kw_attn_weights,
                kw_scores=batch["kw_scores"],
                kw_mask=batch["kw_mask"],
                fusion_gate_values=output.fusion_gate_values,
                kal_gate_values=output.kal_gate_values,
                kw_input_ids=batch.get("kw_input_ids"),
            )
            for k, v in components.items():
                accum[k] += v
            accum["total"]      += total_loss.item()
            accum["gate_fusion"] += output.fusion_gate_values.mean().item()
            accum["gate_kal"]    += output.kal_gate_values.mean().item()
            n_batches += 1

        denom   = max(1, n_batches)
        metrics = {"val/" + k: v / denom for k, v in accum.items()}

        if self.is_main:
            self.metrics_logger.log(metrics, step=self.global_step)
        return metrics

    def save(
        self,
        path:    Union[str, Path],
        epoch:   int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save a checkpoint (main process only)."""
        if not self.is_main:
            return
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            step=self.global_step,
            stage=self.current_stage,
            metrics=metrics,
            config_dict=self.config.to_dict(),
        )

    def load(
        self,
        path: Union[str, Path],
        weights_only: bool = False,
    ) -> Dict[str, Any]:
        """Load a checkpoint and restore trainer state.

        Parameters
        ----------
        path         : path to the .pt checkpoint file.
        weights_only : if True, only restore model weights — skip optimizer,
                       scheduler, and step/stage counters.  Use this when
                       loading a Stage 1 checkpoint to start Stage 2 with a
                       fresh optimizer (``--stage 2 --resume best.pt``).
        """
        meta = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=None if weights_only else self.optimizer,
            scheduler=None if weights_only else self.scheduler,
        )
        if not weights_only:
            self.global_step   = meta["step"]
            self.current_stage = meta["stage"]
        return meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unwrap(self) -> DualEncoderSummarizer:
        """Return the underlying model (unwrap DDP if necessary)."""
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_dataloader(
        self,
        dataset: SummarizationDataset,
        shuffle: bool,
    ) -> DataLoader:
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=shuffle,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(self.device.type == "cuda"),
        )

    def _to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _maybe_save(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Save best and periodic checkpoints (main process only)."""
        if not self.is_main:
            return
        ckpt_dir = Path(self.config.checkpoint_dir)

        # Periodic save
        if self.global_step > 0 and self.global_step % self.config.save_every_steps == 0:
            self.save(ckpt_dir / f"step_{self.global_step:07d}.pt", epoch, val_metrics)

        # Best save (by total val loss)
        val_loss = val_metrics.get("val/total", float("inf"))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save(ckpt_dir / "best.pt", epoch, val_metrics)
            logger.info(
                "New best val/total=%.4f → %s/best.pt",
                val_loss, ckpt_dir,
            )
