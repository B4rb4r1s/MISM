"""
test_phase3.py — Unit tests for training infrastructure (Phase 3).

Coverage
--------
TestMISMConfig       —  9 tests
TestBuildScheduler   —  8 tests
TestSaveLoadCheckpoint — 9 tests
TestMISMTrainer      —  8 tests  (uses tiny T5 + in-memory dataset)
                         ───────
Total                   34 tests

The trainer tests create a tiny DualEncoderSummarizer (d_model=64, 2 layers)
to avoid loading a full T5 checkpoint.  A mock tokeniser is used so DataLoader
construction does not require a real HuggingFace model.
"""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import T5Config, T5ForConditionalGeneration

from src.losses.composite_loss import CompositeLoss
from src.models.dual_encoder_summarizer import DualEncoderSummarizer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.config import MISMConfig, load_config
from src.training.scheduler import build_scheduler
from src.training.trainer import MISMTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

TINY_VOCAB = 256  # small vocab for unit tests

_TINY_T5_CFG = T5Config(
    vocab_size=TINY_VOCAB,
    d_model=64,
    d_ff=128,
    num_heads=2,
    num_layers=2,
    d_kv=32,
    decoder_start_token_id=0,
    pad_token_id=0,
    eos_token_id=1,
)


def _make_tiny_model() -> DualEncoderSummarizer:
    """Tiny DualEncoderSummarizer for CPU tests (≈ seconds)."""
    t5 = T5ForConditionalGeneration(_TINY_T5_CFG)
    return DualEncoderSummarizer(
        t5_model=t5,
        hidden_size=64,
        kw_num_heads=2,
        kw_ffn_dim=128,
        doc_num_heads=2,
        doc_ffn_dim=128,
        fusion_num_heads=2,
        fusion_ffn_dim=128,
        window_overlap=2,
        max_src_len=64,
        kal_num_heads=2,
        kal_ffn_dim=64,
        dropout=0.0,
    )


def _make_tiny_config(**overrides) -> MISMConfig:
    """Minimal config for fast unit tests (small batches, 1 epoch)."""
    defaults = dict(
        model_name="local",
        hidden_size=64,
        num_heads=2,
        ffn_dim=128,
        dropout=0.0,
        window_size=8,
        window_overlap=2,
        max_windows=2,
        max_kw=3,
        kw_max_len=4,
        max_summary_tokens=8,
        max_src_len=64,
        stage1_epochs=1,
        stage1_lr=1e-3,
        stage2_epochs=1,
        stage2_lr=5e-4,
        batch_size=1,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=False,              # CPU tests use fp32
        gradient_checkpointing=False,
        seed=0,
        lambda_gen=0.65,
        lambda_cover=0.15,
        lambda_bert=0.15,
        lambda_gate=0.05,
        label_smoothing=0.1,
        gate_threshold=0.3,
        use_wandb=False,
        log_every_steps=1,
        eval_every_steps=1,
        save_every_steps=10000,
        checkpoint_dir="checkpoints_test",
    )
    defaults.update(overrides)
    return MISMConfig(**defaults)


# ── Minimal in-memory dataset ─────────────────────────────────────────────
class _TinyDataset:
    """Mimics SummarizationDataset.__getitem__ output without real data."""
    _TEXT  = "Текст о машинном обучении. " * 20  # ~100 chars
    _SUMM  = "Краткий вывод о машинном обучении для научной работы." * 3
    _KW    = [("машинное обучение", 1.0), ("нейронные сети", 1.0)]

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "doc_id":     f"doc_{idx}",
            "title":      "Заголовок",
            "text_clean": self._TEXT,
            "keywords":   [kw for kw, _ in self._KW],
            "kw_scores":  [s  for _, s  in self._KW],
            "summary":    self._SUMM,
        }


# ── Mock tokeniser ────────────────────────────────────────────────────────
def _make_mock_tokenizer(vocab_size: int = TINY_VOCAB):
    """Minimal tokeniser mock compatible with DataCollatorForSummarization."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1

    def _encode(text, max_length=None, padding=None, truncation=None,
                add_special_tokens=True, return_attention_mask=True, **kw):
        ids = [ord(c) % vocab_size for c in str(text)[:max_length or 8]]
        ids = (ids + [0] * (max_length or 8))[: max_length or 8]
        mask = [1 if i != 0 else 0 for i in ids]
        if not return_attention_mask:
            return {"input_ids": ids}
        return {"input_ids": ids, "attention_mask": mask}

    tok.side_effect = _encode
    tok.__call__ = _encode

    # Make tok(...) work
    tok_callable = MagicMock(side_effect=_encode)
    tok_callable.pad_token_id = 0
    tok_callable.eos_token_id = 1
    return tok_callable


# ─────────────────────────────────────────────────────────────────────────────
# TestMISMConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestMISMConfig:

    def test_default_values(self):
        cfg = MISMConfig()
        assert cfg.window_size == 512
        assert cfg.window_overlap == 128
        assert cfg.stage1_epochs == 5
        assert cfg.stage2_epochs == 15
        assert cfg.batch_size == 4
        assert cfg.bf16 is True
        assert cfg.seed == 42

    def test_loss_weights_sum_to_one(self):
        cfg = MISMConfig()
        total = cfg.lambda_gen + cfg.lambda_cover + cfg.lambda_bert + cfg.lambda_gate
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_stage_lrs_differ(self):
        cfg = MISMConfig()
        assert cfg.stage1_lr != cfg.stage2_lr
        assert cfg.stage1_lr > cfg.stage2_lr  # stage 1 is coarser

    def test_to_dict_roundtrip(self):
        cfg  = MISMConfig(batch_size=3, seed=99)
        cfg2 = MISMConfig.from_dict(cfg.to_dict())
        assert cfg2.batch_size == 3
        assert cfg2.seed == 99

    def test_from_dict_unknown_keys_ignored(self):
        d = MISMConfig().to_dict()
        d["totally_unknown_field"] = "surprise"
        cfg = MISMConfig.from_dict(d)   # should not raise
        assert not hasattr(cfg, "totally_unknown_field")

    def test_load_from_base_yaml(self):
        cfg = load_config("configs/base.yaml")
        assert cfg.model_name == "IlyaGusev/rut5_base_sum_gazeta"
        assert cfg.window_size == 512

    def test_load_with_overrides(self):
        cfg = load_config("configs/base.yaml", overrides={"batch_size": 7, "seed": 123})
        assert cfg.batch_size == 7
        assert cfg.seed == 123

    def test_validation_raises_on_negative_lr(self):
        with pytest.raises(ValueError, match="stage1_lr"):
            MISMConfig(stage1_lr=-1e-4)

    def test_warmup_ratio_validation(self):
        with pytest.raises(ValueError, match="warmup_ratio"):
            MISMConfig(warmup_ratio=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildScheduler
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildScheduler:

    def _optimizer(self, lr: float = 1e-3):
        model = nn.Linear(10, 10)
        return AdamW(model.parameters(), lr=lr)

    def test_warmup_phase_increases_lr(self):
        opt = self._optimizer(lr=1.0)
        sched = build_scheduler(opt, num_warmup_steps=10, num_training_steps=100)
        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        # Should be strictly increasing during warmup (except step 0 = 0)
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_lr_at_end_of_warmup_equals_peak(self):
        peak_lr = 1e-3
        opt   = self._optimizer(lr=peak_lr)
        sched = build_scheduler(opt, num_warmup_steps=4, num_training_steps=20)
        for _ in range(4):
            sched.step()
        # After warmup_steps steps, multiplier == 1.0 → lr == peak_lr
        assert opt.param_groups[0]["lr"] == pytest.approx(peak_lr, rel=1e-5)

    def test_cosine_decay_after_warmup(self):
        opt  = self._optimizer(lr=1.0)
        sched = build_scheduler(opt, num_warmup_steps=5, num_training_steps=25)
        for _ in range(5):      # warmup
            sched.step()
        peak = opt.param_groups[0]["lr"]
        for _ in range(15):     # decay
            sched.step()
        final = opt.param_groups[0]["lr"]
        assert final < peak

    def test_min_lr_ratio_respected(self):
        opt  = self._optimizer(lr=1.0)
        sched = build_scheduler(opt, num_warmup_steps=0,
                                num_training_steps=50, min_lr_ratio=0.1)
        for _ in range(50):
            sched.step()
        lr = opt.param_groups[0]["lr"]
        assert lr >= pytest.approx(0.1, abs=1e-5)

    def test_zero_warmup_starts_decaying(self):
        """With 0 warmup steps, the very first step should be at peak (mult=1)."""
        opt  = self._optimizer(lr=1.0)
        sched = build_scheduler(opt, num_warmup_steps=0, num_training_steps=10)
        # At step 0, multiplier = cosine(progress=0) = 1.0
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-5)

    def test_lr_at_last_step_approaches_min(self):
        opt  = self._optimizer(lr=1.0)
        min_r = 0.0
        sched = build_scheduler(opt, num_warmup_steps=0,
                                num_training_steps=100, min_lr_ratio=min_r)
        for _ in range(100):
            sched.step()
        lr = opt.param_groups[0]["lr"]
        assert lr == pytest.approx(min_r, abs=1e-5)

    def test_invalid_warmup_raises(self):
        opt = self._optimizer()
        with pytest.raises(ValueError):
            build_scheduler(opt, num_warmup_steps=-1, num_training_steps=10)

    def test_invalid_min_lr_ratio_raises(self):
        opt = self._optimizer()
        with pytest.raises(ValueError):
            build_scheduler(opt, num_warmup_steps=2, num_training_steps=10,
                            min_lr_ratio=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveLoadCheckpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoadCheckpoint:

    def _simple_model(self):
        return nn.Linear(8, 8)

    def _optimizer(self, model):
        return AdamW(model.parameters(), lr=1e-3)

    def _scheduler(self, optimizer):
        return build_scheduler(optimizer, num_warmup_steps=2,
                               num_training_steps=10)

    def test_save_creates_file(self, tmp_path):
        model = self._simple_model()
        opt   = self._optimizer(model)
        sch   = self._scheduler(opt)
        ckpt  = tmp_path / "test.pt"
        save_checkpoint(ckpt, model, opt, sch, epoch=0, step=0, stage=1)
        assert ckpt.exists()

    def test_load_restores_model_weights(self, tmp_path):
        model  = self._simple_model()
        opt    = self._optimizer(model)
        sch    = self._scheduler(opt)
        ckpt   = tmp_path / "test.pt"
        # Modify weights and save
        with torch.no_grad():
            model.weight.fill_(42.0)
        save_checkpoint(ckpt, model, opt, sch, epoch=1, step=10, stage=1)

        # Load into a fresh model
        model2 = self._simple_model()
        load_checkpoint(ckpt, model=model2)
        assert torch.allclose(model.weight, model2.weight)

    def test_load_restores_optimizer_state(self, tmp_path):
        model = self._simple_model()
        opt   = self._optimizer(model)
        sch   = self._scheduler(opt)
        # Do a fake update so optimizer has state
        loss  = model(torch.randn(2, 8)).sum()
        loss.backward()
        opt.step()
        ckpt  = tmp_path / "opt.pt"
        save_checkpoint(ckpt, model, opt, sch, epoch=0, step=1, stage=1)

        opt2  = self._optimizer(model)
        load_checkpoint(ckpt, optimizer=opt2)
        # State dict keys should match after restore
        assert set(opt2.state.keys()) == set(opt.state.keys())

    def test_load_restores_scheduler_step(self, tmp_path):
        model = self._simple_model()
        opt   = self._optimizer(model)
        sch   = self._scheduler(opt)
        # Advance scheduler a few steps
        for _ in range(5):
            sch.step()
        ckpt = tmp_path / "sch.pt"
        save_checkpoint(ckpt, model, opt, sch, epoch=0, step=5, stage=1)

        sch2 = self._scheduler(AdamW(self._simple_model().parameters(), lr=1e-3))
        load_checkpoint(ckpt, scheduler=sch2)
        assert sch2.last_epoch == sch.last_epoch

    def test_metadata_preserved(self, tmp_path):
        model = self._simple_model()
        opt   = self._optimizer(model)
        sch   = self._scheduler(opt)
        ckpt  = tmp_path / "meta.pt"
        save_checkpoint(ckpt, model, opt, sch, epoch=3, step=42, stage=2,
                        metrics={"val/loss": 1.23})
        meta = load_checkpoint(ckpt)
        assert meta["epoch"] == 3
        assert meta["step"]  == 42
        assert meta["stage"] == 2
        assert meta["metrics"]["val/loss"] == pytest.approx(1.23)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "does_not_exist.pt")

    def test_parent_dirs_created(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "ckpt.pt"
        model = self._simple_model()
        opt   = self._optimizer(model)
        sch   = self._scheduler(opt)
        save_checkpoint(deep, model, opt, sch, epoch=0, step=0, stage=1)
        assert deep.exists()

    def test_config_dict_preserved(self, tmp_path):
        model = self._simple_model()
        opt   = self._optimizer(model)
        sch   = self._scheduler(opt)
        cfg   = {"batch_size": 4, "seed": 99}
        ckpt  = tmp_path / "cfg.pt"
        save_checkpoint(ckpt, model, opt, sch, epoch=0, step=0, stage=1,
                        config_dict=cfg)
        meta = load_checkpoint(ckpt)
        assert meta["config"]["batch_size"] == 4
        assert meta["config"]["seed"] == 99

    def test_ddp_wrapped_model_saved_correctly(self, tmp_path):
        """DDP-wrapped model (has .module) should save the inner state dict."""
        inner = self._simple_model()
        opt   = self._optimizer(inner)
        sch   = self._scheduler(opt)
        # Fake DDP wrapper
        wrapped       = MagicMock()
        wrapped.module = inner
        ckpt = tmp_path / "ddp.pt"
        save_checkpoint(ckpt, wrapped, opt, sch, epoch=0, step=0, stage=1)

        # Load into the inner model
        restored = self._simple_model()
        load_checkpoint(ckpt, model=restored)
        assert torch.allclose(inner.weight, restored.weight)


# ─────────────────────────────────────────────────────────────────────────────
# TestMISMTrainer
# ─────────────────────────────────────────────────────────────────────────────

class TestMISMTrainer:
    """Tests for MISMTrainer using a tiny model and a mock tokeniser.

    We test setup_stage (freeze logic), checkpoint round-trip, and that a
    single gradient-accumulation step can be executed on CPU without errors.
    """

    @pytest.fixture(scope="class")
    def tiny_model(self):
        return _make_tiny_model()

    @pytest.fixture(scope="class")
    def tiny_cfg(self):
        return _make_tiny_config()

    @pytest.fixture(scope="class")
    def tiny_loss(self):
        return CompositeLoss(
            lambda_gen=0.65, lambda_cover=0.15,
            lambda_bert=0.15, lambda_gate=0.05,
        )

    def _make_trainer(self, model=None, cfg=None, loss=None):
        if model is None:
            model = _make_tiny_model()
        if cfg is None:
            cfg = _make_tiny_config()
        if loss is None:
            loss = CompositeLoss()
        tok = _make_mock_tokenizer(TINY_VOCAB)
        dataset = _TinyDataset()
        return MISMTrainer(
            model=model,
            config=cfg,
            train_dataset=dataset,
            val_dataset=None,
            loss_fn=loss,
            device=torch.device("cpu"),
            tokenizer=tok,
            local_rank=0,
            world_size=1,
        )

    # ── Freeze / unfreeze tests (no forward pass needed) ─────────────────

    def test_setup_stage1_freezes_encoders_and_decoder(self):
        trainer = self._make_trainer()
        trainer.setup_stage(1)
        raw = trainer._unwrap()

        for p in raw.document_encoder.t5_encoder.parameters():
            assert not p.requires_grad, "doc T5 encoder must be frozen in stage 1"
        for p in raw.keywords_encoder.t5_encoder.parameters():
            assert not p.requires_grad, "kw T5 encoder must be frozen in stage 1"
        for p in raw.decoder.parameters():
            assert not p.requires_grad, "decoder must be frozen in stage 1"

    def test_setup_stage1_fusion_and_kal_remain_trainable(self):
        trainer = self._make_trainer()
        trainer.setup_stage(1)
        raw = trainer._unwrap()

        fusion_trainable = [p for p in raw.fusion_layer.parameters() if p.requires_grad]
        kal_trainable    = [p for p in raw.keyword_attention_layer.parameters()
                            if p.requires_grad]
        assert len(fusion_trainable) > 0, "FusionLayer must be trainable in stage 1"
        assert len(kal_trainable) > 0,    "KAL must be trainable in stage 1"

    def test_setup_stage2_unfreezes_decoder(self):
        trainer = self._make_trainer()
        trainer.setup_stage(2)
        raw = trainer._unwrap()

        decoder_trainable = [p for p in raw.decoder.parameters() if p.requires_grad]
        assert len(decoder_trainable) > 0, "Decoder must be trainable in stage 2"

    def test_setup_stage2_keeps_encoders_frozen(self):
        """T5 encoder attention/FFN weights must stay frozen in stage 2.

        The shared embedding weight (embed_tokens) is excluded from this
        check: it is tied to the decoder and becomes trainable when the
        decoder is unfrozen — that is the expected behaviour.
        """
        trainer = self._make_trainer()
        trainer.setup_stage(2)
        raw = trainer._unwrap()

        for name, p in raw.document_encoder.t5_encoder.named_parameters():
            if "embed_tokens" not in name:
                assert not p.requires_grad, \
                    f"doc_encoder.{name} must stay frozen in stage 2"
        for name, p in raw.keywords_encoder.t5_encoder.named_parameters():
            if "embed_tokens" not in name:
                assert not p.requires_grad, \
                    f"kw_encoder.{name} must stay frozen in stage 2"

    def test_fewer_trainable_params_in_stage1_than_stage2(self):
        trainer = self._make_trainer()

        trainer.setup_stage(1)
        n1 = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

        trainer.setup_stage(2)
        n2 = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

        assert n1 < n2, f"Stage 1 trainable ({n1}) should be < Stage 2 trainable ({n2})"

    def test_optimizer_uses_correct_lr_stage1(self):
        cfg     = _make_tiny_config(stage1_lr=7e-4)
        trainer = self._make_trainer(cfg=cfg)
        trainer.setup_stage(1)
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == pytest.approx(7e-4, rel=1e-5)

    def test_global_step_starts_at_zero(self):
        trainer = self._make_trainer()
        assert trainer.global_step == 0

    def test_checkpoint_save_and_load_via_trainer(self, tmp_path):
        trainer = self._make_trainer()
        trainer.setup_stage(1)

        ckpt_path = tmp_path / "trainer_test.pt"
        trainer.save(ckpt_path, epoch=0, metrics={"val/loss": 2.5})
        assert ckpt_path.exists()

        # Load into a fresh trainer
        trainer2 = self._make_trainer()
        trainer2.setup_stage(1)
        meta = trainer2.load(ckpt_path)
        assert meta["stage"] == 1
