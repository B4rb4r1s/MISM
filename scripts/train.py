#!/usr/bin/env python3
"""
train.py — Обучение LLM-суммаризатора с LoRA.

Запуск (single GPU):
    python scripts/train.py --config configs/qwen_lora.yaml

Запуск (multi-GPU с DeepSpeed):
    torchrun --nproc_per_node=6 scripts/train.py --config configs/qwen_lora.yaml

Override параметров:
    torchrun --nproc_per_node=6 scripts/train.py \
        --config configs/qwen_lora.yaml \
        --set training.learning_rate=1e-4 data.max_keywords=30 training.epochs=5
"""

import argparse
import logging
import os
import sys

# Корень проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import TrainingArguments, Trainer

from src.training.config import load_config
from src.models.lora_setup import load_tokenizer, load_model_for_training
from src.data.dataset import (
    SummarizationDataset,
    DataCollatorWithPadding,
    load_data,
    train_val_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_overrides(args_list):
    """Парсит --set key=value key2=value2 в словарь."""
    overrides = {}
    if not args_list:
        return overrides
    for item in args_list:
        if "=" in item:
            key, val = item.split("=", 1)
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            overrides[key] = val
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Train LLM summarizer with LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--set", nargs="*", default=[], help="Override config: key=value")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = load_config(args.config, overrides=overrides)

    logger.info("=== Конфигурация ===")
    logger.info("Модель: %s", cfg.model.name)
    logger.info("LoRA r=%d, alpha=%d", cfg.lora.r, cfg.lora.alpha)
    logger.info("Max seq len: %d", cfg.model.max_seq_len)
    logger.info("Max keywords: %d", cfg.data.max_keywords)
    logger.info("Epochs: %d, LR: %s", cfg.training.epochs, cfg.training.learning_rate)

    # ── Токенизатор ──────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg)

    # ── Данные ───────────────────────────────────────────────────
    train_data = load_data(cfg.data.train_path)

    if cfg.data.val_path:
        val_data = load_data(cfg.data.val_path)
    else:
        train_data, val_data = train_val_split(
            train_data, val_ratio=cfg.data.val_ratio, seed=cfg.training.seed,
        )

    train_dataset = SummarizationDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_seq_len=cfg.model.max_seq_len,
        max_keywords=cfg.data.max_keywords,
        min_keyword_score=cfg.data.min_keyword_score,
        min_summary_len=cfg.data.min_summary_len,
        system_prompt=cfg.data.system_prompt,
        summary_field=cfg.data.summary_field,
        is_train=True,
    )

    val_dataset = SummarizationDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_seq_len=cfg.model.max_seq_len,
        max_keywords=cfg.data.max_keywords,
        min_keyword_score=cfg.data.min_keyword_score,
        min_summary_len=cfg.data.min_summary_len,
        system_prompt=cfg.data.system_prompt,
        summary_field=cfg.data.summary_field,
        is_train=True,  # val тоже с labels для eval_loss
    )

    data_collator = DataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=cfg.model.max_seq_len,
    )

    logger.info("Train: %d samples, Val: %d samples", len(train_dataset), len(val_dataset))

    # ── Модель ───────────────────────────────────────────────────
    model = load_model_for_training(cfg)

    # ── Training Arguments ───────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        seed=cfg.training.seed,
        report_to=cfg.training.report_to,
        deepspeed=cfg.training.deepspeed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        logging_first_step=True,
    )

    # ── Trainer ──────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ── Обучение ─────────────────────────────────────────────────
    logger.info("=== Начало обучения ===")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Сохранение ───────────────────────────────────────────────
    final_dir = os.path.join(cfg.training.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Модель сохранена в %s", final_dir)

    logger.info("=== Обучение завершено ===")


if __name__ == "__main__":
    main()
