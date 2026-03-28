"""
lora_setup.py — Загрузка базовой LLM + настройка LoRA адаптера.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from src.training.config import Config

logger = logging.getLogger(__name__)

TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_tokenizer(cfg: Config) -> AutoTokenizer:
    """Загружает токенизатор и настраивает pad_token."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
        padding_side="left",  # decoder-only: паддинг слева
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(
        "Tokenizer: vocab_size=%d, pad_token=%s",
        tokenizer.vocab_size, tokenizer.pad_token,
    )
    return tokenizer


def load_base_model(
    cfg: Config,
    quantize_4bit: bool = False,
) -> AutoModelForCausalLM:
    """Загружает базовую модель (опционально с 4-bit квантизацией)."""
    dtype = TORCH_DTYPE_MAP.get(cfg.model.torch_dtype, torch.bfloat16)

    kwargs = dict(
        pretrained_model_name_or_path=cfg.model.name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)
    else:
        kwargs["device_map"] = None  # управляется DeepSpeed / accelerate

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    if quantize_4bit:
        model = prepare_model_for_kbit_training(model)

    total = sum(p.numel() for p in model.parameters())
    logger.info("Base model: %s, %.1fB params", cfg.model.name, total / 1e9)

    return model


def apply_lora(model: AutoModelForCausalLM, cfg: Config) -> AutoModelForCausalLM:
    """Применяет LoRA адаптер к модели."""
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model


def load_model_for_training(
    cfg: Config,
    quantize_4bit: bool = False,
) -> AutoModelForCausalLM:
    """Загружает модель с LoRA для обучения."""
    model = load_base_model(cfg, quantize_4bit=quantize_4bit)
    model = apply_lora(model, cfg)

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    return model


def load_model_for_inference(
    cfg: Config,
    adapter_path: str,
    merge_adapter: bool = True,
) -> AutoModelForCausalLM:
    """Загружает модель с обученным LoRA адаптером для инференса.

    Args:
        merge_adapter: если True, мержит LoRA веса в базовую модель
                       (быстрее инференс, больше памяти).
    """
    dtype = TORCH_DTYPE_MAP.get(cfg.model.torch_dtype, torch.bfloat16)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    if merge_adapter:
        model = model.merge_and_unload()
        logger.info("LoRA адаптер вмержен в базовую модель")
    else:
        logger.info("LoRA адаптер загружен (adapter mode)")

    model.eval()
    return model
