"""
config.py — Конфигурация для LLM + LoRA суммаризатора.

Загружает параметры из YAML, предоставляет типизированный доступ через dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_len: int = 8192
    torch_dtype: str = "bfloat16"


@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    train_path: str = "dataset/dataset-SM-17k.json"
    val_path: Optional[str] = None
    val_ratio: float = 0.1
    test_path: str = "dataset/dataset-480.json"
    max_keywords: int = 20
    min_keyword_score: float = 0.0
    min_summary_len: int = 50
    summary_field: str = "auto"
    system_prompt: str = (
        "Ты — система автоматического реферирования научных статей "
        "на русском языке. Составь краткий реферат статьи объёмом "
        "около 100 слов, опираясь на ключевые слова и полный текст."
    )


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints/qwen_lora"
    epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    seed: int = 42
    report_to: str = "tensorboard"
    deepspeed: Optional[str] = "configs/deepspeed_zero2.json"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 300
    min_new_tokens: int = 30
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    do_sample: bool = True
    num_beams: int = 1


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _apply_dict(dc: Any, d: Dict[str, Any]) -> None:
    """Рекурсивно обновляет поля dataclass из словаря."""
    for k, v in d.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(
    path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Загружает конфигурацию из YAML с опциональными override'ами.

    Overrides задаются в формате "section.key=value" или плоским dict:
        {"training.learning_rate": 1e-5, "data.max_keywords": 30}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    for section_name in ("model", "lora", "data", "training", "generation"):
        section_data = raw.get(section_name, {})
        if section_data:
            _apply_dict(getattr(cfg, section_name), section_data)

    if overrides:
        for key, value in overrides.items():
            if "." in key:
                section, attr = key.split(".", 1)
                if hasattr(cfg, section):
                    setattr(getattr(cfg, section), attr, value)
            else:
                for section_name in ("model", "lora", "data", "training", "generation"):
                    sec = getattr(cfg, section_name)
                    if hasattr(sec, key):
                        setattr(sec, key, value)
                        break

    return cfg
