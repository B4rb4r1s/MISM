#!/usr/bin/env python3
"""
generate_zeroshot.py — Zero-shot генерация рефератов (без обучения LoRA).

Загружает базовую Qwen2.5-7B-Instruct и генерирует рефераты
используя только системный промпт + ключевые слова + текст.

Пример:
    python scripts/generate_zeroshot.py \
        --config configs/qwen_lora.yaml \
        --data dataset/dataset-480.json \
        --n 20 \
        --output results/zeroshot_480.json
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.training.config import load_config
from src.models.lora_setup import _detect_attn_implementation
from src.data.dataset import (
    _format_keywords,
    _detect_summary_field,
    build_messages,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@torch.inference_mode()
def generate_summary(model, tokenizer, text, keywords, cfg):
    """Генерирует реферат для одного документа."""
    keywords_str = _format_keywords(
        keywords, cfg.data.max_keywords, min_score=0.0,
    )
    messages = build_messages(text, keywords_str, cfg.data.system_prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.model.max_seq_len,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.generation.max_new_tokens,
        min_new_tokens=cfg.generation.min_new_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        repetition_penalty=cfg.generation.repetition_penalty,
        do_sample=cfg.generation.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )

    generated_ids = outputs[0][prompt_len:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return summary.strip(), keywords_str, prompt_len


def main():
    parser = argparse.ArgumentParser(description="Zero-shot summarization baseline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data", required=True, help="Path to JSON dataset")
    parser.add_argument("--n", type=int, default=None, help="Limit samples")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--set", nargs="*", default=[], help="Override: key=value")
    args = parser.parse_args()

    from scripts.train import parse_overrides
    overrides = parse_overrides(args.set)
    cfg = load_config(args.config, overrides=overrides)

    # ── Загрузка базовой модели (без LoRA) ───────────────────────
    dtype = TORCH_DTYPE_MAP.get(cfg.model.torch_dtype, torch.bfloat16)
    logger.info("Загрузка %s (zero-shot, dtype=%s)...", cfg.model.name, cfg.model.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=_detect_attn_implementation(),
    )
    model.eval()
    logger.info("Модель загружена, device: %s", next(model.parameters()).device)

    # ── Данные ───────────────────────────────────────────────────
    data = load_data(args.data)
    if args.n is not None:
        data = data[:args.n]

    summary_field = _detect_summary_field(data[0]) if data else "summary"

    # ── Генерация ────────────────────────────────────────────────
    results = []
    for item in tqdm(data, desc="Zero-shot генерация"):
        doc_id = item.get("doc_id", "unknown")
        text = item.get("text", "")
        keywords = item.get("keywords", [])
        reference = item.get(summary_field, "")

        if not text or len(text.strip()) < 100:
            continue

        generated, keywords_str, prompt_tokens = generate_summary(
            model, tokenizer, text, keywords, cfg,
        )

        result = {
            "doc_id": doc_id,
            "generated_summary": generated,
            "reference_summary": reference,
            "keywords_used": keywords_str,
            "text_preview": text[:200].strip(),
            "text_length": len(text),
            "generated_length": len(generated),
            "prompt_tokens": prompt_tokens,
            "num_keywords": min(len(keywords), cfg.data.max_keywords),
        }
        results.append(result)

        logger.info("--- %s (prompt: %d tok) ---", doc_id, prompt_tokens)
        logger.info("Сгенерировано [%d сим.]: %s", len(generated), generated[:200])

    # ── Сохранение ───────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Результаты: %s (%d записей)", args.output, len(results))
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    logger.info("=== Готово ===")


if __name__ == "__main__":
    main()
