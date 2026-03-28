#!/usr/bin/env python3
"""
generate.py — Генерация рефератов обученной LLM + LoRA моделью.

Примеры:
    # Генерация из обучающего датасета (train split)
    python scripts/generate.py \
        --config configs/qwen_lora.yaml \
        --adapter checkpoints/qwen_lora/final \
        --data dataset/dataset-SM-17k.json \
        --n 20 --output results/samples_train.json

    # Генерация из внешнего набора 480
    python scripts/generate.py \
        --config configs/qwen_lora.yaml \
        --adapter checkpoints/qwen_lora/final \
        --data dataset/dataset-480.json \
        --output results/samples_480.json

    # Один документ через stdin
    python scripts/generate.py \
        --config configs/qwen_lora.yaml \
        --adapter checkpoints/qwen_lora/final \
        --interactive
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

from src.training.config import load_config
from src.models.lora_setup import load_tokenizer, load_model_for_inference
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


@torch.inference_mode()
def generate_summary(
    model,
    tokenizer,
    text: str,
    keywords: list,
    system_prompt: str,
    max_keywords: int = 20,
    max_seq_len: int = 8192,
    gen_kwargs: Optional[Dict] = None,
) -> str:
    """Генерирует реферат для одного документа."""
    keywords_str = _format_keywords(keywords, max_keywords, min_score=0.0)
    messages = build_messages(text, keywords_str, system_prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(model.device)

    gen_kwargs = gen_kwargs or {}
    outputs = model.generate(
        **inputs,
        **gen_kwargs,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )

    # Декодируем только сгенерированную часть (без промпта)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return summary.strip()


def process_batch(
    model,
    tokenizer,
    data: List[Dict],
    cfg,
    n: Optional[int] = None,
) -> List[Dict]:
    """Обрабатывает набор документов и возвращает результаты."""
    if n is not None:
        data = data[:n]

    gen_kwargs = {
        "max_new_tokens": cfg.generation.max_new_tokens,
        "min_new_tokens": cfg.generation.min_new_tokens,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "repetition_penalty": cfg.generation.repetition_penalty,
        "do_sample": cfg.generation.do_sample,
        "num_beams": cfg.generation.num_beams,
    }

    # Определяем поле реферата для reference
    summary_field = _detect_summary_field(data[0]) if data else "summary"

    results = []
    for item in tqdm(data, desc="Генерация рефератов"):
        doc_id = item.get("doc_id", "unknown")
        text = item.get("text", "")
        keywords = item.get("keywords", [])
        reference = item.get(summary_field, "")

        if not text or len(text.strip()) < 100:
            logger.warning("Пропуск %s: пустой текст", doc_id)
            continue

        generated = generate_summary(
            model=model,
            tokenizer=tokenizer,
            text=text,
            keywords=keywords,
            system_prompt=cfg.data.system_prompt,
            max_keywords=cfg.data.max_keywords,
            max_seq_len=cfg.model.max_seq_len,
            gen_kwargs=gen_kwargs,
        )

        result = {
            "doc_id": doc_id,
            "generated_summary": generated,
            "reference_summary": reference,
            "text_length": len(text),
            "generated_length": len(generated),
            "num_keywords": len(keywords),
        }
        results.append(result)

        logger.info("--- %s ---", doc_id)
        logger.info("Сгенерировано [%d символов]: %s", len(generated), generated[:200])
        if reference:
            logger.info("Эталон [%d символов]: %s", len(reference), reference[:200])

    return results


def interactive_mode(model, tokenizer, cfg):
    """Интерактивный режим: ввод текста, получение реферата."""
    gen_kwargs = {
        "max_new_tokens": cfg.generation.max_new_tokens,
        "min_new_tokens": cfg.generation.min_new_tokens,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "repetition_penalty": cfg.generation.repetition_penalty,
        "do_sample": cfg.generation.do_sample,
        "num_beams": cfg.generation.num_beams,
    }

    print("\n=== Интерактивный режим ===")
    print("Введите текст статьи (Ctrl+D для завершения ввода, 'quit' для выхода):\n")

    while True:
        try:
            print("--- Новый документ ---")
            kw_input = input("Ключевые слова (через запятую, или пусто): ").strip()
            print("Текст (завершите пустой строкой):")

            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            text = "\n".join(lines)
            if text.lower() == "quit":
                break

            keywords = []
            if kw_input:
                for kw in kw_input.split(","):
                    keywords.append({"surface_form": kw.strip(), "score": 1.0})

            summary = generate_summary(
                model=model,
                tokenizer=tokenizer,
                text=text,
                keywords=keywords,
                system_prompt=cfg.data.system_prompt,
                max_keywords=cfg.data.max_keywords,
                max_seq_len=cfg.model.max_seq_len,
                gen_kwargs=gen_kwargs,
            )

            print(f"\n=== Реферат ({len(summary)} символов) ===")
            print(summary)
            print()

        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate summaries with LLM+LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--data", default=None, help="Path to JSON dataset")
    parser.add_argument("--n", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge LoRA weights")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--set", nargs="*", default=[], help="Override: key=value")
    args = parser.parse_args()

    from scripts.train import parse_overrides
    overrides = parse_overrides(args.set)
    cfg = load_config(args.config, overrides=overrides)

    # ── Загрузка модели ──────────────────────────────────────────
    tokenizer = load_tokenizer(cfg)
    model = load_model_for_inference(
        cfg, adapter_path=args.adapter, merge_adapter=not args.no_merge,
    )
    logger.info("Модель загружена, device: %s", next(model.parameters()).device)

    if args.interactive:
        interactive_mode(model, tokenizer, cfg)
        return

    if not args.data:
        parser.error("Укажите --data или --interactive")

    # ── Генерация ────────────────────────────────────────────────
    data = load_data(args.data)
    results = process_batch(model, tokenizer, data, cfg, n=args.n)

    # ── Сохранение ───────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Результаты сохранены: %s (%d записей)", args.output, len(results))
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    logger.info("=== Готово ===")


if __name__ == "__main__":
    main()
