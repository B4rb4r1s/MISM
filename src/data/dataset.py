"""
dataset.py — Dataset для LLM-суммаризации научных статей.

Форматирует данные в ChatML (Qwen2.5), создаёт input_ids и labels
с маскировкой промпта (-100) — модель обучается только на рефератах.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _detect_summary_field(sample: Dict[str, Any], hint: str = "auto") -> str:
    """Определяет имя поля реферата в данных."""
    if hint != "auto":
        return hint
    for candidate in ("summary", "target-summary", "abstract"):
        if candidate in sample and sample[candidate]:
            return candidate
    return "summary"


def _format_keywords(
    keywords: List[Dict[str, Any]],
    max_keywords: int,
    min_score: float,
) -> str:
    """Форматирует ключевые слова в строку для промпта.

    Сортирует по score (убывание), берёт top-N, фильтрует по min_score.
    """
    if not keywords:
        return ""

    filtered = [
        kw for kw in keywords
        if isinstance(kw, dict) and kw.get("score", 0) >= min_score
    ]
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    filtered = filtered[:max_keywords]

    parts = []
    for kw in filtered:
        surface = kw.get("surface_form", kw.get("keyword", ""))
        score = kw.get("score", 0)
        parts.append(f"{surface} ({score:.2f})")

    return ", ".join(parts)


def trim_to_last_sentence(text: str) -> str:
    """Обрезает текст до последнего полного предложения.

    Если текст уже заканчивается на '.', '!' или '?' — возвращает как есть.
    Иначе ищет последний такой символ и обрезает после него.
    """
    text = text.strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text

    # Ищем последнюю точку/восклицательный/вопросительный
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?":
            return text[: i + 1]

    # Не нашли ни одного — возвращаем как есть
    return text


def build_messages(
    text: str,
    keywords_str: str,
    system_prompt: str,
    summary: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Строит список сообщений в формате ChatML."""
    user_content = ""
    if keywords_str:
        user_content += f"Ключевые слова: {keywords_str}\n\n"
    user_content += f"Текст статьи:\n{text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if summary is not None:
        messages.append({"role": "assistant", "content": summary})

    return messages


def truncate_text_to_fit(
    tokenizer: Any,
    text: str,
    keywords_str: str,
    system_prompt: str,
    max_seq_len: int,
    reserve_tokens: int = 100,
) -> str:
    """Обрезает текст статьи так, чтобы полный промпт влезал в max_seq_len.

    Сначала измеряет длину «обёртки» (system prompt + keywords + ChatML теги),
    затем обрезает текст, чтобы суммарно не превышать лимит.
    reserve_tokens — запас под generation prompt и safety margin.

    Возвращает текст (возможно обрезанный).
    """
    # Измеряем обёртку без текста статьи
    shell_messages = build_messages("", keywords_str, system_prompt)
    shell_text = tokenizer.apply_chat_template(
        shell_messages, tokenize=False, add_generation_prompt=True,
    )
    shell_tokens = len(tokenizer.encode(shell_text, add_special_tokens=False))

    # Сколько токенов доступно для текста статьи
    available = max_seq_len - shell_tokens - reserve_tokens
    if available <= 0:
        logger.warning("Обёртка промпта (%d tok) превышает max_seq_len (%d)", shell_tokens, max_seq_len)
        available = 512  # минимальный fallback

    # Токенизируем текст и обрезаем если нужно
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(text_ids) <= available:
        return text  # влезает целиком

    # Обрезаем и декодируем обратно в строку
    truncated_ids = text_ids[:available]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    # Обрезаем до последнего полного предложения (точка, !, ?)
    last_period = -1
    for i in range(len(truncated_text) - 1, max(0, len(truncated_text) - 200), -1):
        if truncated_text[i] in '.!?':
            last_period = i
            break

    if last_period > len(truncated_text) * 0.8:  # не теряем больше 20%
        truncated_text = truncated_text[: last_period + 1]

    logger.debug(
        "Текст обрезан: %d → %d токенов (доступно %d)",
        len(text_ids), available, available,
    )
    return truncated_text


class SummarizationDataset(Dataset):
    """Dataset для обучения LLM-суммаризатора.

    Каждый элемент — пара (input_ids, labels), где labels = -100
    для всех токенов промпта (system + user) и реальные id для реферата.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int = 32768,
        max_keywords: int = 20,
        min_keyword_score: float = 0.0,
        min_summary_len: int = 50,
        system_prompt: str = "",
        summary_field: str = "auto",
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_keywords = max_keywords
        self.min_keyword_score = min_keyword_score
        self.system_prompt = system_prompt
        self.is_train = is_train

        if data:
            self.summary_field = _detect_summary_field(data[0], summary_field)
        else:
            self.summary_field = "summary"

        self.samples = self._filter(data)
        logger.info(
            "Dataset: %d samples (из %d), is_train=%s, summary_field=%s",
            len(self.samples), len(data), is_train, self.summary_field,
        )

    def _filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Фильтрует пустые / короткие записи."""
        result = []
        for item in data:
            text = item.get("text", "")
            if not text or len(text.strip()) < 100:
                continue

            if self.is_train:
                summary = item.get(self.summary_field, "")
                if not summary or len(summary.strip()) < self.min_summary_len:
                    continue

            result.append(item)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        text = item["text"]
        keywords = item.get("keywords", [])
        summary = item.get(self.summary_field, "") if self.is_train else None

        keywords_str = _format_keywords(
            keywords, self.max_keywords, self.min_keyword_score,
        )

        if self.is_train:
            return self._encode_train(text, keywords_str, summary)
        else:
            return self._encode_inference(text, keywords_str, item.get("doc_id", ""))

    def _encode_train(
        self,
        text: str,
        keywords_str: str,
        summary: str,
    ) -> Dict[str, torch.Tensor]:
        """Кодирует пример для обучения с маскировкой промпта."""
        # Оцениваем длину реферата, чтобы зарезервировать под него место
        end_token = "<|im_end|>"
        response_text = summary + end_token
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        # Умная обрезка текста ДО токенизации полного промпта
        text = truncate_text_to_fit(
            tokenizer=self.tokenizer,
            text=text,
            keywords_str=keywords_str,
            system_prompt=self.system_prompt,
            max_seq_len=self.max_seq_len,
            reserve_tokens=len(response_ids) + 50,
        )

        # Токенизируем промпт (system + user + начало assistant)
        prompt_messages = build_messages(text, keywords_str, self.system_prompt)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # Safety: жёсткая обрезка на случай если всё ещё не влезает
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _encode_inference(
        self,
        text: str,
        keywords_str: str,
        doc_id: str,
    ) -> Dict[str, Any]:
        """Кодирует пример для инференса (только промпт)."""
        # Умная обрезка текста ДО токенизации
        text = truncate_text_to_fit(
            tokenizer=self.tokenizer,
            text=text,
            keywords_str=keywords_str,
            system_prompt=self.system_prompt,
            max_seq_len=self.max_seq_len,
            reserve_tokens=100,
        )

        prompt_messages = build_messages(text, keywords_str, self.system_prompt)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        return {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "doc_id": doc_id,
        }


class DataCollatorWithPadding:
    """Collator: паддит батч до макс. длины в батче (dynamic padding)."""

    def __init__(self, pad_token_id: int, max_seq_len: int = 32768):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_seq_len,
        )

        input_ids = []
        attention_mask = []
        labels = []
        has_labels = "labels" in features[0]

        for f in features:
            ids = f["input_ids"]
            pad_len = max_len - len(ids)

            # Паддинг слева (стандарт для decoder-only моделей)
            input_ids.append(
                torch.cat([
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ids[:max_len],
                ])
            )
            attention_mask.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    f["attention_mask"][:max_len],
                ])
            )

            if has_labels:
                lab = f["labels"]
                labels.append(
                    torch.cat([
                        torch.full((pad_len,), -100, dtype=torch.long),
                        lab[:max_len],
                    ])
                )

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
        }
        if has_labels:
            batch["labels"] = torch.stack(labels)

        return batch


def load_data(path: str | Path) -> List[Dict[str, Any]]:
    """Загружает JSON-данные."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Загружено %d записей из %s", len(data), path)
    return data


def train_val_split(
    data: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Разделяет данные на train и val."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    val_size = int(len(data) * val_ratio)
    val_indices = set(indices[:val_size])

    train = [data[i] for i in range(len(data)) if i not in val_indices]
    val = [data[i] for i in range(len(data)) if i in val_indices]

    logger.info("Split: train=%d, val=%d", len(train), len(val))
    return train, val
