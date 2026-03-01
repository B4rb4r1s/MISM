"""
tests/test_phase0.py — Unit tests for Phase 0: Data Pipeline.

Run:
    source .mism/bin/activate
    cd /home/b4rb4r1s/ForAll/MISM
    pytest tests/test_phase0.py -v

Coverage:
    TextCleaner            — BOM, whitespace, service markers
    KeywordProcessor       — JSON parsing, lowercasing, empty handling
    SlidingWindowProcessor — window count, sizes, overlap, max_windows cap
    SummarizationDataset   — filtering, __len__, __getitem__, from_json
    DataCollatorForSummarization — output shapes, dtypes, padding, kw_mask
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import T5Tokenizer

from src.data.preprocessing import KeywordProcessor, SlidingWindowProcessor, TextCleaner
from src.data.dataset import SummarizationDataset
from src.data.collator import DataCollatorForSummarization


# ============================================================
# Fixtures
# ============================================================

TOKENIZER_NAME = "ai-forever/ruT5-base"

@pytest.fixture(scope="session")
def tokenizer():
    """Load ruT5-base tokeniser once for the entire test session."""
    tok = T5Tokenizer.from_pretrained(TOKENIZER_NAME)
    return tok


def _make_records(n: int = 4, summary_len: int = 400) -> List[Dict[str, Any]]:
    """Generate *n* minimal valid pre-processed records."""
    records = []
    for i in range(n):
        records.append({
            "doc_id":             f"doc_{i}",
            "title":              f"Заголовок {i}",
            "text_clean":         f"Текст документа номер {i}. " * 60,
            "keywords_processed": [
                ("нейронные сети", 1.0),
                ("глубокое обучение", 1.0),
                ("обработка текста", 1.0),
            ],
            "summary":            "Краткое содержание статьи. " * (summary_len // 28),
            "summary_bucket":     "medium",
        })
    return records


# ============================================================
# 1. TextCleaner
# ============================================================

class TestTextCleaner:

    def setup_method(self):
        self.cleaner = TextCleaner(remove_service_markers=True)

    def test_removes_bom(self):
        text = "\ufeffПривет мир"
        assert self.cleaner.clean(text) == "Привет мир"

    def test_normalises_crlf(self):
        text = "строка 1\r\nстрока 2\rстрока 3"
        cleaned = self.cleaner.clean(text)
        assert "\r" not in cleaned
        assert "строка 1\nстрока 2\nстрока 3" == cleaned

    def test_collapses_multiple_blank_lines(self):
        text = "абзац 1\n\n\n\n\nабзац 2"
        cleaned = self.cleaner.clean(text)
        assert "\n\n\n" not in cleaned
        assert "абзац 1" in cleaned
        assert "абзац 2" in cleaned

    def test_collapses_inline_spaces(self):
        text = "слово1   слово2\t\tслово3"
        cleaned = self.cleaner.clean(text)
        assert "  " not in cleaned
        assert "слово1 слово2 слово3" == cleaned

    def test_removes_udk_line(self):
        text = "УДК 004.штрих.42\nОсновной текст"
        cleaned = self.cleaner.clean(text)
        assert "УДК" not in cleaned
        assert "Основной текст" in cleaned

    def test_removes_doi_line(self):
        text = "DOI: 10.1234/test.2023\nОсновной текст"
        cleaned = self.cleaner.clean(text)
        assert "DOI" not in cleaned

    def test_strips_surrounding_whitespace(self):
        text = "   \n  текст  \n  "
        cleaned = self.cleaner.clean(text)
        assert cleaned == "текст"

    def test_preserves_content(self):
        text = "Введение в машинное обучение.\nМетоды и результаты исследования."
        cleaned = self.cleaner.clean(text)
        assert "машинное обучение" in cleaned
        assert "результаты исследования" in cleaned

    def test_handles_non_string_gracefully(self):
        assert self.cleaner.clean(None) == ""   # type: ignore[arg-type]
        assert self.cleaner.clean(123)  == ""   # type: ignore[arg-type]

    def test_service_markers_optional(self):
        cleaner_no_strip = TextCleaner(remove_service_markers=False)
        text = "УДК 004.42\nТекст"
        cleaned = cleaner_no_strip.clean(text)
        assert "УДК" in cleaned


# ============================================================
# 2. KeywordProcessor
# ============================================================

class TestKeywordProcessor:

    def setup_method(self):
        self.proc = KeywordProcessor()

    def test_parse_normal(self):
        raw = '["НЕЙРОННЫЕ СЕТИ", "Глубокое Обучение", "NLP"]'
        result = self.proc.parse(raw)
        assert result == ["нейронные сети", "глубокое обучение", "nlp"]

    def test_parse_lowercases_all(self):
        raw = '["КРАУДСОРСИНГ", "Crowdsourcing Engine"]'
        result = self.proc.parse(raw)
        assert all(kw == kw.lower() for kw in result)

    def test_parse_filters_empty_strings(self):
        raw = '["нейросети", "", "  ", "трансформер"]'
        result = self.proc.parse(raw)
        assert "" not in result
        assert "нейросети" in result
        assert "трансформер" in result
        assert len(result) == 2

    def test_parse_invalid_json(self):
        assert self.proc.parse("не json") == []
        assert self.proc.parse(None) == []         # type: ignore[arg-type]
        assert self.proc.parse("") == []

    def test_parse_ignores_non_strings(self):
        raw = '["нейросети", 42, null, "обучение"]'
        result = self.proc.parse(raw)
        assert 42 not in result
        assert None not in result
        assert len(result) == 2

    def test_process_returns_score_one(self):
        raw = '["нейросети", "обучение"]'
        result = self.proc.process(raw)
        assert all(score == 1.0 for _, score in result)

    def test_max_keywords_truncation(self):
        proc = KeywordProcessor(max_keywords=3)
        raw = json.dumps([f"kw{i}" for i in range(10)])
        result = proc.parse(raw)
        assert len(result) == 3

    def test_bilingual_keywords(self):
        raw = '["КРАУДСОРСИНГ", "CROWDSOURCING ENGINE"]'
        result = self.proc.parse(raw)
        assert "краудсорсинг" in result
        assert "crowdsourcing engine" in result


# ============================================================
# 3. SlidingWindowProcessor
# ============================================================

class TestSlidingWindowProcessor:

    def _proc(self, window=6, overlap=2, max_windows=0):
        return SlidingWindowProcessor(
            window_size=window,
            overlap=overlap,
            pad_token_id=0,
            max_windows=max_windows,
        )

    def test_single_window_short_input(self):
        proc = self._proc(window=6, overlap=2)
        ids = [1, 2, 3]
        win_ids, win_masks = proc.create_windows(ids)
        assert len(win_ids) == 1
        assert len(win_ids[0]) == 6
        # First 3 are real tokens, last 3 are padding
        assert win_ids[0][:3] == [1, 2, 3]
        assert win_ids[0][3:] == [0, 0, 0]
        assert win_masks[0][:3] == [1, 1, 1]
        assert win_masks[0][3:] == [0, 0, 0]

    def test_exact_window_size_no_padding(self):
        proc = self._proc(window=4, overlap=1)
        ids = [1, 2, 3, 4]
        win_ids, _ = proc.create_windows(ids)
        assert len(win_ids) == 1
        assert win_ids[0] == [1, 2, 3, 4]

    def test_stride_and_overlap(self):
        # window=6, overlap=2, stride=4
        proc = self._proc(window=6, overlap=2)
        ids = list(range(12))   # 0..11
        win_ids, _ = proc.create_windows(ids)
        assert len(win_ids) == 3
        assert win_ids[0] == [0, 1, 2, 3, 4, 5]
        assert win_ids[1] == [4, 5, 6, 7, 8, 9]
        assert win_ids[2] == [8, 9, 10, 11, 0, 0]  # last 2 are padding

    def test_all_windows_same_length(self):
        proc = self._proc(window=8, overlap=2)
        ids = list(range(25))
        win_ids, win_masks = proc.create_windows(ids)
        for w_ids, w_mask in zip(win_ids, win_masks):
            assert len(w_ids) == 8
            assert len(w_mask) == 8

    def test_max_windows_cap(self):
        proc = self._proc(window=4, overlap=1, max_windows=3)
        ids = list(range(100))
        win_ids, _ = proc.create_windows(ids)
        assert len(win_ids) <= 3

    def test_attention_mask_matches_real_tokens(self):
        proc = self._proc(window=6, overlap=2)
        ids = [10, 20, 30]
        win_ids, win_masks = proc.create_windows(ids)
        # Real token positions have mask=1, padding positions have mask=0
        for i, (tok, mask) in enumerate(zip(win_ids[0], win_masks[0])):
            if i < 3:
                assert tok != 0 and mask == 1
            else:
                assert tok == 0 and mask == 0

    def test_custom_attention_mask(self):
        proc = self._proc(window=6, overlap=2)
        ids  = [1, 2, 3, 4, 5, 6]
        mask = [1, 1, 0, 0, 0, 0]  # first 2 real, rest already padded
        _, win_masks = proc.create_windows(ids, mask)
        assert win_masks[0] == [1, 1, 0, 0, 0, 0]

    def test_num_windows_matches_create_windows(self):
        proc = SlidingWindowProcessor(window_size=512, overlap=128, max_windows=32)
        for n_tokens in [100, 512, 1024, 5000, 10000]:
            ids = list(range(n_tokens))
            actual = len(proc.create_windows(ids)[0])
            expected = proc.num_windows(n_tokens)
            assert actual == expected, \
                f"num_windows({n_tokens}) predicted {expected} but got {actual}"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            SlidingWindowProcessor(window_size=4, overlap=4)  # overlap == window_size


# ============================================================
# 4. SummarizationDataset
# ============================================================

class TestSummarizationDataset:

    def test_len(self):
        records = _make_records(n=5)
        ds = SummarizationDataset(records)
        assert len(ds) == 5

    def test_getitem_keys(self):
        ds = SummarizationDataset(_make_records(n=2))
        item = ds[0]
        expected_keys = {"doc_id", "title", "text_clean", "keywords", "kw_scores", "summary"}
        assert set(item.keys()) == expected_keys

    def test_getitem_types(self):
        ds = SummarizationDataset(_make_records(n=2))
        item = ds[0]
        assert isinstance(item["keywords"],  list)
        assert isinstance(item["kw_scores"], list)
        assert all(isinstance(k, str)   for k in item["keywords"])
        assert all(isinstance(s, float) for s in item["kw_scores"])

    def test_short_summary_filtered(self):
        records = _make_records(n=3)
        records[1]["summary"] = "Мало"   # < 100 chars
        ds = SummarizationDataset(records, min_summary_len=100)
        assert len(ds) == 2

    def test_no_keywords_filtered(self):
        records = _make_records(n=3)
        records[0]["keywords_processed"] = []
        ds = SummarizationDataset(records)
        assert len(ds) == 2

    def test_empty_text_filtered(self):
        records = _make_records(n=3)
        records[2]["text_clean"] = "   "
        ds = SummarizationDataset(records)
        assert len(ds) == 2

    def test_max_keywords_truncation(self):
        records = _make_records(n=1)
        records[0]["keywords_processed"] = [(f"kw{i}", 1.0) for i in range(30)]
        ds = SummarizationDataset(records, max_keywords=10)
        item = ds[0]
        assert len(item["keywords"]) == 10
        assert len(item["kw_scores"]) == 10

    def test_from_json(self, tmp_path):
        records = _make_records(n=3)
        path = tmp_path / "data.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False)
        ds = SummarizationDataset.from_json(path)
        assert len(ds) == 3

    def test_summary_length_stats(self):
        ds = SummarizationDataset(_make_records(n=4))
        stats = ds.summary_length_stats()
        assert "min" in stats and "max" in stats and "mean" in stats
        assert stats["count"] == 4


# ============================================================
# 5. DataCollatorForSummarization
# ============================================================

class TestDataCollatorForSummarization:

    @pytest.fixture(autouse=True)
    def setup(self, tokenizer):
        """Create a small collator used in most tests."""
        self.tok = tokenizer
        self.collator = DataCollatorForSummarization(
            tokenizer=tokenizer,
            max_kw=5,
            kw_max_len=16,
            window_size=64,
            window_overlap=16,
            max_windows=8,
            max_summary_tokens=64,
        )
        self.ds = SummarizationDataset(_make_records(n=4))

    # ── Output structure ──────────────────────────────────────────────

    def test_output_keys(self):
        batch = [self.ds[i] for i in range(2)]
        out = self.collator(batch)
        expected = {
            "input_windows", "window_attention_mask",
            "kw_input_ids", "kw_attention_mask",
            "kw_scores", "kw_mask", "labels",
        }
        assert set(out.keys()) == expected

    def test_all_values_are_tensors(self):
        batch = [self.ds[i] for i in range(2)]
        out = self.collator(batch)
        for key, val in out.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"

    # ── Shapes ────────────────────────────────────────────────────────

    def test_batch_size_dimension(self):
        B = 3
        batch = [self.ds[i] for i in range(B)]
        out = self.collator(batch)
        assert out["input_windows"].shape[0]         == B
        assert out["kw_input_ids"].shape[0]          == B
        assert out["labels"].shape[0]                == B

    def test_window_shape(self):
        batch = [self.ds[i] for i in range(2)]
        out = self.collator(batch)
        B, W, S = out["input_windows"].shape
        assert S == self.collator.window_size
        assert W <= self.collator.max_windows

    def test_kw_shape(self):
        batch = [self.ds[i] for i in range(2)]
        out = self.collator(batch)
        B, K, L = out["kw_input_ids"].shape
        assert K == self.collator.max_kw
        assert L == self.collator.kw_max_len

    def test_labels_shape(self):
        batch = [self.ds[i] for i in range(2)]
        out = self.collator(batch)
        B, T = out["labels"].shape
        assert T == self.collator.max_summary_tokens

    # ── dtypes ────────────────────────────────────────────────────────

    def test_input_windows_dtype_long(self):
        batch = [self.ds[0]]
        out = self.collator(batch)
        assert out["input_windows"].dtype == torch.long
        assert out["window_attention_mask"].dtype == torch.long

    def test_kw_ids_dtype_long(self):
        batch = [self.ds[0]]
        out = self.collator(batch)
        assert out["kw_input_ids"].dtype == torch.long
        assert out["kw_attention_mask"].dtype == torch.long

    def test_kw_scores_dtype_float(self):
        batch = [self.ds[0]]
        out = self.collator(batch)
        assert out["kw_scores"].dtype == torch.float

    def test_kw_mask_dtype_bool(self):
        batch = [self.ds[0]]
        out = self.collator(batch)
        assert out["kw_mask"].dtype == torch.bool

    def test_labels_dtype_long(self):
        batch = [self.ds[0]]
        out = self.collator(batch)
        assert out["labels"].dtype == torch.long

    # ── Correctness ────────────────────────────────────────────────────

    def test_labels_padding_is_minus_100(self):
        """Padding positions in labels should be -100, not pad_token_id.

        We construct a deliberately short summary (≈ 5 tokens) so that the
        64-token label tensor is mostly padding → at least some -100 values.
        """
        short_item = {
            "doc_id": "x",
            "title": "",
            "text_clean": "Тест.",
            "keywords": ["нейросети"],
            "kw_scores": [1.0],
            "summary": "Краткий вывод.",   # ~5 tokens, well under max_summary_tokens=64
        }
        out = self.collator([short_item])
        labels = out["labels"]
        assert (labels == -100).any(), (
            "Expected -100 padding in labels for a short summary, "
            f"but got labels={labels}"
        )

    def test_kw_mask_true_for_real_keywords(self):
        """kw_mask should be True for real keywords and False for pad slots."""
        item = self.ds[0]
        n_kw = len(item["keywords"])
        batch = [item]
        out = self.collator(batch)
        kw_mask = out["kw_mask"][0]  # [max_kw]
        # First n_kw should be True
        assert kw_mask[:n_kw].all(), "Real keyword slots should be True"
        # Remaining should be False
        if n_kw < self.collator.max_kw:
            assert not kw_mask[n_kw:].any(), "Pad keyword slots should be False"

    def test_window_attention_mask_zeros_for_padded_windows(self):
        """Padded windows (added to match max batch width) must have all-zero masks."""
        # Create two items with very different doc lengths to force different window counts
        short_item = {
            "doc_id": "short",
            "title": "",
            "text_clean": "Короткий текст.",  # ~ 1 window
            "keywords": ["нейросети"],
            "kw_scores": [1.0],
            "summary": "Краткое содержание." * 20,
        }
        long_text = "Это длинный текст для проверки. " * 100  # many windows
        long_item = {
            "doc_id": "long",
            "title": "",
            "text_clean": long_text,
            "keywords": ["нейросети"],
            "kw_scores": [1.0],
            "summary": "Краткое содержание." * 20,
        }
        batch = [short_item, long_item]
        out = self.collator(batch)

        # Short document's padding windows should have zero attention mask
        W_short = len(self.tok(
            short_item["text_clean"], add_special_tokens=True, truncation=False
        )["input_ids"])
        n_short_windows = self.collator._window_proc.num_windows(W_short)
        total_windows = out["input_windows"].shape[1]

        if n_short_windows < total_windows:
            padded_mask = out["window_attention_mask"][0, n_short_windows:]
            assert (padded_mask == 0).all(), \
                "Padded windows for short document should have all-zero attention mask"

    def test_batch_consistency(self):
        """All items in a batch must produce tensors of identical shape."""
        B = 4
        batch = [self.ds[i] for i in range(B)]
        out = self.collator(batch)
        for key, tensor in out.items():
            assert tensor.shape[0] == B, f"{key} batch dim mismatch"

    def test_kw_scores_zero_for_padded_slots(self):
        """kw_scores should be 0.0 for padded keyword slots."""
        item = self.ds[0]
        n_kw = len(item["keywords"])
        out = self.collator([item])
        scores = out["kw_scores"][0]
        if n_kw < self.collator.max_kw:
            assert (scores[n_kw:] == 0.0).all(), "Padded kw slots must have score 0.0"
