"""
dataset.py — SummarizationDataset: lightweight wrapper around pre-processed records.

Each record (produced by ``scripts/prepare_data.py``) is expected to contain:

    doc_id              : str
    title               : str
    text_clean          : str                    — cleaned document text
    keywords_processed  : List[Tuple[str, float]]— [(kw, score), ...]
    summary             : str                    — target abstract
    summary_bucket      : str                    — "short" | "medium" | "long"

The Dataset does **not** tokenise; that responsibility belongs to the
``DataCollatorForSummarization``, which is called by the DataLoader.
This separation lets multiprocessing workers avoid pickling the tokeniser.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    """Maps integer index → pre-processed record dict.

    Parameters
    ----------
    records : List[Dict]
        Pre-processed records (output of ``scripts/prepare_data.py``).
    min_summary_len : int
        Records with ``summary`` shorter than this (in chars) are dropped.
    max_keywords : int
        Keyword list is truncated to this many items (top-scored first).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        min_summary_len: int = 100,
        max_keywords: int = 20,
    ) -> None:
        self.max_keywords = max_keywords

        # Validate and filter
        self._records: List[Dict[str, Any]] = []
        n_skipped = 0
        for rec in records:
            if len(rec.get("summary", "")) < min_summary_len:
                n_skipped += 1
                continue
            if not rec.get("keywords_processed"):
                n_skipped += 1
                continue
            if not rec.get("text_clean", "").strip():
                n_skipped += 1
                continue
            self._records.append(rec)

        if n_skipped:
            logger.info(
                "SummarizationDataset: skipped %d records "
                "(summary too short / no keywords / empty text)",
                n_skipped,
            )
        logger.info("SummarizationDataset: %d usable records", len(self._records))

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        min_summary_len: int = 100,
        max_keywords: int = 20,
    ) -> "SummarizationDataset":
        """Load from a JSON file saved by ``scripts/prepare_data.py``."""
        path = Path(path)
        logger.info("Loading dataset from %s …", path)
        with open(path, "r", encoding="utf-8") as fh:
            records: List[Dict[str, Any]] = json.load(fh)
        logger.info("Loaded %d raw records", len(records))
        return cls(records, min_summary_len=min_summary_len, max_keywords=max_keywords)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._records[idx]

        # Truncate keyword list to max_keywords (already ordered by score desc)
        kws: List[Tuple[str, float]] = rec["keywords_processed"][: self.max_keywords]

        return {
            "doc_id": rec.get("doc_id", ""),
            "title": rec.get("title", ""),
            "text_clean": rec["text_clean"],
            "keywords": [kw for kw, _ in kws],        # List[str]
            "kw_scores": [float(s) for _, s in kws],  # List[float]
            "summary": rec["summary"],
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary_length_stats(self) -> Dict[str, float]:
        """Return basic statistics on summary character lengths."""
        lens = [len(r["summary"]) for r in self._records]
        if not lens:
            return {}
        return {
            "count": len(lens),
            "min": min(lens),
            "max": max(lens),
            "mean": sum(lens) / len(lens),
        }
