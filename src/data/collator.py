"""
collator.py — DataCollatorForSummarization.

Responsibilities
----------------
1. Tokenise each document text with the sliding-window approach.
2. Tokenise each keyword phrase independently; pad/truncate to
   (max_kw, kw_max_len).
3. Tokenise the target summary; pad to max_summary_tokens.
4. Collate all samples into a single batch of tensors.

Returned batch keys
-------------------
input_windows          [B, num_win, window_size]   long
window_attention_mask  [B, num_win, window_size]   long
kw_input_ids           [B, max_kw,  kw_max_len]    long
kw_attention_mask      [B, max_kw,  kw_max_len]    long
kw_scores              [B, max_kw]                 float
kw_mask                [B, max_kw]                 bool  (True = real KW)
labels                 [B, max_summary_tokens]      long  (-100 for padding)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from src.data.preprocessing import SlidingWindowProcessor

logger = logging.getLogger(__name__)


class DataCollatorForSummarization:
    """Tokenise and batch samples for the Dual-Encoder Summarizer.

    Parameters
    ----------
    tokenizer          : HuggingFace tokeniser (T5Tokenizer / T5TokenizerFast).
    max_kw             : Maximum number of keywords per sample.
    kw_max_len         : Maximum tokens per keyword phrase (incl. special tokens).
    window_size        : Sliding-window size in tokens.
    window_overlap     : Overlap between consecutive windows in tokens.
    max_windows        : Hard cap on number of document windows.
    max_summary_tokens : Maximum target-sequence length in tokens.
    label_pad_id       : Value substituted for padding positions in labels
                         (``-100`` is ignored by ``CrossEntropyLoss``).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_kw: int = 20,
        kw_max_len: int = 32,
        window_size: int = 512,
        window_overlap: int = 128,
        max_windows: int = 32,
        max_summary_tokens: int = 512,
        label_pad_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_kw = max_kw
        self.kw_max_len = kw_max_len
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.max_windows = max_windows
        self.max_summary_tokens = max_summary_tokens
        self.label_pad_id = label_pad_id
        self.pad_id: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self._window_proc = SlidingWindowProcessor(
            window_size=window_size,
            overlap=window_overlap,
            pad_token_id=self.pad_id,
            max_windows=max_windows,
        )

        # Pre-compute the padded keyword encoding for empty (pad) slots
        self._pad_kw_enc = tokenizer(
            "",
            max_length=kw_max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a list of ``SummarizationDataset`` items into a batch."""
        all_win_ids:    List[List[List[int]]] = []
        all_win_masks:  List[List[List[int]]] = []
        all_kw_ids:     List[List[List[int]]] = []
        all_kw_masks:   List[List[List[int]]] = []
        all_kw_scores:  List[List[float]]     = []
        all_kw_bool:    List[List[bool]]       = []
        all_labels:     List[List[int]]        = []

        max_num_windows = 0

        for item in batch:
            # ── Document → sliding windows ──────────────────────────────
            enc_doc = self.tokenizer(
                item["text_clean"],
                add_special_tokens=True,
                truncation=False,       # length handled by SlidingWindowProcessor
                return_attention_mask=True,
            )
            win_ids, win_masks = self._window_proc.create_windows(
                enc_doc["input_ids"],
                enc_doc["attention_mask"],
            )
            all_win_ids.append(win_ids)
            all_win_masks.append(win_masks)
            max_num_windows = max(max_num_windows, len(win_ids))

            # ── Keywords → (max_kw, kw_max_len) ────────────────────────
            kw_ids, kw_masks, kw_scores, kw_bool = self._tokenise_keywords(
                item["keywords"], item["kw_scores"]
            )
            all_kw_ids.append(kw_ids)
            all_kw_masks.append(kw_masks)
            all_kw_scores.append(kw_scores)
            all_kw_bool.append(kw_bool)

            # ── Summary → labels ────────────────────────────────────────
            all_labels.append(self._tokenise_summary(item["summary"]))

        # ── Pad document windows across the batch ──────────────────────
        input_windows, window_attention_mask = self._pad_windows(
            all_win_ids, all_win_masks, max_num_windows
        )

        return {
            "input_windows":          input_windows,          # [B, W, S]
            "window_attention_mask":  window_attention_mask,  # [B, W, S]
            "kw_input_ids":           torch.tensor(all_kw_ids,    dtype=torch.long),   # [B, K, L]
            "kw_attention_mask":      torch.tensor(all_kw_masks,  dtype=torch.long),   # [B, K, L]
            "kw_scores":              torch.tensor(all_kw_scores, dtype=torch.float),  # [B, K]
            "kw_mask":                torch.tensor(all_kw_bool,   dtype=torch.bool),   # [B, K]
            "labels":                 self._pad_labels(all_labels),                    # [B, T]
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenise_keywords(
        self,
        keywords: List[str],
        scores: List[float],
    ) -> Tuple[List[List[int]], List[List[int]], List[float], List[bool]]:
        """Tokenise up to ``max_kw`` keyword phrases; pad remaining slots.

        Returns
        -------
        ids        : [max_kw, kw_max_len]
        masks      : [max_kw, kw_max_len]
        scores     : [max_kw]   (0.0 for padded slots)
        bool_mask  : [max_kw]   (True = real keyword, False = padding)
        """
        ids_out:   List[List[int]] = []
        masks_out: List[List[int]] = []
        scores_out: List[float]    = []
        bool_out:  List[bool]      = []

        for kw, score in zip(keywords[: self.max_kw], scores[: self.max_kw]):
            enc = self.tokenizer(
                kw,
                max_length=self.kw_max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            ids_out.append(enc["input_ids"])
            masks_out.append(enc["attention_mask"])
            scores_out.append(float(score))
            bool_out.append(True)

        # Fill remaining slots with pad encoding
        n_pad = self.max_kw - len(ids_out)
        for _ in range(n_pad):
            ids_out.append(self._pad_kw_enc["input_ids"])
            masks_out.append(self._pad_kw_enc["attention_mask"])
            scores_out.append(0.0)
            bool_out.append(False)

        return ids_out, masks_out, scores_out, bool_out

    def _tokenise_summary(self, summary: str) -> List[int]:
        """Tokenise target summary; replace pad tokens with ``label_pad_id``."""
        enc = self.tokenizer(
            summary,
            max_length=self.max_summary_tokens,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        # Mask padding positions so CrossEntropyLoss ignores them
        return [
            self.label_pad_id if tok_id == self.pad_id else tok_id
            for tok_id in enc["input_ids"]
        ]

    def _pad_windows(
        self,
        all_win_ids:   List[List[List[int]]],
        all_win_masks: List[List[List[int]]],
        max_num_windows: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad samples with fewer windows to ``max_num_windows``.

        Returns
        -------
        input_windows          : [B, max_num_windows, window_size]  long
        window_attention_mask  : [B, max_num_windows, window_size]  long
        """
        B = len(all_win_ids)
        W = max_num_windows
        S = self.window_size

        ids_t  = torch.zeros(B, W, S, dtype=torch.long)
        mask_t = torch.zeros(B, W, S, dtype=torch.long)

        for b, (win_ids, win_masks) in enumerate(zip(all_win_ids, all_win_masks)):
            n = len(win_ids)
            if n > 0:
                ids_t[b, :n]  = torch.tensor(win_ids,  dtype=torch.long)
                mask_t[b, :n] = torch.tensor(win_masks, dtype=torch.long)
            # Windows beyond n stay zero → attention_mask = 0 (padding)

        return ids_t, mask_t

    def _pad_labels(self, all_labels: List[List[int]]) -> torch.Tensor:
        """Stack already-padded label lists into a tensor."""
        # All lists are padded to max_summary_tokens in _tokenise_summary
        return torch.tensor(all_labels, dtype=torch.long)
