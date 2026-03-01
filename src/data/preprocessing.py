"""
preprocessing.py — Text and keyword preprocessing utilities.

Classes
-------
TextCleaner
    Cleans raw document text: removes BOM, normalises whitespace,
    optionally strips service markers (УДК, DOI, ISSN, ORCID).

KeywordProcessor
    Parses the ``keywords_original`` JSON string and returns a list of
    (lowercase keyword phrase, score) tuples with uniform score = 1.0.

SlidingWindowProcessor
    Splits a flat token-ID sequence into overlapping fixed-size windows.
    Pads the last window if necessary.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """Clean raw scientific document text scraped from CyberLeninka.

    Processing steps (applied in order):
        1. Remove BOM (``\\ufeff``).
        2. Normalise line endings (``\\r\\n`` → ``\\n``).
        3. Optionally strip service-marker lines (УДК, DOI, ISSN, ORCID).
        4. Collapse ≥ 3 consecutive blank lines to a single blank line.
        5. Collapse multiple inline spaces (preserving newlines).
        6. Strip leading/trailing whitespace.

    Parameters
    ----------
    remove_service_markers : bool
        When *True* (default), lines that consist solely of УДК/DOI/ISSN/ORCID
        codes are removed.
    """

    _SERVICE_PATTERNS: List[re.Pattern] = [
        re.compile(r'^\s*УДК\b[^\n]*$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^\s*DOI\s*:?\s*\S+\s*$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^\s*ISSN\s*[\w\-]+\s*$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^\s*ORCID\s*:?\s*\S+\s*$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^\s*e-?ISSN\s*[\w\-]+\s*$', re.MULTILINE | re.IGNORECASE),
    ]

    def __init__(self, remove_service_markers: bool = True) -> None:
        self.remove_service_markers = remove_service_markers

    def clean(self, text: str) -> str:
        """Return a cleaned version of *text*."""
        if not isinstance(text, str):
            return ""

        # 1. BOM
        text = text.replace('\ufeff', '')

        # 2. Line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 3. Service markers
        if self.remove_service_markers:
            for pattern in self._SERVICE_PATTERNS:
                text = pattern.sub('', text)

        # 4. Collapse excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 5. Collapse multiple inline spaces (keep newlines intact)
        text = re.sub(r'[^\S\n]+', ' ', text)

        # 6. Strip
        return text.strip()


# ---------------------------------------------------------------------------
# KeywordProcessor
# ---------------------------------------------------------------------------

class KeywordProcessor:
    """Parse and normalise the ``keywords_original`` field.

    The field is stored as a JSON-encoded list of strings, e.g.::

        '["КРАУДСОРСИНГ", "МЕХАНИЗИРОВАННЫЙ ТРУД", "CROWDSOURCING ENGINE"]'

    All keywords are lowercased and stripped.  Empty strings after stripping
    are discarded.  Scores are set uniformly to **1.0** (author-provided
    keywords carry equal importance by default).

    Parameters
    ----------
    max_keywords : int | None
        If provided, the list is truncated to at most *max_keywords* items.
    """

    def __init__(self, max_keywords: Optional[int] = None) -> None:
        self.max_keywords = max_keywords

    # ------------------------------------------------------------------

    def parse(self, keywords_original: str) -> List[str]:
        """Parse *keywords_original* JSON string → ``List[str]`` (lowercase)."""
        if not keywords_original or not isinstance(keywords_original, str):
            return []
        try:
            raw: list = json.loads(keywords_original)
        except (json.JSONDecodeError, ValueError):
            logger.debug("KeywordProcessor: failed to parse %r", keywords_original[:80])
            return []

        result: List[str] = []
        for kw in raw:
            if not isinstance(kw, str):
                continue
            cleaned = kw.strip().lower()
            if cleaned:
                result.append(cleaned)

        if self.max_keywords is not None:
            result = result[: self.max_keywords]
        return result

    def process(self, keywords_original: str) -> List[Tuple[str, float]]:
        """Return ``[(keyword, 1.0), …]`` tuples ready for the model."""
        return [(kw, 1.0) for kw in self.parse(keywords_original)]


# ---------------------------------------------------------------------------
# SlidingWindowProcessor
# ---------------------------------------------------------------------------

class SlidingWindowProcessor:
    """Split a flat token-ID list into overlapping fixed-size windows.

    Parameters
    ----------
    window_size  : int   Tokens per window (default 512, matching T5 limit).
    overlap      : int   Token overlap between consecutive windows (default 128).
    pad_token_id : int   ID used to right-pad the last window.
    max_windows  : int   Hard upper bound on the number of windows (0 = no cap).

    The stride equals ``window_size - overlap``.

    Example (window_size=6, overlap=2, stride=4, tokens=[0..11])::

        window 0 → [0, 1, 2, 3, 4, 5]
        window 1 → [4, 5, 6, 7, 8, 9]
        window 2 → [8, 9, 10, 11, PAD, PAD]
    """

    def __init__(
        self,
        window_size: int = 512,
        overlap: int = 128,
        pad_token_id: int = 0,
        max_windows: int = 32,
    ) -> None:
        if overlap >= window_size:
            raise ValueError(f"overlap ({overlap}) must be < window_size ({window_size})")
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.pad_token_id = pad_token_id
        self.max_windows = max_windows

    # ------------------------------------------------------------------

    def create_windows(
        self,
        input_ids: List[int],
        attention_mask: Optional[List[int]] = None,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Split *input_ids* into overlapping windows.

        Parameters
        ----------
        input_ids      : Flat list of token IDs.
        attention_mask : Flat mask list (1 = real token, 0 = pad). If *None*,
                         all-ones are assumed.

        Returns
        -------
        windows_ids   : List of windows, each of length ``window_size``.
        windows_masks : Corresponding attention masks of the same shape.
        """
        n = len(input_ids)
        if attention_mask is None:
            attention_mask = [1] * n

        windows_ids: List[List[int]] = []
        windows_masks: List[List[int]] = []

        start = 0
        while True:
            end = start + self.window_size
            ids_chunk = input_ids[start:end]
            mask_chunk = attention_mask[start:end]

            # Right-pad the last (possibly short) window
            pad_len = self.window_size - len(ids_chunk)
            if pad_len > 0:
                ids_chunk = ids_chunk + [self.pad_token_id] * pad_len
                mask_chunk = mask_chunk + [0] * pad_len

            windows_ids.append(ids_chunk)
            windows_masks.append(mask_chunk)

            # Stop if we've hit the max_windows cap
            if self.max_windows and len(windows_ids) >= self.max_windows:
                break

            # Advance to the next starting position
            next_start = start + self.stride

            # Stop when the next window would contain NO novel tokens beyond
            # the overlap region (i.e. next_start + overlap >= n).
            # This prevents near-empty trailing windows when the input length
            # is exactly (or just above) a multiple of window_size.
            if next_start + self.overlap >= n:
                break

            start = next_start

        return windows_ids, windows_masks

    def num_windows(self, num_tokens: int) -> int:
        """Return the expected number of windows for *num_tokens* tokens.

        Matches the stopping criterion of :meth:`create_windows`:
        at least one window is always created; additional windows are created
        while ``next_start + overlap < n``.
        """
        if num_tokens <= 0:
            return 0
        # Number of additional windows beyond the first:
        #   k-th additional window starts at k*stride;
        #   it is created while k*stride + overlap < n
        #   → k < (n - overlap) / stride
        #   → k_max = floor((n - overlap - 1) / stride)  (largest valid k)
        n_additional = max(0, (num_tokens - self.overlap - 1) // self.stride)
        total = 1 + n_additional
        if self.max_windows:
            total = min(total, self.max_windows)
        return total
