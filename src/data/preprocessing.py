"""
preprocessing.py — Text and keyword preprocessing utilities.

Classes
-------
TextCleaner
    Cleans raw document text: removes BOM, normalises whitespace,
    optionally strips service markers (УДК, DOI, ISSN, ORCID).

AbstractRemover
    Removes the author abstract (and keyword line) from the cleaned
    document text so that the model cannot simply copy it.  Uses
    direct matching against the ``summary`` field with marker-based
    fallback.

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
# AbstractRemover
# ---------------------------------------------------------------------------

class AbstractRemover:
    """Remove the author abstract (аннотация) from cleaned document text.

    The CyberLeninka dataset embeds the abstract directly in the source text,
    which causes the model to learn a trivial copy strategy instead of true
    summarisation.  This class strips the abstract so that ``text_clean``
    contains only the paper body.

    Removal strategy (tried in order):
        1. **Direct match** — locate the ``summary`` text inside
           ``text_clean`` (after whitespace normalisation) and excise it.
        2. **Marker-based fallback** — if direct match fails, use regex to
           remove text between ``Аннотация[.:]`` and ``Ключевые слова``.
        3. **Keywords-line removal** — always strip the
           ``Ключевые слова: ...`` line (metadata, not body content).
        4. **English trailer removal** — optionally strip the trailing
           English section (title + abstract + keywords) that many
           CyberLeninka papers append.

    Parameters
    ----------
    remove_keywords_line : bool
        Also remove the ``Ключевые слова: …`` line (default True).
    remove_english_trailer : bool
        Also remove the English duplicate section at the end (default True).
    min_body_len : int
        If removal would leave fewer than this many characters, the
        original text is returned unchanged (safety guard).
    """

    # ── Compiled patterns ─────────────────────────────────────────────
    _RE_ANNOT = re.compile(
        r'Аннотация\s*[.:]\s*',
        re.IGNORECASE,
    )
    _RE_KW_RU = re.compile(
        r'^[ \t]*Ключевые слова\s*[:：\-–—]\s*[^\n]+\.?[ \t]*$',
        re.MULTILINE | re.IGNORECASE,
    )
    _RE_KW_EN = re.compile(
        r'^[ \t]*Key\s*words?\s*[:：]\s*[^\n]+\.?[ \t]*$',
        re.MULTILINE | re.IGNORECASE,
    )
    # English trailer: English title (ALL CAPS or Title Case) followed by
    # authors and "Abstract" near the end of the text.
    _RE_ENGLISH_TRAILER = re.compile(
        r'\n(?=\s*[A-Z][A-Z\s,:&\-]{15,}\n)'  # English title in CAPS
        r'.*',                                  # everything after
        re.DOTALL,
    )

    def __init__(
        self,
        remove_keywords_line: bool = True,
        remove_english_trailer: bool = True,
        min_body_len: int = 200,
    ) -> None:
        self.remove_keywords_line = remove_keywords_line
        self.remove_english_trailer = remove_english_trailer
        self.min_body_len = min_body_len

    # ── Normalisation helpers ─────────────────────────────────────────

    @staticmethod
    def _normalise_ws(text: str) -> str:
        """Collapse all whitespace to single spaces for matching."""
        return re.sub(r'\s+', ' ', text).strip()

    # ── Core removal ──────────────────────────────────────────────────

    def remove(self, text_clean: str, summary: str) -> str:
        """Return *text_clean* with the abstract removed.

        Parameters
        ----------
        text_clean : str
            Document text that has already been through ``TextCleaner``.
        summary : str
            The target summary (= author abstract) for this record.

        Returns
        -------
        str  — cleaned body text without the abstract.
        """
        original = text_clean

        # ── Step 1: Direct match ──────────────────────────────────────
        text_clean = self._remove_by_direct_match(text_clean, summary)

        # ── Step 2: Marker-based fallback ─────────────────────────────
        if self._still_contains_abstract(text_clean, summary):
            text_clean = self._remove_by_marker(text_clean)

        # ── Step 3: Remove "Ключевые слова: ..." line ─────────────────
        if self.remove_keywords_line:
            text_clean = self._RE_KW_RU.sub('', text_clean)
            text_clean = self._RE_KW_EN.sub('', text_clean)

        # ── Step 4: English trailer ───────────────────────────────────
        if self.remove_english_trailer:
            text_clean = self._remove_english_section(text_clean)

        # ── Collapse blank lines left by removals ─────────────────────
        text_clean = re.sub(r'\n{3,}', '\n\n', text_clean).strip()

        # ── Safety guard ──────────────────────────────────────────────
        if len(text_clean) < self.min_body_len:
            logger.warning(
                "AbstractRemover: body too short after removal (%d chars), "
                "keeping original (%d chars)",
                len(text_clean), len(original),
            )
            return original

        return text_clean

    # ── Strategy 1: direct substring match ────────────────────────────

    def _remove_by_direct_match(self, text: str, summary: str) -> str:
        """Try to find *summary* verbatim (or whitespace-normalised) in *text*."""
        if not summary or len(summary) < 30:
            return text

        # First try: literal substring
        idx = text.find(summary)
        if idx >= 0:
            return text[:idx] + text[idx + len(summary):]

        # Second try: use first 80 chars as an anchor (whitespace-normalised)
        anchor = self._normalise_ws(summary[:80])
        norm_text = self._normalise_ws(text)
        idx_norm = norm_text.find(anchor)
        if idx_norm < 0:
            return text  # cannot locate

        # Now find the approximate position in the original text.
        # Walk through original text, building normalised prefix until
        # the normalised position is reached.
        orig_pos = self._map_norm_pos_to_orig(text, idx_norm)
        if orig_pos is None:
            return text

        # Find the end: try to locate the last ~60 chars of summary
        summary_tail = self._normalise_ws(summary[-60:])
        end_norm = norm_text.find(summary_tail, idx_norm)
        if end_norm < 0:
            # Fallback: remove from start to next "Ключевые слова" or next
            # section heading
            end_orig = self._find_next_section(text, orig_pos)
        else:
            end_orig = self._map_norm_pos_to_orig(
                text, end_norm + len(summary_tail)
            )
            if end_orig is None:
                end_orig = self._find_next_section(text, orig_pos)

        return text[:orig_pos] + text[end_orig:]

    @staticmethod
    def _map_norm_pos_to_orig(text: str, norm_pos: int) -> Optional[int]:
        """Map a position in whitespace-normalised text back to original."""
        count = 0      # normalised-space position counter
        in_ws = False
        for i, ch in enumerate(text):
            if ch in (' ', '\t', '\n', '\r'):
                if not in_ws and count > 0:
                    count += 1  # collapsed whitespace = 1 space
                    if count >= norm_pos:
                        return i
                in_ws = True
            else:
                in_ws = False
                count += 1
                if count > norm_pos:
                    return i
        return len(text) if count >= norm_pos else None

    @staticmethod
    def _find_next_section(text: str, start: int) -> int:
        """Find the next section heading or 'Ключевые слова' after *start*."""
        patterns = [
            re.compile(r'\n\s*Ключевые слова', re.IGNORECASE),
            re.compile(r'\n\s*Введение\b', re.IGNORECASE),
            re.compile(r'\n\s*1\.\s+\S'),
            re.compile(r'\n\s*ВВЕДЕНИЕ\b'),
        ]
        best = len(text)
        for pat in patterns:
            m = pat.search(text, start)
            if m and m.start() < best:
                best = m.start()
        return best

    # ── Strategy 2: marker-based removal ──────────────────────────────

    def _remove_by_marker(self, text: str) -> str:
        """Remove text between 'Аннотация' marker and 'Ключевые слова'."""
        m_start = self._RE_ANNOT.search(text)
        if not m_start:
            return text

        # Find the end boundary
        kw_match = self._RE_KW_RU.search(text, m_start.end())
        if kw_match:
            end = kw_match.start()
        else:
            end = self._find_next_section(text, m_start.end())

        return text[:m_start.start()] + text[end:]

    # ── Check if abstract is still present ────────────────────────────

    def _still_contains_abstract(self, text: str, summary: str) -> bool:
        """Quick check: does *text* still contain a significant portion
        of *summary*?"""
        if not summary or len(summary) < 50:
            return False
        anchor = self._normalise_ws(summary[:50])
        return anchor in self._normalise_ws(text)

    # ── English trailer removal ───────────────────────────────────────

    def _remove_english_section(self, text: str) -> str:
        """Strip trailing English title + abstract + keywords section."""
        # Look for "Abstract" in the last 30% of the text
        cutoff = int(len(text) * 0.70)
        tail = text[cutoff:]

        # Find "Abstract" marker in tail
        m_abs = re.search(
            r'\n\s*Abstract\s*[.:]\s*', tail, re.IGNORECASE,
        )
        if not m_abs:
            return text

        # Walk backwards to find the English title (line of mostly Latin
        # uppercase characters)
        lines_before = tail[:m_abs.start()].split('\n')
        eng_start = m_abs.start()  # default: just remove from Abstract

        for i in range(len(lines_before) - 1, -1, -1):
            line = lines_before[i].strip()
            if not line:
                continue
            # Check if line is mostly Latin characters (English title)
            latin_ratio = sum(1 for c in line if c.isascii() and c.isalpha()) / max(1, len(line))
            if latin_ratio > 0.7 and len(line) > 15:
                eng_start = sum(len(l) + 1 for l in lines_before[:i])
                break
            # Stop searching if we hit Cyrillic content
            cyrillic_ratio = sum(1 for c in line if '\u0400' <= c <= '\u04ff') / max(1, len(line))
            if cyrillic_ratio > 0.5:
                break

        return text[:cutoff + eng_start]


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
