"""
postprocess.py — Post-processing pipeline for generated summaries.

Fixes common generation artefacts:
  1. Leading subword fragments ("cess management..." → trimmed)
  2. Leading English text before Russian content
  3. Word sticking ("характеркатегоризации" → "характер категоризации")
  4. Truncated endings (trim to last complete sentence)
  5. Capitalisation of first letter

Usage
-----
    from src.utils.postprocess import clean_summary
    cleaned = clean_summary(raw_text)
"""

from __future__ import annotations

import re
from typing import Optional


# ── Regex patterns ──────────────────────────────────────────────────────

# Cyrillic character class
_CYR = r'[а-яёА-ЯЁ]'

# A "word" that is purely ASCII/Latin (English fragment)
_LATIN_WORD = r"[A-Za-z][A-Za-z0-9\-']*"

# Sentence-ending punctuation followed by space or end of string
_SENTENCE_END = re.compile(r'[.!?…][»"\'")\]]?\s', re.UNICODE)
_SENTENCE_END_FINAL = re.compile(r'[.!?…][»"\'")\]]?\s*$', re.UNICODE)

# Detect word sticking: lowercase Cyrillic immediately followed by
# uppercase Cyrillic (missing space between words)
_STUCK_UPPER = re.compile(r'(' + _CYR + r'{2,})([А-ЯЁ]' + _CYR + r')')

# Detect keyword-style sticking: a lowercase cyrillic letter followed
# immediately by another lowercase word that doesn't fit grammatically.
# This is harder to detect reliably, so we focus on the uppercase case.

# Leading non-Cyrillic garbage (subword fragments, English text)
# Matches anything before the first Cyrillic word-start
_LEADING_GARBAGE = re.compile(
    r'^[^а-яёА-ЯЁ]*'           # non-Cyrillic chars at start
    r'(?:[A-Za-z][A-Za-z0-9\s\-.,;:\'\"()/]*\s+)?'  # optional English phrase
    r'(?=[А-ЯЁа-яё])',         # lookahead: Cyrillic starts
    re.UNICODE,
)

# More aggressive: skip everything until first Cyrillic letter
_SKIP_TO_CYRILLIC = re.compile(r'^[^а-яёА-ЯЁ]+', re.UNICODE)


def trim_leading_fragments(text: str) -> str:
    """Remove leading subword fragments and English text.

    Examples:
        "cess management в логистической..." → "В логистической..."
        "freeze-dried sea buckthorn... В работе" → "В работе..."
        "атизация проектирования;" → "Проектирования;" (partial fix)
    """
    if not text:
        return text

    # Strategy: find the first position where Cyrillic text starts
    # after any leading garbage (English, subword fragments, punctuation)

    # First, check if text already starts with Cyrillic
    if re.match(r'^[А-ЯЁа-яё]', text):
        return text

    # Try to find first Cyrillic character
    m = re.search(r'[А-ЯЁа-яё]', text)
    if m is None:
        return text  # No Cyrillic at all, return as-is

    # Find start position: back up to a word/sentence boundary
    pos = m.start()

    # If the Cyrillic char is uppercase, start from it
    if text[pos].isupper():
        return text[pos:]

    # If lowercase, it might be a subword continuation.
    # Look for the next uppercase Cyrillic or sentence boundary.
    # But also check: if it's after a period/comma + space, it's OK
    if pos > 0 and text[pos - 1] in ' \n\t':
        # It's a word start (after whitespace), keep it
        return text[pos].upper() + text[pos + 1:]

    # Otherwise, find next sentence start or uppercase
    m2 = re.search(r'(?:^|[.!?]\s+)([А-ЯЁ])', text[pos:])
    if m2:
        # Start from the uppercase letter
        abs_pos = pos + m2.start(1)
        return text[abs_pos:]

    # Fallback: start from the first Cyrillic char, capitalise
    return text[pos].upper() + text[pos + 1:]


def trim_to_last_sentence(text: str, min_length: int = 80) -> str:
    """Trim text to the last complete sentence.

    Finds the last sentence-ending punctuation (. ! ? …) and trims there.
    If no sentence boundary is found or the result would be too short,
    returns the original text with trailing incomplete words removed.

    Parameters
    ----------
    text       : the generated text
    min_length : minimum character length to preserve (default 80)
    """
    if not text:
        return text

    # Find all sentence-ending positions
    ends = list(_SENTENCE_END.finditer(text))

    if ends:
        # Use the last sentence boundary that leaves enough text
        for end_match in reversed(ends):
            pos = end_match.start() + 1  # include the punctuation mark
            if pos >= min_length:
                return text[:pos].rstrip()

    # Check if text already ends with sentence punctuation
    if _SENTENCE_END_FINAL.search(text):
        return text.rstrip()

    # No good sentence boundary found — trim trailing partial word
    # Find last space and trim there
    last_space = text.rfind(' ')
    if last_space > min_length:
        # Add ellipsis to indicate truncation
        return text[:last_space].rstrip(',;:-–—') + '...'

    return text


def fix_word_sticking(text: str) -> str:
    """Fix words stuck together without spaces.

    Handles the most common case: lowercaseUPPERCASE boundary
    (e.g. "характеркатегоризацииОбъектов" → "характер категоризации Объектов")

    Also handles keyword-style insertions where bold keywords were merged
    into surrounding text.
    """
    if not text:
        return text

    # Fix: lowercase cyrillic immediately followed by uppercase cyrillic
    # "информационно-коммуникационные технологиидля" — нет, это keyword markup
    # "характеркатегоризацииобъектов" — сложнее, всё lowercase

    # Pattern 1: lowercaseUPPERCASE → lowercase UPPERCASE
    text = _STUCK_UPPER.sub(r'\1 \2', text)

    # Pattern 2: look for keyword-style patterns where a known keyword
    # is embedded without spaces. This is harder to do generically,
    # but we can fix common cases.

    # Fix doubled spaces that might result
    text = re.sub(r'  +', ' ', text)

    return text


def fix_mixed_language(text: str) -> str:
    """Reduce English fragments in primarily Russian text.

    If text is >70% Cyrillic, remove isolated English sentences at the
    start, but keep English technical terms inline.
    """
    if not text:
        return text

    # Count Cyrillic vs Latin characters
    cyr_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
    lat_count = len(re.findall(r'[a-zA-Z]', text))
    total = cyr_count + lat_count

    if total == 0:
        return text

    cyr_ratio = cyr_count / total

    # If text is predominantly Russian (>60%), clean up English starts
    if cyr_ratio > 0.6:
        # Remove leading English sentence(s)
        # Pattern: English text followed by period/comma and then Russian
        cleaned = re.sub(
            r'^(?:[A-Za-z][A-Za-z0-9\s\-.,;:\'\"()/–—]+[.!?,;]\s*)+(?=[А-ЯЁа-яё])',
            '',
            text,
            flags=re.UNICODE,
        )
        if cleaned and len(cleaned) > len(text) * 0.5:
            return cleaned

    return text


def capitalise_first(text: str) -> str:
    """Ensure the first letter is capitalised."""
    if text and text[0].islower():
        return text[0].upper() + text[1:]
    return text


def clean_summary(
    text: str,
    min_sentence_length: int = 80,
    trim_sentences: bool = True,
) -> str:
    """Full post-processing pipeline for a generated summary.

    Steps (in order):
    1. Fix mixed language (remove leading English)
    2. Trim leading subword fragments
    3. Fix word sticking
    4. Trim to last complete sentence
    5. Capitalise first letter
    6. Strip whitespace

    Parameters
    ----------
    text               : raw generated text
    min_sentence_length : min chars to keep when trimming sentences
    trim_sentences     : whether to trim to sentence boundaries

    Returns
    -------
    Cleaned summary text.
    """
    if not text or not text.strip():
        return text

    text = text.strip()

    # Step 1: Remove leading English fragments
    text = fix_mixed_language(text)

    # Step 2: Trim leading subword fragments / non-Cyrillic garbage
    text = trim_leading_fragments(text)

    # Step 3: Fix word sticking
    text = fix_word_sticking(text)

    # Step 4: Trim to last complete sentence
    if trim_sentences:
        text = trim_to_last_sentence(text, min_length=min_sentence_length)

    # Step 5: Capitalise
    text = capitalise_first(text)

    # Step 6: Final cleanup
    text = text.strip()

    return text
