#!/usr/bin/env python3
"""
filter_train_data.py — Filter training data to remove samples where
the reference summary leaks into the source text.

Two filters:
  1. Abstract leak: first 50 chars of summary found verbatim in text_clean
     (same check as verify_abstract_removal.py)
  2. High n-gram overlap: > threshold of summary's word 4-grams appear
     in text_clean → model can learn extractive shortcut

Usage:
    source .mism/bin/activate
    python scripts/filter_train_data.py \
        --input  dataset/splits/train.json \
        --output dataset/splits/train_clean.json \
        [--overlap-threshold 0.55] \
        [--ngram-size 4] \
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return _WS.sub(" ", text.lower()).strip()


def word_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """Extract word-level n-grams from normalized text."""
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def ngram_overlap_ratio(summary: str, source: str, n: int) -> float:
    """
    Compute fraction of summary's word n-grams that appear in source.
    Returns value in [0.0, 1.0].
    """
    summary_norm = normalise(summary)
    source_norm = normalise(source)

    summary_ngrams = word_ngrams(summary_norm, n)
    if not summary_ngrams:
        return 0.0

    source_ngrams_set: Set[Tuple[str, ...]] = set(word_ngrams(source_norm, n))

    matches = sum(1 for ng in summary_ngrams if ng in source_ngrams_set)
    return matches / len(summary_ngrams)


def abstract_leak_check(text_clean: str, summary: str) -> bool:
    """
    Check if the first 50 chars of summary are still present in text_clean.
    Same logic as verify_abstract_removal.py.
    """
    anchor = normalise(summary[:50])
    return anchor in normalise(text_clean)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter training data to remove summary-source overlap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="dataset/splits/train.json",
        help="Path to original train split",
    )
    parser.add_argument(
        "--output", default="dataset/splits/train_clean.json",
        help="Path for filtered output",
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.55,
        help="Max allowed fraction of summary 4-grams found in source. "
             "Samples above this are removed.",
    )
    parser.add_argument(
        "--ngram-size", type=int, default=4,
        help="N-gram size for overlap calculation",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print details for each filtered sample",
    )
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────
    input_path = Path(args.input)
    logger.info("Loading %s ...", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    logger.info("Loaded %d training samples", len(data))

    # ── Filter ───────────────────────────────────────────────────────
    kept: List[Dict[str, Any]] = []
    removed_leak: List[str] = []
    removed_overlap: List[str] = []
    overlap_values: List[float] = []

    for i, rec in enumerate(data):
        doc_id = rec.get("doc_id", f"rec_{i}")
        text_clean = rec.get("text_clean", "")
        summary = rec.get("summary", "")

        # Filter 1: abstract leak
        if abstract_leak_check(text_clean, summary):
            removed_leak.append(doc_id)
            if args.verbose:
                logger.info(
                    "  LEAK     [%d] doc_id=%s  summary[:60]=%s",
                    i, doc_id, summary[:60],
                )
            continue

        # Filter 2: high n-gram overlap
        overlap = ngram_overlap_ratio(summary, text_clean, args.ngram_size)
        overlap_values.append(overlap)

        if overlap > args.overlap_threshold:
            removed_overlap.append(doc_id)
            if args.verbose:
                logger.info(
                    "  OVERLAP  [%d] doc_id=%s  overlap=%.2f  summary[:60]=%s",
                    i, doc_id, overlap, summary[:60],
                )
            continue

        kept.append(rec)

    # ── Statistics ───────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("FILTERING RESULTS")
    logger.info("=" * 70)
    logger.info("  Original samples:        %d", len(data))
    logger.info("  Removed (abstract leak): %d", len(removed_leak))
    logger.info("  Removed (high overlap):  %d  (threshold=%.2f)",
                len(removed_overlap), args.overlap_threshold)
    logger.info("  Total removed:           %d  (%.1f%%)",
                len(removed_leak) + len(removed_overlap),
                100.0 * (len(removed_leak) + len(removed_overlap)) / max(1, len(data)))
    logger.info("  Kept:                    %d  (%.1f%%)",
                len(kept), 100.0 * len(kept) / max(1, len(data)))

    if overlap_values:
        import statistics
        logger.info("")
        logger.info("  Overlap distribution (among non-leak samples):")
        logger.info("    min=%.3f  median=%.3f  mean=%.3f  max=%.3f",
                     min(overlap_values),
                     statistics.median(overlap_values),
                     statistics.mean(overlap_values),
                     max(overlap_values))

        # Histogram
        buckets = [0] * 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        for v in overlap_values:
            idx = min(int(v * 10), 9)
            buckets[idx] += 1
        logger.info("    Histogram:")
        for b in range(10):
            low = b / 10
            high = (b + 1) / 10
            bar = "█" * (buckets[b] * 50 // max(1, max(buckets)))
            logger.info("      [%.1f-%.1f) %5d %s", low, high, buckets[b], bar)

    # ── Bucket distribution ──────────────────────────────────────────
    orig_buckets = Counter(r["summary_bucket"] for r in data)
    kept_buckets = Counter(r["summary_bucket"] for r in kept)
    logger.info("")
    logger.info("  Bucket distribution:")
    for bucket in sorted(set(orig_buckets) | set(kept_buckets)):
        orig = orig_buckets.get(bucket, 0)
        k = kept_buckets.get(bucket, 0)
        logger.info("    %-8s  orig=%5d  kept=%5d  removed=%4d",
                     bucket, orig, k, orig - k)

    # ── Save ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    logger.info("")
    logger.info("Saved %d clean samples → %s", len(kept), output_path)

    # ── Save removal log ─────────────────────────────────────────────
    log_path = output_path.with_suffix(".removed.json")
    removal_log = {
        "total_original": len(data),
        "total_kept": len(kept),
        "total_removed": len(removed_leak) + len(removed_overlap),
        "removed_abstract_leak": removed_leak,
        "removed_high_overlap": removed_overlap,
        "overlap_threshold": args.overlap_threshold,
        "ngram_size": args.ngram_size,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(removal_log, f, ensure_ascii=False, indent=2)
    logger.info("Removal log → %s", log_path)


if __name__ == "__main__":
    main()
