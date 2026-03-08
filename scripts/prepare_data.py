#!/usr/bin/env python3
"""
prepare_data.py — Preprocess and split the CyberLeninka dataset.

Steps
-----
1. Load ``dataset/dataset-SM-17k.json``.
2. Validate each record (non-empty text, summary ≥ min_summary_len chars,
   non-empty keywords_original).
3. Clean document text (BOM, whitespace, service markers).
4. Parse ``keywords_original`` → lowercase ``[(phrase, 1.0), ...]``.
5. Assign a ``summary_bucket`` label for stratified splitting.
6. Stratified train / val / test split.
7. Save splits to ``dataset/splits/{train,val,test}.json`` and a
   ``stats.json`` summary.

Usage
-----
    source .mism/bin/activate
    python scripts/prepare_data.py \\
        --input  dataset/dataset-SM-17k.json \\
        --output dataset/splits \\
        [--min_summary_len 100] \\
        [--val_ratio 0.10] \\
        [--test_ratio 0.10] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import TextCleaner, AbstractRemover, KeywordProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary_bucket(length_chars: int) -> str:
    """Coarse-grained label used for stratified splitting."""
    if length_chars < 300:
        return "short"
    if length_chars < 800:
        return "medium"
    return "long"


def stratified_split(
    records: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split *records* into (train, val, test) preserving bucket proportions."""
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict]] = {}
    for rec in records:
        buckets.setdefault(rec["summary_bucket"], []).append(rec)

    train_all: List[Dict] = []
    val_all:   List[Dict] = []
    test_all:  List[Dict] = []

    for bucket, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        test_all.extend(items[:n_test])
        val_all.extend(items[n_test: n_test + n_val])
        train_all.extend(items[n_test + n_val:])
        logger.info(
            "  Bucket %-8s │ total=%5d  train=%5d  val=%4d  test=%4d",
            bucket, n,
            n - n_test - n_val, n_val, n_test,
        )

    return train_all, val_all, test_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess and split the CyberLeninka dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",           default="dataset/dataset-SM-17k.json",
                        help="Path to the raw JSON dataset.")
    parser.add_argument("--output",          default="dataset/splits",
                        help="Directory for output split files.")
    parser.add_argument("--min_summary_len", type=int, default=100,
                        help="Drop records with summary shorter than this (chars).")
    parser.add_argument("--val_ratio",       type=float, default=0.10)
    parser.add_argument("--test_ratio",      type=float, default=0.10)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--no-strip-abstract", action="store_true", default=False,
                        help="Do NOT remove the abstract from source text. "
                             "By default the abstract is stripped so that the "
                             "model cannot learn a trivial copy strategy.")
    args = parser.parse_args()

    strip_abstract = not args.no_strip_abstract
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    logger.info("Loading %s …", input_path)
    with open(input_path, "r", encoding="utf-8") as fh:
        raw_data: List[Dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d raw records", len(raw_data))

    # ── Preprocess & filter ─────────────────────────────────────────────
    cleaner = TextCleaner(remove_service_markers=True)
    abs_remover = AbstractRemover() if strip_abstract else None
    kw_proc = KeywordProcessor()
    records: List[Dict[str, Any]] = []
    skipped: Counter = Counter()
    abstract_removed_count = 0

    if strip_abstract:
        logger.info("Abstract stripping is ENABLED")
    else:
        logger.info("Abstract stripping is DISABLED (--no-strip-abstract)")

    for rec in raw_data:
        # ── Validate required fields ────────────────────────────────────
        text = rec.get("text", "")
        if not isinstance(text, str) or not text.strip():
            skipped["no_text"] += 1
            continue

        summary = rec.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            skipped["no_summary"] += 1
            continue
        summary = summary.strip()
        if len(summary) < args.min_summary_len:
            skipped["summary_too_short"] += 1
            continue

        kw_orig = rec.get("keywords_original", "")
        if not kw_orig:
            skipped["no_keywords_original"] += 1
            continue

        kws = kw_proc.process(kw_orig)   # List[(str, float)]
        if not kws:
            skipped["empty_keywords"] += 1
            continue

        # ── Clean text ────────────────────────────────────────────────────
        cleaned = cleaner.clean(text)

        # ── Strip abstract from source ────────────────────────────────────
        if abs_remover is not None:
            cleaned_before = len(cleaned)
            cleaned = abs_remover.remove(cleaned, summary)
            if len(cleaned) < cleaned_before:
                abstract_removed_count += 1

        # ── Build processed record ──────────────────────────────────────
        records.append({
            "doc_id":              rec.get("doc_id", ""),
            "title":               rec.get("title", ""),
            "text_clean":          cleaned,
            "keywords_processed":  kws,
            "summary":             summary,
            "summary_bucket":      _summary_bucket(len(summary)),
        })

    logger.info(
        "After filtering: %d records kept, %d skipped %s",
        len(records), sum(skipped.values()), dict(skipped),
    )
    if strip_abstract:
        logger.info(
            "Abstract removed from %d / %d records (%.1f%%)",
            abstract_removed_count, len(records),
            100.0 * abstract_removed_count / max(1, len(records)),
        )

    # ── Split ────────────────────────────────────────────────────────────
    logger.info(
        "Splitting — val=%.0f%%  test=%.0f%%  seed=%d",
        args.val_ratio * 100, args.test_ratio * 100, args.seed,
    )
    train, val, test = stratified_split(
        records, args.val_ratio, args.test_ratio, args.seed
    )
    logger.info(
        "Split result │ train=%d  val=%d  test=%d  total=%d",
        len(train), len(val), len(test), len(train) + len(val) + len(test),
    )

    # ── Save splits ──────────────────────────────────────────────────────
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{split_name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(split_data, fh, ensure_ascii=False, indent=2)
        logger.info("Saved %-5s → %s  (%d records)", split_name, path, len(split_data))

    # ── Save stats ───────────────────────────────────────────────────────
    stats = {
        "total_loaded": len(raw_data),
        "total_valid":  len(records),
        "skipped":      dict(skipped),
        "abstract_stripped": strip_abstract,
        "abstract_removed_count": abstract_removed_count if strip_abstract else 0,
        "splits": {
            name: {
                "count": len(data),
                "bucket_distribution": dict(Counter(r["summary_bucket"] for r in data)),
            }
            for name, data in [("train", train), ("val", val), ("test", test)]
        },
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    logger.info("Stats → %s", stats_path)

    logger.info("prepare_data.py finished successfully.")


if __name__ == "__main__":
    main()
