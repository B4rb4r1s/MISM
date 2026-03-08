#!/usr/bin/env python
"""
verify_abstract_removal.py — Test AbstractRemover on a few records.

Quick sanity check:
    python scripts/verify_abstract_removal.py \
        --input dataset/dataset-SM-17k.json \
        --n 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import TextCleaner, AbstractRemover


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/dataset-SM-17k.json")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of records to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rng = random.Random(args.seed)
    sampled = rng.sample(raw_data, min(args.n, len(raw_data)))

    cleaner = TextCleaner(remove_service_markers=True)
    remover = AbstractRemover()

    results = {"success": 0, "partial": 0, "failed": 0, "total": 0}

    for i, rec in enumerate(sampled):
        text = rec.get("text", "")
        summary = rec.get("summary", "")
        if not text.strip() or not summary.strip():
            continue

        results["total"] += 1
        cleaned = cleaner.clean(text)
        body = remover.remove(cleaned, summary)

        # Check: first 50 chars of summary should NOT be in body
        anchor = AbstractRemover._normalise_ws(summary[:50])
        still_present = anchor in AbstractRemover._normalise_ws(body)

        len_diff = len(cleaned) - len(body)
        pct_removed = 100.0 * len_diff / max(1, len(cleaned))

        status = "OK" if not still_present else "STILL_PRESENT"
        if not still_present:
            results["success"] += 1
        else:
            results["failed"] += 1

        doc_id = rec.get("doc_id", f"rec_{i}")
        print(f"\n{'='*80}")
        print(f"[{i+1}] doc_id={doc_id}  status={status}")
        print(f"     original:  {len(cleaned):,} chars")
        print(f"     after:     {len(body):,} chars")
        print(f"     removed:   {len_diff:,} chars ({pct_removed:.1f}%)")
        print(f"     summary:   {summary[:100]}...")
        print(f"     body start: {body[:150]}...")
        if still_present:
            print(f"  *** WARNING: abstract still detected in body ***")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {results['success']}/{results['total']} successful, "
          f"{results['failed']} failed")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
