#!/usr/bin/env python
"""
test_postprocess.py — Test post-processing on existing generated samples.

Reads a samples JSON file, applies post-processing, and shows before/after
comparison for each sample. No model loading required.

Usage
-----
    python scripts/test_postprocess.py results/samples_val.json
    python scripts/test_postprocess.py results/samples_val.json --save results/samples_val_pp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.utils.postprocess import (
    clean_summary,
    trim_leading_fragments,
    fix_mixed_language,
    fix_word_sticking,
    trim_to_last_sentence,
    capitalise_first,
)


def show_diff(label: str, before: str, after: str, max_chars: int = 300):
    """Show before/after comparison for a single transform."""
    if before == after:
        return
    print(f"  [{label}]")
    # Show first difference
    for i, (a, b) in enumerate(zip(before, after)):
        if a != b:
            ctx_start = max(0, i - 30)
            print(f"    BEFORE: ...{before[ctx_start:ctx_start+80]}...")
            print(f"    AFTER:  ...{after[ctx_start:ctx_start+80]}...")
            break
    else:
        # Length difference
        if len(before) > len(after):
            print(f"    Trimmed: {len(before)} → {len(after)} chars")
        else:
            print(f"    Extended: {len(before)} → {len(after)} chars")


def main():
    parser = argparse.ArgumentParser(description="Test post-processing pipeline")
    parser.add_argument("input", help="Path to samples JSON file")
    parser.add_argument("--save", default=None, help="Save processed results to JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show step-by-step transforms")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {args.input}")
    print("=" * 80)

    changed_count = 0
    total_chars_before = 0
    total_chars_after = 0

    for i, sample in enumerate(samples):
        gen_raw = sample["generated"]
        gen_clean = clean_summary(gen_raw)

        total_chars_before += len(gen_raw)
        total_chars_after += len(gen_clean)

        if gen_raw != gen_clean:
            changed_count += 1
            doc_id = sample.get("id", sample.get("doc_id", f"sample_{i}"))
            print(f"\n{'─'*80}")
            print(f"[{i+1}/{len(samples)}] {doc_id}")
            print(f"{'─'*80}")

            if args.verbose:
                # Show step-by-step
                t1 = fix_mixed_language(gen_raw.strip())
                t2 = trim_leading_fragments(t1)
                t3 = fix_word_sticking(t2)
                t4 = trim_to_last_sentence(t3, min_length=80)
                t5 = capitalise_first(t4)

                show_diff("fix_mixed_language", gen_raw.strip(), t1)
                show_diff("trim_leading_fragments", t1, t2)
                show_diff("fix_word_sticking", t2, t3)
                show_diff("trim_to_last_sentence", t3, t4)
                show_diff("capitalise_first", t4, t5)

            print(f"\n  RAW ({len(gen_raw)} chars):")
            print(f"    {gen_raw[:200]}...")
            print(f"\n  CLEAN ({len(gen_clean)} chars):")
            print(f"    {gen_clean[:200]}...")

        # Update sample with cleaned version
        sample["generated_raw"] = gen_raw
        sample["generated"] = gen_clean

    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"  Total samples:    {len(samples)}")
    print(f"  Changed:          {changed_count} ({100*changed_count/max(1,len(samples)):.1f}%)")
    print(f"  Avg chars before: {total_chars_before/max(1,len(samples)):.0f}")
    print(f"  Avg chars after:  {total_chars_after/max(1,len(samples)):.0f}")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
