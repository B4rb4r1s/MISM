#!/usr/bin/env python
"""
check_dataset.py — Scan train/val/test JSON splits for NaN and Inf in kw_scores.

Usage:
    python scripts/check_dataset.py --split dataset/splits/train.json
    python scripts/check_dataset.py --split dataset/splits/train.json --fix
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def check_split(path: Path, fix: bool = False) -> int:
    """Return number of records with bad kw_scores. Optionally fix in-place."""
    records = json.loads(path.read_text(encoding="utf-8"))
    bad_indices: list[int] = []

    for i, rec in enumerate(records):
        scores = rec.get("kw_scores", [])
        bad_pos = [
            (j, s) for j, s in enumerate(scores)
            if isinstance(s, (int, float)) and not math.isfinite(float(s))
        ]
        if bad_pos:
            bad_indices.append(i)
            kws = rec.get("keywords", [])
            print(f"  Record {i}: {len(bad_pos)} bad score(s)")
            for j, s in bad_pos[:5]:
                kw = kws[j] if j < len(kws) else "?"
                print(f"    slot {j}: kw={kw!r:30s}  score={s!r}")
            if fix:
                for j, _ in bad_pos:
                    rec["kw_scores"][j] = 0.0

    print(f"\n{'[FIXED] ' if fix else ''}Records with NaN/Inf kw_scores: "
          f"{len(bad_indices)} / {len(records)}")

    if fix and bad_indices:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=None),
                        encoding="utf-8")
        print(f"Saved fixed file: {path}")

    return len(bad_indices)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True,
                   help="Path to JSON split (train/val/test)")
    p.add_argument("--fix", action="store_true",
                   help="Replace NaN/Inf scores with 0.0 and overwrite the file")
    args = p.parse_args()

    path = Path(args.split)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {path} ...")
    n_bad = check_split(path, fix=args.fix)
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
