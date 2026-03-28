#!/usr/bin/env python
"""
export_to_db.py — Export MISM-generated summaries into CSV and SQLite.

Reads a JSON file with generated summaries (from generate_samples_480.py)
and adds a "MISM" column to the existing CSV and SQLite database.

Matching is done by target-summary text (first 80 chars) since doc_ids
in dataset-480.json don't directly map to CSV row IDs.

Usage
-----
    python scripts/export_to_db.py \
        --results results/samples_480_full.json \
        --csv dataset/data-full+LLM.csv \
        --db  dataset/data-full+LLM.db \
        --column MISM

    # Dry-run: show what would be done without writing
    python scripts/export_to_db.py \
        --results results/samples_480_full.json \
        --csv dataset/data-full+LLM.csv \
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_to_db")


# ── Tag mapping ─────────────────────────────────────────────────────────
# dataset-480 doc_id prefixes -> CSV tag values
TAG_MAP = {
    "chem": "chemistry",
    "law": "law",
    "med": "medicine",
    "journalism": "journalism",
    "ecnm": "economics",
    "hist": "history",
    "ling": "linguistics",
    "inf": "it",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export MISM summaries to CSV and SQLite",
    )
    p.add_argument(
        "--results", required=True,
        help="Path to JSON with generated summaries "
             "(from generate_samples_480.py --all)",
    )
    p.add_argument(
        "--dataset", default="dataset/dataset-480.json",
        help="Path to dataset-480 JSON (for reference text matching)",
    )
    p.add_argument(
        "--csv", default="dataset/data-full+LLM.csv",
        help="Path to CSV file to update",
    )
    p.add_argument(
        "--db", default="dataset/data-full+LLM.db",
        help="Path to SQLite DB to update (optional, skipped if not exists)",
    )
    p.add_argument(
        "--column", default="MISM",
        help="Column name for MISM summaries (default: MISM)",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Show what would be done without writing files",
    )
    return p.parse_args()


def build_ref_to_generated(results_path: str, dataset_path: str) -> dict:
    """Build a mapping from reference text (first 80 chars) -> generated summary.

    Uses dataset-480.json for the original reference text (target-summary),
    and the results JSON for the generated text.
    """
    # Load results
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Build doc_id -> generated mapping
    gen_by_id = {}
    for r in results:
        doc_id = r.get("doc_id", "")
        # Prefer 'generated' (post-processed), fall back to 'generated_raw'
        gen_text = r.get("generated", r.get("generated_raw", ""))
        if doc_id and gen_text:
            gen_by_id[doc_id] = gen_text

    logger.info("Loaded %d generated summaries from results", len(gen_by_id))

    # Load dataset-480 for reference texts
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Build ref_prefix -> generated
    ref_to_gen = {}
    for rec in dataset:
        doc_id = rec.get("doc_id", "")
        ref_text = rec.get("target-summary", "")
        if doc_id in gen_by_id and ref_text:
            # Use first 80 chars of reference as key
            key = ref_text[:80].strip()
            ref_to_gen[key] = gen_by_id[doc_id]

    logger.info("Built %d reference->generated mappings", len(ref_to_gen))
    return ref_to_gen


def update_csv(csv_path: str, ref_to_gen: dict, column: str,
               dry_run: bool = False) -> int:
    """Add/update a column in CSV with MISM summaries.

    Returns number of matched rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return 0

    # Read CSV
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    logger.info("Read %d rows from CSV, columns: %d", len(rows), len(fieldnames))

    # Add column if not present
    if column not in fieldnames:
        fieldnames.append(column)
        logger.info("Adding new column: %s", column)

    # Match and fill
    matched = 0
    for row in rows:
        ref_text = row.get("target_summary", "")
        key = ref_text[:80].strip()
        if key in ref_to_gen:
            row[column] = ref_to_gen[key]
            matched += 1
        elif column not in row:
            row[column] = ""

    logger.info("Matched %d / %d CSV rows", matched, len(rows))

    if dry_run:
        logger.info("[DRY RUN] Would write %d rows to %s", len(rows), csv_path)
        return matched

    # Write back
    backup_path = csv_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        import shutil
        shutil.copy2(csv_path, backup_path)
        logger.info("Backup saved: %s", backup_path)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Written %d rows to %s (column: %s)", len(rows), csv_path, column)
    return matched


def update_sqlite(db_path: str, ref_to_gen: dict, column: str,
                  dry_run: bool = False) -> int:
    """Add/update a column in SQLite with MISM summaries.

    Returns number of matched rows.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("SQLite DB not found: %s — skipping", db_path)
        return 0

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Find the main table name
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        logger.error("No tables found in DB")
        conn.close()
        return 0

    table = tables[0]
    logger.info("Using table: %s", table)

    # Check existing columns
    cur.execute(f"PRAGMA table_info({table})")
    existing_cols = {r[1] for r in cur.fetchall()}

    if column not in existing_cols:
        if dry_run:
            logger.info("[DRY RUN] Would add column %s to table %s", column, table)
        else:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT DEFAULT ''")
            logger.info("Added column %s to table %s", column, table)

    # Get all rows with their target_summary
    cur.execute(f"SELECT rowid, target_summary FROM {table}")
    db_rows = cur.fetchall()

    matched = 0
    updates = []
    for rowid, ref_text in db_rows:
        if ref_text:
            key = ref_text[:80].strip()
            if key in ref_to_gen:
                updates.append((ref_to_gen[key], rowid))
                matched += 1

    logger.info("Matched %d / %d DB rows", matched, len(db_rows))

    if dry_run:
        logger.info("[DRY RUN] Would update %d rows in %s", matched, db_path)
        conn.close()
        return matched

    # Batch update
    cur.executemany(
        f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
        updates,
    )
    conn.commit()
    conn.close()

    logger.info("Updated %d rows in %s (column: %s)", matched, db_path, column)
    return matched


def main() -> None:
    args = parse_args()

    # Build mapping
    ref_to_gen = build_ref_to_generated(args.results, args.dataset)

    if not ref_to_gen:
        logger.error("No mappings built. Check results and dataset files.")
        sys.exit(1)

    # Update CSV
    csv_matched = update_csv(args.csv, ref_to_gen, args.column, args.dry_run)

    # Update SQLite
    db_matched = update_sqlite(args.db, ref_to_gen, args.column, args.dry_run)

    # Summary
    print()
    print("=" * 60)
    print(f"  Generated summaries loaded: {len(ref_to_gen)}")
    print(f"  CSV matched:               {csv_matched}")
    print(f"  SQLite matched:            {db_matched}")
    if args.dry_run:
        print(f"  Mode:                      DRY RUN (no files written)")
    print("=" * 60)


if __name__ == "__main__":
    main()
