#!/usr/bin/env python
"""
generate_samples_480.py — Generate summaries on the external dataset-480.

Dataset-480 has a different JSON structure from the training data:
    doc_id          : str               e.g. "chem_1"
    text            : str               full paper text (body, no abstract)
    target-summary  : str               reference summary
    keywords        : List[Dict]        [{"surface_form": "...", "score": 0.xxx}, ...]

This script converts records to the format expected by the MISM pipeline
and runs generation on a random sample (default 15 documents).

Usage
-----
    python scripts/generate_samples_480.py \
        --config configs/gazeta_2stage.yaml \
        --checkpoint checkpoints/gazeta_2stage/best.pt \
        --dataset dataset/dataset-480.json \
        --n 15 \
        --output results/samples_480.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

# ── Ensure project root is on PYTHONPATH ──────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.collator import DataCollatorForSummarization
from src.models.dual_encoder_summarizer import DualEncoderSummarizer
from src.training.checkpoint import load_checkpoint
from src.training.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gen480")


# ─────────────────────────────────────────────────────────────────────────
# Format conversion
# ─────────────────────────────────────────────────────────────────────────

def convert_480_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dataset-480 record to the format expected by the collator.

    Expected collator input keys:
        text_clean  : str
        keywords    : List[str]
        kw_scores   : List[float]
        summary     : str
        doc_id      : str
    """
    # Keywords: list of {"surface_form": ..., "score": ...}
    kw_raw = raw.get("keywords", [])
    # Sort by score descending
    kw_sorted = sorted(kw_raw, key=lambda x: x.get("score", 0.0), reverse=True)

    return {
        "doc_id":     raw.get("doc_id", ""),
        "text_clean": raw.get("text", ""),
        "keywords":   [kw["surface_form"] for kw in kw_sorted],
        "kw_scores":  [float(kw["score"]) for kw in kw_sorted],
        "summary":    raw.get("target-summary", ""),
    }


# ─────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate summaries on dataset-480 (external evaluation)",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g. configs/gazeta_2stage.yaml)",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint (.pt)",
    )
    p.add_argument(
        "--dataset", default="dataset/dataset-480.json",
        help="Path to dataset-480 JSON file",
    )
    p.add_argument(
        "--n", type=int, default=15,
        help="Number of samples to generate (default: 15)",
    )
    p.add_argument(
        "--output", default="results/samples_480.json",
        help="Path for output JSON (default: results/samples_480.json)",
    )
    p.add_argument(
        "--max-length", type=int, default=256,
        help="Maximum generation length in tokens (default: 256)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    p.add_argument(
        "--src-max-chars", type=int, default=2000,
        help="Truncate source text in output JSON to this many chars "
             "(0 = no truncation, default: 2000)",
    )
    p.add_argument(
        "--device", default=None,
        help="Device: 'cuda', 'cuda:0', 'cpu' (default: auto-detect)",
    )
    p.add_argument(
        "--bypass-kal", action="store_true", default=False,
        help="Diagnostic mode: skip KAL, use T5 decoder -> lm_head directly.",
    )

    # ── Decoding parameters ─────────────────────────────────────────
    p.add_argument(
        "--num-beams", type=int, default=1,
        help="Beam width for beam search (1 = greedy, default: 1)",
    )
    p.add_argument(
        "--min-length", type=int, default=0,
        help="Suppress EOS before this many tokens (default: 0)",
    )
    p.add_argument(
        "--length-penalty", type=float, default=1.0,
        help="Beam search length normalisation exponent "
             "(>1 favours longer, default: 1.0)",
    )
    p.add_argument(
        "--repetition-penalty", type=float, default=1.2,
        help="Repetition penalty (>1 discourages repetition, default: 1.2)",
    )
    p.add_argument(
        "--no-repeat-ngram-size", type=int, default=4,
        help="Block repeated n-grams of this size (default: 4)",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Config ────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Device ────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── Load dataset-480 ──────────────────────────────────────────────
    ds_path = Path(args.dataset)
    logger.info("Loading dataset from %s ...", ds_path)
    with open(ds_path, "r", encoding="utf-8") as f:
        raw_records: List[Dict[str, Any]] = json.load(f)
    logger.info("Loaded %d raw records", len(raw_records))

    # Convert to internal format
    records = [convert_480_record(r) for r in raw_records]

    # Filter out records with empty text or summary
    records = [
        r for r in records
        if r["text_clean"].strip()
        and r["summary"].strip()
        and len(r["summary"]) >= 50
        and len(r["keywords"]) > 0
    ]
    logger.info("After filtering: %d usable records", len(records))

    # ── Sample ────────────────────────────────────────────────────────
    n = min(args.n, len(records))
    sampled = random.sample(records, n)
    logger.info("Selected %d samples (seed=%d)", n, args.seed)

    # ── Tokeniser ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Collator ──────────────────────────────────────────────────────
    collator = DataCollatorForSummarization(
        tokenizer=tokenizer,
        max_kw=cfg.max_kw,
        kw_max_len=cfg.kw_max_len,
        window_size=cfg.window_size,
        window_overlap=cfg.window_overlap,
        max_windows=cfg.max_windows,
        max_summary_tokens=cfg.max_summary_tokens,
    )

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Loading model: %s", cfg.model_name)
    model = DualEncoderSummarizer.from_pretrained(
        cfg.model_name,
        hidden_size=cfg.hidden_size,
        window_overlap=cfg.window_overlap,
        max_src_len=cfg.max_src_len,
        dropout=cfg.dropout,
    )

    load_checkpoint(path=args.checkpoint, model=model)
    model.to(device)
    model.eval()
    if args.bypass_kal:
        logger.info(
            "*** DIAGNOSTIC MODE: bypass_kal=True — KAL is skipped ***"
        )
    logger.info("Model loaded and moved to %s", device)

    # ── Generate ──────────────────────────────────────────────────────
    results: List[dict] = []
    t0 = time.time()

    for i, sample in enumerate(sampled):
        # Collate single sample -> batch of 1
        batch = collator([sample])
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Generate
        gen_ids = model.generate(
            input_windows=batch["input_windows"],
            window_attention_mask=batch["window_attention_mask"],
            kw_input_ids=batch["kw_input_ids"],
            kw_attention_mask=batch["kw_attention_mask"],
            kw_scores=batch["kw_scores"],
            kw_mask=batch["kw_mask"],
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            bypass_kal=args.bypass_kal,
        )

        # Decode
        gen_text = tokenizer.decode(
            gen_ids[0], skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        ref_text = sample["summary"]
        src_text = sample["text_clean"]
        keywords = sample["keywords"]
        doc_id   = sample["doc_id"]

        if args.src_max_chars > 0 and len(src_text) > args.src_max_chars:
            src_display = src_text[: args.src_max_chars] + " [...]"
        else:
            src_display = src_text

        record = {
            "doc_id":    doc_id,
            "keywords":  keywords[:10],  # truncate for readability
            "source":    src_display,
            "reference": ref_text,
            "generated": gen_text,
        }
        results.append(record)

        # Console preview
        logger.info(
            "[%d/%d] %s  (src=%d chars, ref=%d chars, gen=%d chars)",
            i + 1, n, doc_id,
            len(src_text), len(ref_text), len(gen_text),
        )
        if i < 3:
            print(f"\n{'='*80}")
            print(f"SAMPLE {i+1}  (doc_id={doc_id})")
            print(f"{'~'*80}")
            print(f"KEYWORDS: {', '.join(keywords[:10])}")
            print(f"{'~'*80}")
            print(f"REFERENCE:\n{ref_text[:500]}")
            print(f"{'~'*80}")
            print(f"GENERATED:\n{gen_text[:500]}")
            print(f"{'='*80}")

    elapsed = time.time() - t0
    logger.info(
        "Generation complete: %d samples in %.1f s (%.2f s/sample)",
        n, elapsed, elapsed / max(1, n),
    )

    # ── Save ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Results saved -> %s", output_path)

    # ── Summary stats ─────────────────────────────────────────────────
    ref_lens = [len(r["reference"]) for r in results]
    gen_lens = [len(r["generated"]) for r in results]
    logger.info(
        "Avg length — reference: %.0f chars, generated: %.0f chars",
        sum(ref_lens) / max(1, len(ref_lens)),
        sum(gen_lens) / max(1, len(gen_lens)),
    )


if __name__ == "__main__":
    main()
