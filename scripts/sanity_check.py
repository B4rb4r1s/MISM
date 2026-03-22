#!/usr/bin/env python
"""
sanity_check.py — Quick sanity check for the MISM model at initialisation.

Loads a fresh pretrained FRED-T5-1.7B (no checkpoint), runs one forward pass
and one greedy generation, and prints diagnostics.  Includes a vanilla T5
baseline comparison to isolate whether the issue is in MISM assembly.

Usage
-----
    python scripts/sanity_check.py --config configs/gazeta_2stage.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.collator import DataCollatorForSummarization
from src.data.dataset import SummarizationDataset
from src.models.dual_encoder_summarizer import DualEncoderSummarizer
from src.training.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sanity_check")


def tensor_stats(t: torch.Tensor, name: str) -> str:
    """One-line summary of tensor statistics."""
    return (
        f"{name}: shape={list(t.shape)}  "
        f"min={t.min().item():.4f}  max={t.max().item():.4f}  "
        f"mean={t.float().mean().item():.4f}  std={t.float().std().item():.4f}  "
        f"nan={t.isnan().sum().item()}  inf={t.isinf().sum().item()}"
    )


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                    label_smoothing: float = 0.1) -> float:
    """Cross-entropy loss matching GenerativeLoss."""
    B, T, V = logits.shape
    ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
    return ce(logits.reshape(B * T, V), labels.reshape(B * T)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description="MISM initialisation sanity check")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Index into val set to use (default: 0)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Device ───────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── Tokeniser ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    # ── Dataset (1 sample) ───────────────────────────────────────────────
    dataset = SummarizationDataset.from_json(cfg.val_path)
    logger.info("Val set: %d samples", len(dataset))
    sample = dataset[args.sample_idx]

    collator = DataCollatorForSummarization(
        tokenizer=tokenizer,
        max_kw=cfg.max_kw,
        kw_max_len=cfg.kw_max_len,
        window_size=cfg.window_size,
        window_overlap=cfg.window_overlap,
        max_windows=cfg.max_windows,
        max_summary_tokens=cfg.max_summary_tokens,
    )
    batch = collator([sample])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # =====================================================================
    # TEST 0: VANILLA FRED-T5 BASELINE
    # =====================================================================
    print("\n" + "=" * 80)
    print("0. VANILLA FRED-T5 BASELINE (no MISM wrapping)")
    print("=" * 80)

    logger.info("Loading vanilla FRED-T5 for baseline...")
    vanilla_t5 = T5ForConditionalGeneration.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32,
    )
    vanilla_t5.to(device)
    vanilla_t5.eval()

    # Tokenize the same document as a flat sequence for vanilla T5
    src_text = sample["text_clean"]
    tgt_text = sample["summary"]

    src_enc = tokenizer(
        src_text, max_length=512, truncation=True,
        padding="max_length", return_tensors="pt",
    ).to(device)
    tgt_enc = tokenizer(
        tgt_text, max_length=cfg.max_summary_tokens, truncation=True,
        padding="max_length", return_tensors="pt",
    ).to(device)
    # Build labels: replace pad_token_id with -100
    vanilla_labels = tgt_enc.input_ids.clone()
    vanilla_labels[vanilla_labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        vanilla_out = vanilla_t5(
            input_ids=src_enc.input_ids,
            attention_mask=src_enc.attention_mask,
            labels=vanilla_labels,
        )
    vanilla_ce = vanilla_out.loss.item()
    vanilla_logits = vanilla_out.logits

    print(f"\n  Vanilla CE loss = {vanilla_ce:.4f}")
    print(f"  Vanilla perplexity = {torch.exp(torch.tensor(vanilla_ce)).item():.1f}")
    print(f"  {tensor_stats(vanilla_logits, 'vanilla_logits')}")

    # Vanilla generation
    with torch.no_grad():
        vanilla_gen = vanilla_t5.generate(
            input_ids=src_enc.input_ids,
            attention_mask=src_enc.attention_mask,
            max_length=256,
            num_beams=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )
    vanilla_text = tokenizer.decode(vanilla_gen[0], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    print(f"\n  Vanilla generated ({len(vanilla_gen[0])} tokens):")
    print(f"  {vanilla_text[:500]}")

    # Free vanilla model memory
    del vanilla_t5
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =====================================================================
    # TEST 1: MISM FORWARD PASS
    # =====================================================================
    print("\n" + "=" * 80)
    print("1. MISM FORWARD PASS (teacher forcing)")
    print("=" * 80)

    logger.info("Loading fresh MISM model: %s", cfg.model_name)
    model = DualEncoderSummarizer.from_pretrained(
        cfg.model_name,
        window_overlap=cfg.window_overlap,
        max_src_len=cfg.max_src_len,
        dropout=cfg.dropout,
    )
    model.to(device)
    model.eval()
    logger.info("Model loaded (fresh pretrained, no checkpoint)")

    with torch.no_grad():
        output = model(
            input_windows=batch["input_windows"],
            window_attention_mask=batch["window_attention_mask"],
            kw_input_ids=batch["kw_input_ids"],
            kw_attention_mask=batch["kw_attention_mask"],
            kw_scores=batch["kw_scores"],
            kw_mask=batch["kw_mask"],
            labels=batch["labels"],
        )

    logits = output.logits
    labels = batch["labels"]
    mism_ce = compute_ce_loss(logits, labels)

    print(f"\n  MISM CE loss = {mism_ce:.4f}")
    print(f"  MISM perplexity = {torch.exp(torch.tensor(mism_ce)).item():.1f}")
    print()
    print(f"  {tensor_stats(logits, 'mism_logits')}")
    if output.decoder_hidden is not None:
        print(f"  {tensor_stats(output.decoder_hidden, 'decoder_hidden')}")
    if output.fusion_gate_values is not None:
        print(f"  {tensor_stats(output.fusion_gate_values, 'fusion_gates')}")
    if output.kal_gate_values is not None:
        print(f"  {tensor_stats(output.kal_gate_values, 'kal_gates')}")

    # =====================================================================
    # TEST 2: MISM — isolate encoder output
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. MISM ENCODER OUTPUT DIAGNOSTICS")
    print("=" * 80)

    with torch.no_grad():
        # Step 1: Keywords encoder
        kw_embs, kw_pooled = model.keywords_encoder(
            batch["kw_input_ids"], batch["kw_attention_mask"],
            batch["kw_scores"], batch["kw_mask"],
        )
        print(f"\n  {tensor_stats(kw_embs, 'kw_embs')}")
        print(f"  {tensor_stats(kw_pooled, 'kw_pooled')}")

        # Step 2: Document encoder
        doc_pooled, full_seq, win_weights = model.document_encoder(
            batch["input_windows"], batch["window_attention_mask"], kw_pooled,
        )
        print(f"  {tensor_stats(full_seq, 'full_sequence')}")
        print(f"  {tensor_stats(doc_pooled, 'doc_pooled')}")

        # Step 3: Fusion layer
        enc_hs, enc_mask, fusion_gates = model.fusion_layer(
            full_seq, batch["window_attention_mask"], kw_embs, batch["kw_mask"],
        )
        print(f"  {tensor_stats(enc_hs, 'encoder_hidden_states')}")
        print(f"  enc_mask: shape={list(enc_mask.shape)}  "
              f"sum={enc_mask.sum().item()}  (real tokens)")

        # Compare: doc tokens from full_seq vs from enc_hs (should be identical at init)
        merged_doc_len = enc_hs.shape[1] - kw_embs.shape[1]
        doc_part = enc_hs[:, :merged_doc_len]
        # Reconstruct what merge_windows would give
        from src.models.fusion_layer import FusionLayer
        fl = model.fusion_layer
        merged_seq, merged_mask = fl._merge_windows(
            full_seq, batch["window_attention_mask"])
        diff = (doc_part - merged_seq).abs().max().item()
        print(f"\n  Doc tokens diff (enc_hs vs raw merged): {diff:.6f}")
        print(f"  (should be ~0 if FusionLayer is identity at init)")

        # Compare kw part
        kw_part = enc_hs[:, merged_doc_len:]
        kw_diff = (kw_part - kw_embs).abs().max().item()
        print(f"  KW tokens diff (enc_hs vs kw_embs):    {kw_diff:.6f}")

    # =====================================================================
    # TEST 3: BYPASS EVERYTHING — use vanilla T5 decoder on MISM encoder output
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. MANUAL DECODER TEST (MISM encoder → T5 decoder → lm_head directly)")
    print("=" * 80)

    with torch.no_grad():
        # Use only the doc portion (no keywords) — should match vanilla T5
        from src.models.dual_encoder_summarizer import _shift_tokens_right
        dec_input_ids = _shift_tokens_right(
            batch["labels"],
            pad_token_id=model.pad_token_id,
            decoder_start_token_id=model.decoder_start_token_id,
        )
        dec_attn_mask = (dec_input_ids != model.pad_token_id).long()

        # Test A: Full encoder_hs (doc + kw)
        dec_out_full = model.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=enc_mask,
            return_dict=True,
        )
        dec_hidden_full = dec_out_full.last_hidden_state
        logits_full = model.lm_head(dec_hidden_full)
        ce_full = compute_ce_loss(logits_full, batch["labels"])
        print(f"\n  CE with full enc_hs (doc+kw):  {ce_full:.4f}")
        print(f"  {tensor_stats(dec_hidden_full, 'dec_hidden_full')}")
        print(f"  {tensor_stats(logits_full, 'logits_full')}")

        # Test B: Only doc portion (no keywords) — closest to vanilla T5
        doc_only_hs = enc_hs[:, :merged_doc_len]
        doc_only_mask = enc_mask[:, :merged_doc_len]
        dec_out_doc = model.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=doc_only_hs,
            encoder_attention_mask=doc_only_mask,
            return_dict=True,
        )
        dec_hidden_doc = dec_out_doc.last_hidden_state
        logits_doc = model.lm_head(dec_hidden_doc)
        ce_doc = compute_ce_loss(logits_doc, batch["labels"])
        print(f"\n  CE with doc-only enc_hs:       {ce_doc:.4f}")
        print(f"  {tensor_stats(dec_hidden_doc, 'dec_hidden_doc')}")
        print(f"  {tensor_stats(logits_doc, 'logits_doc')}")

        # Test C: Single window (window 0 only) — most similar to vanilla T5
        win0_hs = full_seq[:, 0]  # [B, S, D]
        win0_mask = batch["window_attention_mask"][:, 0].long()  # [B, S]
        dec_out_w0 = model.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=win0_hs,
            encoder_attention_mask=win0_mask,
            return_dict=True,
        )
        dec_hidden_w0 = dec_out_w0.last_hidden_state
        logits_w0 = model.lm_head(dec_hidden_w0)
        ce_w0 = compute_ce_loss(logits_w0, batch["labels"])
        print(f"\n  CE with window-0 only:         {ce_w0:.4f}")
        print(f"  {tensor_stats(logits_w0, 'logits_w0')}")

    # =====================================================================
    # TEST 4: GENERATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("4. MISM GENERATION (greedy)")
    print("=" * 80)

    with torch.no_grad():
        gen_ids = model.generate(
            input_windows=batch["input_windows"],
            window_attention_mask=batch["window_attention_mask"],
            kw_input_ids=batch["kw_input_ids"],
            kw_attention_mask=batch["kw_attention_mask"],
            kw_scores=batch["kw_scores"],
            kw_mask=batch["kw_mask"],
            max_length=256,
            num_beams=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            bypass_kal=False,
        )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
    print(f"\n  MISM generated ({len(gen_ids[0])} tokens):")
    print(f"  {gen_text[:500]}")

    # =====================================================================
    # REFERENCE & SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("5. REFERENCE")
    print("=" * 80)
    print(f"\n  Keywords: {', '.join(sample['keywords'][:10])}")
    print(f"  Reference: {tgt_text[:300]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Vanilla T5 CE:         {vanilla_ce:.4f}  (baseline)")
    print(f"  MISM CE (full):        {mism_ce:.4f}")
    print(f"  MISM CE (doc+kw, no KAL): {ce_full:.4f}")
    print(f"  MISM CE (doc only):    {ce_doc:.4f}")
    print(f"  MISM CE (window 0):    {ce_w0:.4f}")
    print(f"  Delta (MISM - vanilla): {mism_ce - vanilla_ce:.4f}")
    print()
    ok = mism_ce < 11.0
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        if ce_w0 < 11.0 and ce_doc > 11.0:
            print("  → Problem is in window merging or multi-window encoding")
        elif ce_doc < 11.0 and ce_full > 11.0:
            print("  → Problem is in keyword concatenation")
        elif ce_full < 11.0 and mism_ce > 11.0:
            print("  → Problem is in KAL")
        elif ce_w0 > 11.0:
            print("  → Problem is fundamental: even single window gives bad CE")
            print("    Compare with vanilla T5 to isolate")
    print("=" * 80)


if __name__ == "__main__":
    main()
