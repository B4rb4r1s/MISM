"""
dual_encoder_summarizer.py — DualEncoderSummarizer: full model assembly.

Architecture
------------
  KeywordsEncoder     — encode keyword phrases (T5 backbone, frozen or trainable)
  DocumentEncoder     — encode document windows (T5 backbone, frozen or trainable)
  FusionLayer         — bidirectional cross-attention + gated fusion
  T5 Decoder          — standard T5 decoder (trainable or partially frozen)
  KeywordAttentionLayer (KAL) — post-decoder explicit keyword cross-attention

Forward pass (training, teacher forcing)
-----------------------------------------
1. kw_embs, kw_pooled = KeywordsEncoder(kw_input_ids, ...)
2. doc_pooled, full_seq, win_weights = DocumentEncoder(windows, ..., kw_pooled)
3. enc_hs, enc_mask, fusion_gates = FusionLayer(full_seq, ..., kw_embs)
4. dec_hidden = T5Decoder(decoder_input_ids, enc_hs, enc_mask)
5. logits, kw_attn_w, kal_gates = KAL(dec_hidden, kw_embs, ...)

Returns  DualEncoderOutput
---------
logits           [B, T, V]   vocabulary logits
kw_attn_weights  [B, T, K]   for coverage loss
fusion_gate_values [B, L]    for gate loss (fusion layer)
kal_gate_values  [B, T]      for gate loss (KAL)
doc_pooled       [B, D]      for alignment loss (if used)
kw_pooled        [B, D]      for alignment loss (if used)
decoder_hidden   [B, T, D]   last decoder layer hidden states (for L_bert)

Classmethod
-----------
DualEncoderSummarizer.from_pretrained(model_name, **kwargs)
    Load from a HuggingFace T5/T5-based model (e.g. rut5_base_sum_gazeta).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config

from src.models.keywords_encoder   import KeywordsEncoder
from src.models.document_encoder   import DocumentEncoder
from src.models.fusion_layer       import FusionLayer
from src.models.keyword_attention  import KeywordAttentionLayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class DualEncoderOutput:
    """Structured output of the DualEncoderSummarizer forward pass."""
    logits:              torch.Tensor                    # [B, T, V]
    kw_attn_weights:     torch.Tensor                    # [B, T, K]
    fusion_gate_values:  torch.Tensor                    # [B, L]
    kal_gate_values:     torch.Tensor                    # [B, T]
    doc_pooled:          torch.Tensor                    # [B, D]
    kw_pooled:           torch.Tensor                    # [B, D]
    decoder_hidden:      torch.Tensor                    # [B, T, D]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _shift_tokens_right(
    labels:                torch.Tensor,   # [B, T]
    pad_token_id:          int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    """Shift label tokens right (teacher forcing: feed gold token at t-1)."""
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0]  = decoder_start_token_id
    # Replace -100 (label padding) with pad_token_id so embeddings work
    shifted = shifted.masked_fill(shifted == -100, pad_token_id)
    return shifted


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DualEncoderSummarizer(nn.Module):
    """Dual-encoder summarization model with keyword-guided generation.

    Parameters
    ----------
    t5_model        : T5ForConditionalGeneration — pretrained T5 backbone.
    hidden_size     : int — T5 d_model (inferred from t5_model.config).
    kw_num_heads    : int — heads in KeywordsEncoder self-attention.
    kw_ffn_dim      : int — FFN inner dim in KeywordsEncoder.
    doc_num_heads   : int — heads in DocumentEncoder window attention.
    doc_ffn_dim     : int — FFN inner dim in DocumentEncoder.
    fusion_num_heads: int — heads in FusionLayer.
    fusion_ffn_dim  : int — FFN inner dim in FusionLayer.
    window_overlap  : int — overlap used by FusionLayer.merge_windows.
    max_src_len     : int — max merged doc tokens for decoder cross-attention.
    kal_num_heads   : int — heads in KeywordAttentionLayer.
    kal_ffn_dim     : int — FFN inner dim in KeywordAttentionLayer.
    dropout         : float — dropout probability.
    """

    def __init__(
        self,
        t5_model:         T5ForConditionalGeneration,
        hidden_size:      int = 768,
        kw_num_heads:     int = 12,
        kw_ffn_dim:       int = 3072,
        doc_num_heads:    int = 12,
        doc_ffn_dim:      int = 3072,
        fusion_num_heads: int = 12,
        fusion_ffn_dim:   int = 3072,
        window_overlap:   int = 128,
        max_src_len:      int = 4096,
        kal_num_heads:    int = 8,
        kal_ffn_dim:      int = 1536,
        dropout:          float = 0.1,
    ) -> None:
        super().__init__()

        cfg: T5Config = t5_model.config

        # ── Shared embedding + lm_head ────────────────────────────────
        self.shared     = t5_model.shared      # nn.Embedding [vocab, D]
        self.lm_head    = t5_model.lm_head     # nn.Linear [D → vocab]
        self.pad_token_id           = cfg.pad_token_id or 0
        self.decoder_start_token_id = cfg.decoder_start_token_id or 0
        self.vocab_size             = cfg.vocab_size

        # ── Document encoder backbone ─────────────────────────────────
        doc_enc_backbone = t5_model.encoder
        doc_enc_backbone.embed_tokens = self.shared

        self.keywords_encoder = KeywordsEncoder(
            t5_encoder=copy.deepcopy(doc_enc_backbone),   # independent copy
            hidden_size=hidden_size,
            num_heads=kw_num_heads,
            ffn_dim=kw_ffn_dim,
            dropout=dropout,
        )
        # Make kw encoder's embed_tokens point to shared (not the copy's copy)
        self.keywords_encoder.t5_encoder.embed_tokens = self.shared

        self.document_encoder = DocumentEncoder(
            t5_encoder=doc_enc_backbone,
            hidden_size=hidden_size,
            num_heads=doc_num_heads,
            ffn_dim=doc_ffn_dim,
            dropout=dropout,
        )

        # ── Fusion layer ──────────────────────────────────────────────
        self.fusion_layer = FusionLayer(
            hidden_size=hidden_size,
            num_heads=fusion_num_heads,
            ffn_dim=fusion_ffn_dim,
            window_overlap=window_overlap,
            max_src_len=max_src_len,
            dropout=dropout,
        )

        # ── T5 Decoder backbone ───────────────────────────────────────
        self.decoder = t5_model.decoder
        self.decoder.embed_tokens = self.shared

        # ── Keyword Attention Layer ───────────────────────────────────
        self.keyword_attention_layer = KeywordAttentionLayer(
            hidden_size=hidden_size,
            num_heads=kal_num_heads,
            ffn_dim=kal_ffn_dim,
            vocab_size=self.vocab_size,
            dropout=dropout,
        )
        self.keyword_attention_layer.set_lm_head(self.lm_head)

        # ── T5 scaling factor (must match what T5ForConditionalGeneration does) ─
        # When tie_word_embeddings=True the original T5 forward applies
        #   hidden = hidden * (d_model ** -0.5)  BEFORE the lm_head.
        # We extracted lm_head into KAL, so we replicate this scaling there.
        self._tie_word_embeddings: bool = cfg.tie_word_embeddings
        self._model_dim: int = hidden_size
        if self._tie_word_embeddings:
            scale = hidden_size ** -0.5
            self.keyword_attention_layer.set_lm_head_scale(scale)
            logger.info(
                "T5 tie_word_embeddings=True → KAL lm_head scale = %.6f",
                scale,
            )

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "DualEncoderSummarizer":
        """Load from a HuggingFace T5 checkpoint.

        Typical usage (GAZETA_2STAGE strategy):
            model = DualEncoderSummarizer.from_pretrained(
                "IlyaGusev/rut5_base_sum_gazeta",
                max_src_len=4096,
            )

        Parameters
        ----------
        pretrained_model_name_or_path : str
            HuggingFace model name or local path.
        **kwargs
            Extra keyword arguments forwarded to the DualEncoderSummarizer
            constructor (e.g. ``max_src_len``, ``dropout``).
        """
        logger.info("Loading backbone: %s", pretrained_model_name_or_path)
        t5_model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
        )
        # Allow caller to override hidden_size; fall back to model config.
        hidden_size = kwargs.pop("hidden_size", t5_model.config.d_model)
        return cls(t5_model=t5_model, hidden_size=hidden_size, **kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_windows:         torch.Tensor,         # [B, W, S]
        window_attention_mask: torch.Tensor,         # [B, W, S]
        kw_input_ids:          torch.Tensor,         # [B, K, L]
        kw_attention_mask:     torch.Tensor,         # [B, K, L]
        kw_scores:             torch.Tensor,         # [B, K]
        kw_mask:               torch.Tensor,         # [B, K]  bool
        labels:                Optional[torch.Tensor] = None,  # [B, T]
    ) -> DualEncoderOutput:
        """Run end-to-end forward pass (teacher forcing during training).

        Parameters
        ----------
        input_windows          : Tokenised document windows.
        window_attention_mask  : Attention mask for document windows.
        kw_input_ids           : Tokenised keyword phrases.
        kw_attention_mask      : Attention mask for keyword phrases.
        kw_scores              : Importance scores for keywords (uniform 1.0
                                 for keywords_original).
        kw_mask                : Boolean mask — True = real keyword.
        labels                 : Target token IDs (with -100 at padding).
                                 If provided, teacher-forced decoder is used.
                                 If None, you must provide decoder_input_ids
                                 externally (for generation).

        Returns
        -------
        DualEncoderOutput
        """
        # Sanitize kw_scores: NaN/Inf can appear when dataset extraction fails
        # (e.g. TF-IDF: 0/0 = NaN, or JSON float "inf").  Replace with 0 so
        # the score-weighted pooling and coverage loss degrade gracefully.
        kw_scores = torch.nan_to_num(kw_scores, nan=0.0, posinf=1.0, neginf=0.0)

        # ── 1. Encode keywords ────────────────────────────────────────
        kw_embs, kw_pooled = self.keywords_encoder(
            kw_input_ids, kw_attention_mask, kw_scores, kw_mask
        )
        # kw_embs:  [B, K, D]
        # kw_pooled:[B, D]
        self._nan_probe("1_kw_embs",   kw_embs)
        self._nan_probe("1_kw_pooled", kw_pooled)

        # ── 2. Encode document ────────────────────────────────────────
        doc_pooled, full_sequence, _ = self.document_encoder(
            input_windows, window_attention_mask, kw_pooled
        )
        # doc_pooled:    [B, D]
        # full_sequence: [B, W, S, D]
        self._nan_probe("2_doc_pooled",     doc_pooled)
        self._nan_probe("2_full_sequence",  full_sequence)

        # ── 3. Fuse doc and keywords ──────────────────────────────────
        encoder_hs, encoder_mask, fusion_gates = self.fusion_layer(
            full_sequence, window_attention_mask, kw_embs, kw_mask
        )
        # encoder_hs:    [B, L+K, D]
        # encoder_mask:  [B, L+K]
        # fusion_gates:  [B, L]
        self._nan_probe("3_encoder_hs",    encoder_hs)
        self._nan_probe("3_fusion_gates",  fusion_gates)

        # ── 4. T5 Decoder (teacher forcing) ───────────────────────────
        if labels is not None:
            decoder_input_ids = _shift_tokens_right(
                labels,
                pad_token_id=self.pad_token_id,
                decoder_start_token_id=self.decoder_start_token_id,
            )
        else:
            raise ValueError(
                "labels must be provided for teacher-forcing forward pass. "
                "For generation, use the generate() method."
            )

        # decoder_attention_mask: 1 for real tokens, 0 for padding
        decoder_attn_mask = (decoder_input_ids != self.pad_token_id).long()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attn_mask,
            encoder_hidden_states=encoder_hs,
            encoder_attention_mask=encoder_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        decoder_hidden = decoder_outputs.last_hidden_state  # [B, T, D]
        self._nan_probe("4_decoder_hidden", decoder_hidden)

        # ── 5. Keyword Attention Layer ────────────────────────────────
        enhanced_logits, kw_attn_weights, kal_gates = self.keyword_attention_layer(
            decoder_hidden, kw_embs, kw_mask, kw_scores
        )
        # enhanced_logits:  [B, T, V]
        # kw_attn_weights:  [B, T, K]
        # kal_gates:        [B, T]
        self._nan_probe("5_logits",    enhanced_logits)
        self._nan_probe("5_kal_gates", kal_gates)

        # Detach informational outputs that do NOT participate in any loss
        # component.  Without .detach(), DDP sees the upstream parameters as
        # "used" in the forward graph but never receiving gradients during
        # backward, which triggers:
        #   RuntimeError: Expected to have finished reduction in the prior
        #   iteration before starting a new one.
        #
        # • doc_pooled  — produced by DocumentEncoder's custom window-level
        #                  layers (self-attn, cross-attn, FFN), which are
        #                  trainable but have NO path to the loss.
        #                  (full_sequence comes from the frozen T5 backbone
        #                   *before* those layers.)
        # • kw_pooled   — its source layers already receive gradients
        #                  through kw_embs → FusionLayer → loss, but the
        #                  returned tensor itself is unused; detach is a
        #                  safety measure.
        # • decoder_hidden — the decoder is frozen in Stage 1 (requires_grad
        #                    is False), so DDP ignores it, but detaching
        #                    makes the contract explicit.
        return DualEncoderOutput(
            logits=enhanced_logits,
            kw_attn_weights=kw_attn_weights,
            fusion_gate_values=fusion_gates,
            kal_gate_values=kal_gates,
            doc_pooled=doc_pooled.detach(),
            kw_pooled=kw_pooled.detach(),
            decoder_hidden=decoder_hidden.detach(),
        )

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_windows:         torch.Tensor,   # [B, W, S]
        window_attention_mask: torch.Tensor,   # [B, W, S]
        kw_input_ids:          torch.Tensor,   # [B, K, L]
        kw_attention_mask:     torch.Tensor,   # [B, K, L]
        kw_scores:             torch.Tensor,   # [B, K]
        kw_mask:               torch.Tensor,   # [B, K]
        max_length:            int   = 256,
        min_length:            int   = 0,
        num_beams:             int   = 1,
        length_penalty:        float = 1.0,
        repetition_penalty:    float = 1.2,
        no_repeat_ngram_size:  int   = 4,
        eos_token_id:          int   = 1,
        bypass_kal:            bool  = False,
    ) -> torch.Tensor:
        """Autoregressive decoding with greedy or beam search.

        Parameters
        ----------
        max_length            : maximum number of generated tokens.
        min_length            : suppress EOS before this many tokens (default 0).
        num_beams             : beam width (1 = greedy, >1 = beam search).
        length_penalty        : beam search length normalisation exponent
                                (>1 favours longer sequences, default 1.0).
        repetition_penalty    : > 1.0 discourages repetition.
        no_repeat_ngram_size  : ban n-grams that already appeared (default 4).
        eos_token_id          : end-of-sequence token id (default 1 for T5).
        bypass_kal            : if True, skip KAL and project decoder hidden
                                states to vocabulary directly via lm_head
                                (diagnostic mode to verify encoder/decoder
                                quality without KAL interference).

        Returns
        -------
        generated_ids : torch.Tensor  [B, T]  (includes decoder_start_token)
        """
        kw_scores = torch.nan_to_num(kw_scores, nan=0.0, posinf=1.0, neginf=0.0)

        # ── 1-3.  Encode (keywords → document → fusion) ───────────────
        kw_embs, kw_pooled = self.keywords_encoder(
            kw_input_ids, kw_attention_mask, kw_scores, kw_mask,
        )
        _, full_sequence, _ = self.document_encoder(
            input_windows, window_attention_mask, kw_pooled,
        )
        encoder_hs, encoder_mask, _ = self.fusion_layer(
            full_sequence, window_attention_mask, kw_embs, kw_mask,
        )

        if num_beams > 1:
            return self._beam_search(
                encoder_hs=encoder_hs,
                encoder_mask=encoder_mask,
                kw_embs=kw_embs,
                kw_mask=kw_mask,
                kw_scores=kw_scores,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_token_id,
                bypass_kal=bypass_kal,
            )

        return self._greedy_search(
            encoder_hs=encoder_hs,
            encoder_mask=encoder_mask,
            kw_embs=kw_embs,
            kw_mask=kw_mask,
            kw_scores=kw_scores,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            bypass_kal=bypass_kal,
        )

    # ------------------------------------------------------------------
    # Greedy search (original logic + min_length)
    # ------------------------------------------------------------------

    def _greedy_search(
        self,
        encoder_hs:           torch.Tensor,
        encoder_mask:         torch.Tensor,
        kw_embs:              torch.Tensor,
        kw_mask:              torch.Tensor,
        kw_scores:            torch.Tensor,
        max_length:           int,
        min_length:           int,
        repetition_penalty:   float,
        no_repeat_ngram_size: int,
        eos_token_id:         int,
        bypass_kal:           bool,
    ) -> torch.Tensor:
        B      = encoder_hs.size(0)
        device = encoder_hs.device
        _bypass_scale = (self._model_dim ** -0.5) if self._tie_word_embeddings else 1.0

        generated = torch.full(
            (B, 1), self.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        finished        = torch.zeros(B, dtype=torch.bool, device=device)
        past_key_values = None

        for step in range(max_length):
            dec_input = generated if step == 0 else generated[:, -1:]

            decoder_out = self.decoder(
                input_ids=dec_input,
                encoder_hidden_states=encoder_hs,
                encoder_attention_mask=encoder_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_out.past_key_values
            dec_hidden = decoder_out.last_hidden_state       # [B, 1, D]

            if bypass_kal:
                next_logits = self.lm_head(
                    dec_hidden * _bypass_scale
                )[:, -1, :]                                  # [B, V]
            else:
                logits, _, _ = self.keyword_attention_layer(
                    dec_hidden, kw_embs, kw_mask, kw_scores,
                )
                next_logits = logits[:, -1, :]               # [B, V]

            # ── min_length: suppress EOS before min_length tokens ──
            if step < min_length:
                next_logits[:, eos_token_id] = float("-inf")

            # ── Repetition penalty ─────────────────────────────────
            if repetition_penalty != 1.0:
                for b in range(B):
                    prev = generated[b].unique()
                    pos = next_logits[b, prev] > 0
                    next_logits[b, prev[pos]]  /= repetition_penalty
                    next_logits[b, prev[~pos]] *= repetition_penalty

            # ── No-repeat n-gram blocking ──────────────────────────
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                for b in range(B):
                    tokens = generated[b].tolist()
                    prefix = tuple(tokens[-(n - 1):])
                    for i in range(len(tokens) - n + 1):
                        if tuple(tokens[i : i + n - 1]) == prefix:
                            next_logits[b, tokens[i + n - 1]] = float("-inf")

            # ── Greedy select ──────────────────────────────────────
            next_token = next_logits.argmax(dim=-1, keepdim=True)   # [B, 1]
            next_token.masked_fill_(finished.unsqueeze(1), self.pad_token_id)
            generated  = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        return generated

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------

    def _beam_search(
        self,
        encoder_hs:           torch.Tensor,   # [B, L+K, D]
        encoder_mask:         torch.Tensor,   # [B, L+K]
        kw_embs:              torch.Tensor,   # [B, K, D]
        kw_mask:              torch.Tensor,   # [B, K]
        kw_scores:            torch.Tensor,   # [B, K]
        max_length:           int,
        min_length:           int,
        num_beams:            int,
        length_penalty:       float,
        repetition_penalty:   float,
        no_repeat_ngram_size: int,
        eos_token_id:         int,
        bypass_kal:           bool,
    ) -> torch.Tensor:
        """Beam search decoding with length normalisation.

        For each batch element, maintains ``num_beams`` hypotheses in parallel.
        Finished hypotheses are scored as  ``log_prob_sum / length^length_penalty``
        and the best one is returned.
        """
        B      = encoder_hs.size(0)
        device = encoder_hs.device
        NB     = num_beams
        _bypass_scale = (self._model_dim ** -0.5) if self._tie_word_embeddings else 1.0

        # ── Expand encoder outputs for beams ──────────────────────────
        # [B, ...] → [B*NB, ...]
        encoder_hs   = encoder_hs.repeat_interleave(NB, dim=0)    # [B*NB, L+K, D]
        encoder_mask = encoder_mask.repeat_interleave(NB, dim=0)  # [B*NB, L+K]
        kw_embs      = kw_embs.repeat_interleave(NB, dim=0)      # [B*NB, K, D]
        kw_mask      = kw_mask.repeat_interleave(NB, dim=0)      # [B*NB, K]
        kw_scores_ex = kw_scores.repeat_interleave(NB, dim=0)    # [B*NB, K]

        # ── Per-beam state ────────────────────────────────────────────
        # generated: [B*NB, 1] — starts with decoder_start_token
        generated = torch.full(
            (B * NB, 1), self.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        beam_scores  = torch.zeros(B * NB, device=device)    # log-prob sums
        # Only first beam per batch should be active initially;
        # fill others with -inf so they don't compete at step 0.
        beam_scores[1::NB] = float("-inf")
        if NB > 2:
            for i in range(2, NB):
                beam_scores[i::NB] = float("-inf")

        finished_beams: list[list[tuple[torch.Tensor, float]]] = [
            [] for _ in range(B)
        ]  # per-batch list of (token_ids, normalised_score)

        past_key_values = None

        for step in range(max_length):
            dec_input = generated if step == 0 else generated[:, -1:]

            decoder_out = self.decoder(
                input_ids=dec_input,
                encoder_hidden_states=encoder_hs,
                encoder_attention_mask=encoder_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_out.past_key_values
            dec_hidden = decoder_out.last_hidden_state       # [B*NB, 1, D]

            if bypass_kal:
                next_logits = self.lm_head(
                    dec_hidden * _bypass_scale
                )[:, -1, :]                                  # [B*NB, V]
            else:
                logits, _, _ = self.keyword_attention_layer(
                    dec_hidden, kw_embs, kw_mask, kw_scores_ex,
                )
                next_logits = logits[:, -1, :]               # [B*NB, V]

            V = next_logits.size(-1)

            # ── min_length: suppress EOS ──────────────────────────
            if step < min_length:
                next_logits[:, eos_token_id] = float("-inf")

            # ── Repetition penalty (per beam) ─────────────────────
            if repetition_penalty != 1.0:
                for idx in range(B * NB):
                    prev = generated[idx].unique()
                    pos = next_logits[idx, prev] > 0
                    next_logits[idx, prev[pos]]  /= repetition_penalty
                    next_logits[idx, prev[~pos]] *= repetition_penalty

            # ── No-repeat n-gram blocking (per beam) ──────────────
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                for idx in range(B * NB):
                    tokens = generated[idx].tolist()
                    prefix = tuple(tokens[-(n - 1):])
                    for i in range(len(tokens) - n + 1):
                        if tuple(tokens[i : i + n - 1]) == prefix:
                            next_logits[idx, tokens[i + n - 1]] = float("-inf")

            # ── Log-softmax scores ────────────────────────────────
            log_probs = torch.nn.functional.log_softmax(next_logits, dim=-1)  # [B*NB, V]

            # Combine with accumulated beam scores
            # [B*NB, V] = beam_scores[:, None] + log_probs
            candidate_scores = beam_scores.unsqueeze(1) + log_probs  # [B*NB, V]

            # ── Reshape to [B, NB*V] for top-k selection ──────────
            candidate_scores = candidate_scores.view(B, NB * V)

            # Select top 2*NB candidates per batch element
            topk_scores, topk_indices = candidate_scores.topk(
                2 * NB, dim=-1, largest=True, sorted=True,
            )  # [B, 2*NB]

            # Decode beam and token indices
            topk_beam_idx  = topk_indices // V  # which beam [0..NB-1]
            topk_token_idx = topk_indices % V   # which token [0..V-1]

            # ── Build next beams ──────────────────────────────────
            new_generated   = []
            new_beam_scores = []
            new_past_beams  = []   # track which beam in B*NB each new beam came from

            all_done = True

            for b in range(B):
                beams_for_b = []
                for rank in range(2 * NB):
                    if len(beams_for_b) >= NB:
                        break

                    local_beam = topk_beam_idx[b, rank].item()
                    token      = topk_token_idx[b, rank].item()
                    score      = topk_scores[b, rank].item()
                    global_idx = b * NB + local_beam

                    if token == eos_token_id:
                        # Finished beam — compute normalised score
                        seq_len = generated.size(1)  # tokens so far (excl. this EOS)
                        norm_score = score / ((seq_len + 1) ** length_penalty)
                        finished_beams[b].append(
                            (generated[global_idx].clone(), norm_score)
                        )
                        continue

                    beams_for_b.append((global_idx, token, score))

                # Pad with dummy beams if needed (shouldn't happen often)
                while len(beams_for_b) < NB:
                    # Reuse last valid beam with very low score
                    g_idx = b * NB
                    beams_for_b.append((g_idx, self.pad_token_id, float("-inf")))

                for g_idx, token, score in beams_for_b:
                    new_past_beams.append(g_idx)
                    new_beam_scores.append(score)
                    new_row = torch.cat([
                        generated[g_idx],
                        torch.tensor([token], device=device, dtype=torch.long),
                    ])
                    new_generated.append(new_row)

                # Check if enough finished beams
                if len(finished_beams[b]) < NB:
                    all_done = False

            # ── Update state ──────────────────────────────────────
            generated   = torch.stack(new_generated, dim=0)             # [B*NB, T+1]
            beam_scores = torch.tensor(new_beam_scores, device=device)  # [B*NB]

            # Reorder KV cache to match new beam order
            reorder_idx = torch.tensor(new_past_beams, device=device, dtype=torch.long)
            past_key_values = self._reorder_cache(past_key_values, reorder_idx)

            if all_done:
                break

        # ── Select best hypothesis per batch element ──────────────────
        results = []
        for b in range(B):
            # Add still-active beams as candidates
            for beam_i in range(NB):
                g_idx = b * NB + beam_i
                seq_len = generated.size(1)
                norm_score = beam_scores[g_idx].item() / (seq_len ** length_penalty)
                finished_beams[b].append(
                    (generated[g_idx].clone(), norm_score)
                )

            # Select best
            best_seq, best_score = max(finished_beams[b], key=lambda x: x[1])
            results.append(best_seq)

        # ── Pad to equal length ───────────────────────────────────────
        max_len = max(r.size(0) for r in results)
        padded  = torch.full(
            (B, max_len), self.pad_token_id,
            dtype=torch.long, device=device,
        )
        for b, seq in enumerate(results):
            padded[b, :seq.size(0)] = seq

        return padded

    @staticmethod
    def _reorder_cache(
        past_key_values,
        beam_idx: torch.Tensor,
    ):
        """Reorder KV cache to match reordered beams.

        Supports both ``DynamicCache`` (transformers ≥ 4.36) and
        the legacy tuple-of-tuples format.
        """
        # DynamicCache — has an in-place reorder method
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        # Legacy tuple format
        reordered = []
        for layer_past in past_key_values:
            reordered.append(
                tuple(state.index_select(0, beam_idx) for state in layer_past)
            )
        return tuple(reordered)

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    @staticmethod
    def _nan_probe(name: str, tensor: torch.Tensor) -> None:
        """Log once per forward pass if tensor contains NaN or Inf."""
        if not tensor.isfinite().all():
            logger.error(
                "NaN probe → %s  shape=%s  nan=%d  inf=%d  "
                "min=%.4g  max=%.4g",
                name,
                list(tensor.shape),
                tensor.isnan().sum().item(),
                tensor.isinf().sum().item(),
                tensor[tensor.isfinite()].min().item()
                if tensor.isfinite().any()
                else float("nan"),
                tensor[tensor.isfinite()].max().item()
                if tensor.isfinite().any()
                else float("nan"),
            )

    # ------------------------------------------------------------------
    # Training strategy helpers
    # ------------------------------------------------------------------

    def freeze_encoders(self) -> None:
        """Freeze both T5 encoder backbones (document + keyword).

        The shared embedding (``embed_tokens``) is explicitly **kept
        trainable** because it is shared with the decoder and lm_head.
        Freezing it here would also freeze the vocabulary projection,
        preventing the model from learning to generate.
        """
        for name, param in self.document_encoder.t5_encoder.named_parameters():
            if "embed_tokens" not in name:
                param.requires_grad_(False)
        for name, param in self.keywords_encoder.t5_encoder.named_parameters():
            if "embed_tokens" not in name:
                param.requires_grad_(False)
        # Explicit safety net: ensure shared stays trainable
        self.shared.weight.requires_grad_(True)
        logger.info("Both T5 encoder backbones frozen (shared embedding kept trainable).")

    def unfreeze_encoders(self) -> None:
        """Unfreeze both T5 encoder backbones."""
        for param in self.document_encoder.t5_encoder.parameters():
            param.requires_grad_(True)
        for param in self.keywords_encoder.t5_encoder.parameters():
            param.requires_grad_(True)
        logger.info("Both T5 encoder backbones unfrozen.")

    def freeze_decoder(self) -> None:
        """Freeze the T5 decoder backbone.

        The shared embedding (``embed_tokens``) is explicitly **kept
        trainable** — see :meth:`freeze_encoders` for rationale.
        """
        for name, param in self.decoder.named_parameters():
            if "embed_tokens" not in name:
                param.requires_grad_(False)
        # Explicit safety net: ensure shared stays trainable
        self.shared.weight.requires_grad_(True)
        logger.info("T5 decoder backbone frozen (shared embedding kept trainable).")

    def unfreeze_decoder(self) -> None:
        """Unfreeze the T5 decoder backbone."""
        for param in self.decoder.parameters():
            param.requires_grad_(True)
        logger.info("T5 decoder backbone unfrozen.")

    def get_trainable_param_count(self) -> dict:
        """Return counts of trainable vs total parameters per component."""
        components = {
            "keywords_encoder":        self.keywords_encoder,
            "document_encoder":        self.document_encoder,
            "fusion_layer":            self.fusion_layer,
            "decoder":                 self.decoder,
            "keyword_attention_layer": self.keyword_attention_layer,
        }
        result = {}
        total_trainable = 0
        total_all = 0
        for name, module in components.items():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in module.parameters())
            result[name] = {"trainable": trainable, "total": all_params}
            total_trainable += trainable
            total_all += all_params
        result["TOTAL"] = {"trainable": total_trainable, "total": total_all}
        return result
