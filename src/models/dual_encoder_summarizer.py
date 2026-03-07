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
        repetition_penalty:    float = 1.2,
        no_repeat_ngram_size:  int   = 3,
        eos_token_id:          int   = 1,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding with KV cache.

        Parameters
        ----------
        max_length            : maximum number of generated tokens.
        repetition_penalty    : > 1.0 discourages repetition.
        no_repeat_ngram_size  : ban n-grams that already appeared.
        eos_token_id          : end-of-sequence token id (default 1 for T5).

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

        B      = encoder_hs.size(0)
        device = encoder_hs.device

        # ── 4.  Autoregressive decode ──────────────────────────────────
        generated = torch.full(
            (B, 1), self.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        finished         = torch.zeros(B, dtype=torch.bool, device=device)
        past_key_values  = None

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

            logits, _, _ = self.keyword_attention_layer(
                dec_hidden, kw_embs, kw_mask, kw_scores,
            )
            next_logits = logits[:, -1, :]                   # [B, V]

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
        """Freeze both T5 encoder backbones (document + keyword)."""
        for param in self.document_encoder.t5_encoder.parameters():
            param.requires_grad_(False)
        for param in self.keywords_encoder.t5_encoder.parameters():
            param.requires_grad_(False)
        logger.info("Both T5 encoder backbones frozen.")

    def unfreeze_encoders(self) -> None:
        """Unfreeze both T5 encoder backbones."""
        for param in self.document_encoder.t5_encoder.parameters():
            param.requires_grad_(True)
        for param in self.keywords_encoder.t5_encoder.parameters():
            param.requires_grad_(True)
        logger.info("Both T5 encoder backbones unfrozen.")

    def freeze_decoder(self) -> None:
        """Freeze the T5 decoder backbone."""
        for param in self.decoder.parameters():
            param.requires_grad_(False)
        logger.info("T5 decoder backbone frozen.")

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
