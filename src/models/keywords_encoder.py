"""
keywords_encoder.py — KeywordsEncoder module.

Encodes up to K keyword phrases into contextualised per-keyword representations
and a single weighted-pooled keyword summary vector.

Processing pipeline
-------------------
1. T5 backbone  : encode each keyword independently  [B*K, L, D]
2. Mean pooling : per-keyword representation         [B, K, D]
3. Self-attention: model inter-keyword relations     [B, K, D]
4. FFN          : position-wise transformation       [B, K, D]
5. Weighted pool: score-weighted summary vector      [B, D]

Input shapes  (B=batch, K=max_kw, L=kw_max_len, D=hidden_size)
-------------
kw_input_ids      [B, K, L]   long
kw_attention_mask [B, K, L]   long
kw_scores         [B, K]      float  (0.0 for padding slots)
kw_mask           [B, K]      bool   (True = real keyword)

Output
------
kw_embeddings  [B, K, D]   per-keyword contextualised representations
kw_pooled      [B, D]      global keyword summary (weighted pool)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


class KeywordsEncoder(nn.Module):
    """Encode a variable-length set of keyword phrases.

    Parameters
    ----------
    t5_encoder    : T5Stack (encoder-mode) — the backbone T5 encoder.
    hidden_size   : int — d_model of the T5 backbone (default 768).
    num_heads     : int — heads in keyword self-attention (default 12).
    ffn_dim       : int — inner dim of the FFN (default 3072).
    dropout       : float — dropout probability (default 0.1).
    """

    def __init__(
        self,
        t5_encoder: nn.Module,
        hidden_size: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t5_encoder = t5_encoder
        self.hidden_size = hidden_size

        # Self-attention between keyword representations
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        # Position-wise FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        kw_input_ids:      torch.Tensor,   # [B, K, L]
        kw_attention_mask: torch.Tensor,   # [B, K, L]
        kw_scores:         torch.Tensor,   # [B, K]
        kw_mask:           torch.Tensor,   # [B, K]  bool
    ):
        """Run the keywords encoder forward pass.

        Returns
        -------
        kw_embeddings : torch.Tensor  [B, K, D]
        kw_pooled     : torch.Tensor  [B, D]
        """
        B, K, L = kw_input_ids.shape
        D = self.hidden_size

        # ── 1. T5 backbone — encode all keywords in one batched call ──
        flat_ids  = kw_input_ids.view(B * K, L)          # [B*K, L]
        flat_mask = kw_attention_mask.view(B * K, L)      # [B*K, L]

        encoder_out: BaseModelOutput = self.t5_encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        hidden = encoder_out.last_hidden_state             # [B*K, L, D]
        # Guard: padding keyword slots (attention_mask=all 0) produce NaN inside
        # T5 self-attention.  Replace with 0 — downstream kw_mask excludes them.
        hidden = torch.nan_to_num(hidden, nan=0.0)

        # ── 2. Mean pooling per keyword (mask-aware) ──────────────────
        mask_exp  = flat_mask.unsqueeze(-1).float()        # [B*K, L, 1]
        sum_h     = (hidden * mask_exp).sum(dim=1)         # [B*K, D]
        count     = mask_exp.sum(dim=1).clamp(min=1.0)     # [B*K, 1]
        kw_embs   = (sum_h / count).view(B, K, D)          # [B, K, D]

        # ── 3. Self-attention between keywords ────────────────────────
        #   key_padding_mask: True = position to ignore (padding slots)
        pad_mask = ~kw_mask                                 # [B, K]  True=pad

        attended, _ = self.self_attn(
            kw_embs, kw_embs, kw_embs,
            key_padding_mask=pad_mask,
        )
        # Guard: when *all* keywords are padding the attention softmax produces
        # NaN (softmax of all -inf).  Replace NaN with 0 so the residual path
        # simply passes through the original kw_embs unchanged.
        attended = torch.nan_to_num(attended, nan=0.0)
        kw_embs = self.norm1(kw_embs + self.dropout(attended))   # [B, K, D]

        # ── 4. FFN ────────────────────────────────────────────────────
        kw_embs = self.norm2(kw_embs + self.dropout(self.ffn(kw_embs)))  # [B, K, D]

        # ── 5. Weighted pooling (by kw_scores, masked) ────────────────
        #   Set padding slots to -inf so softmax → 0.0
        scores_masked = kw_scores.masked_fill(~kw_mask, float("-inf"))  # [B, K]
        weights = torch.softmax(scores_masked, dim=-1)                  # [B, K]
        # Guard against all-masked rows (produces NaN) → replace with 0
        weights = torch.nan_to_num(weights, nan=0.0)
        kw_pooled = (weights.unsqueeze(-1) * kw_embs).sum(dim=1)        # [B, D]

        return kw_embs, kw_pooled
