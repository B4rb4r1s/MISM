"""
document_encoder.py — DocumentEncoder module.

Encodes a long document (split into overlapping windows) with keyword-guided
window aggregation.

Processing pipeline
-------------------
1. T5 backbone  : encode each window independently      [B*W, S, D]
2. Mean pooling : per-window representation             [B, W, D]
3. Self-attention: contextualise windows w.r.t. each other   [B, W, D]
4. Cross-attention (KW → Windows):
      Q = kw_pooled [B, 1, D]
      K = V = attended_windows [B, W, D]
      → window_weights [B, W]                          (attention distribution)
5. Weighted sum → doc_pooled                            [B, D]

The full per-token sequence [B, W, S, D] is also returned for the FusionLayer.

Input shapes  (B=batch, W=num_windows, S=window_size, D=hidden_size)
-------------
input_windows          [B, W, S]   long
window_attention_mask  [B, W, S]   long
kw_pooled              [B, D]      float  (from KeywordsEncoder)

Output
------
doc_pooled      [B, D]         keyword-guided global doc representation
full_sequence   [B, W, S, D]   per-token hidden states (all windows)
window_weights  [B, W]         normalised window relevance weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


class DocumentEncoder(nn.Module):
    """Encode a long document via sliding windows with keyword-guided aggregation.

    Parameters
    ----------
    t5_encoder    : T5Stack (encoder-mode) — the backbone T5 encoder.
    hidden_size   : int — d_model of the T5 backbone (default 768).
    num_heads     : int — heads in window self- and cross-attention (default 12).
    ffn_dim       : int — inner dim of the window FFN (default 3072).
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

        # Window-level self-attention (contextualise windows w.r.t. each other)
        self.window_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.window_norm1 = nn.LayerNorm(hidden_size)

        # KW-guided cross-attention: kw_pooled queries the window sequence
        # to produce a relevance weight per window
        self.window_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN on aggregated window representations
        self.window_ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.window_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_windows:          torch.Tensor,   # [B, W, S]
        window_attention_mask:  torch.Tensor,   # [B, W, S]
        kw_pooled:              torch.Tensor,   # [B, D]
    ):
        """Run the document encoder forward pass.

        Returns
        -------
        doc_pooled     : torch.Tensor  [B, D]
        full_sequence  : torch.Tensor  [B, W, S, D]
        window_weights : torch.Tensor  [B, W]
        """
        B, W, S = input_windows.shape
        D = self.hidden_size

        # ── 1. T5 backbone — encode all windows in one batched call ───
        flat_ids  = input_windows.view(B * W, S)           # [B*W, S]
        flat_mask = window_attention_mask.view(B * W, S)   # [B*W, S]

        encoder_out: BaseModelOutput = self.t5_encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        hidden = encoder_out.last_hidden_state             # [B*W, S, D]
        full_sequence = hidden.view(B, W, S, D)            # [B, W, S, D]

        # ── 2. Mean pooling per window (mask-aware) ───────────────────
        mask_exp    = flat_mask.unsqueeze(-1).float()      # [B*W, S, 1]
        sum_h       = (hidden * mask_exp).sum(dim=1)       # [B*W, D]
        count       = mask_exp.sum(dim=1).clamp(min=1.0)   # [B*W, 1]
        win_embs    = (sum_h / count).view(B, W, D)        # [B, W, D]

        # ── 3. Window self-attention ──────────────────────────────────
        #   Padding windows (all tokens masked) should be ignored.
        window_valid = window_attention_mask.any(dim=-1)    # [B, W]  True=real
        win_pad_mask = ~window_valid                        # True = ignore

        attended, _ = self.window_self_attn(
            win_embs, win_embs, win_embs,
            key_padding_mask=win_pad_mask,
        )
        win_embs = self.window_norm1(win_embs + self.dropout(attended))  # [B, W, D]

        # Apply FFN on window representations
        win_embs = self.window_norm2(
            win_embs + self.dropout(self.window_ffn(win_embs))
        )                                                   # [B, W, D]

        # ── 4. KW-guided cross-attention → window weights ─────────────
        #   Q = kw_pooled (one query per sample) [B, 1, D]
        #   K = V = attended windows             [B, W, D]
        query = kw_pooled.unsqueeze(1)                     # [B, 1, D]

        cross_out, window_weights = self.window_cross_attn(
            query=query,
            key=win_embs,
            value=win_embs,
            key_padding_mask=win_pad_mask,
        )
        # cross_out:      [B, 1, D]
        # window_weights: [B, num_heads, 1, W] averaged → need [B, W]
        # MultiheadAttention returns attn_output_weights averaged over heads
        # when average_attn_weights=True (default): shape [B, 1, W]
        window_weights = window_weights.squeeze(1)          # [B, W]

        # Mask padding windows (set to 0 so they don't contribute)
        window_weights = window_weights.masked_fill(win_pad_mask, 0.0)

        # ── 5. Weighted sum → doc_pooled ──────────────────────────────
        doc_pooled = (window_weights.unsqueeze(-1) * win_embs).sum(dim=1)  # [B, D]

        return doc_pooled, full_sequence, window_weights
