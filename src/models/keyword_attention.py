"""
keyword_attention.py — KeywordAttentionLayer (KAL) module.

Applied AFTER the T5 decoder, this module explicitly attends to keyword
representations once more, giving the decoder a second keyword signal
(the first is through the concatenated encoder_hidden_states).

Decoder "sees" keywords TWICE:
  1. Implicitly — via T5 cross-attention on encoder_hidden_states
     which contains kw_fused positions (from FusionLayer concat).
  2. Explicitly — via this module's dedicated cross-attention.

Processing pipeline
-------------------
1. Cross-attention (Dec → KW)  : per-step keyword attention        [B, T, D]
   Optionally weight attention scores by kw_scores (score-weighted attention).
2. Gated fusion                : blend attended KW into decoder repr [B, T, D]
3. FFN                         : position-wise transform              [B, T, D]
4. LM Head (shared with T5)   : project to vocabulary logits         [B, T, V]

Output
------
enhanced_logits  [B, T, V]   vocabulary logits (after KW-enhanced hidden states)
kw_attn_weights  [B, T, K]   per-step keyword attention weights (for coverage loss)
gate_values      [B, T]      per-step gate scalar for gate loss
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KeywordAttentionLayer(nn.Module):
    """Explicit post-decoder cross-attention to keyword representations.

    Parameters
    ----------
    hidden_size  : int — d_model (default 768).
    num_heads    : int — attention heads (default 8).
    ffn_dim      : int — inner FFN dimension (default 1536).
    vocab_size   : int — vocabulary size (default 32128 for ruT5-base).
    dropout      : float — dropout probability (default 0.1).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads:   int = 8,
        ffn_dim:     int = 1536,
        vocab_size:  int = 32128,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # ── Cross-attention: decoder hidden → keyword representations ─
        self.kw_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.kw_norm = nn.LayerNorm(hidden_size)

        # ── Gated fusion ──────────────────────────────────────────────
        # gate ∈ (0, 1)^D per (batch, position, dim)
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)

        # ── FFN ───────────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

        # ── LM Head — initialized to None; MUST be set before use ─────
        # The lm_head weights are shared with T5 (set via set_lm_head).
        self.lm_head: nn.Module | None = None

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Attach the T5 LM head (shared weights — no duplication)."""
        self.lm_head = lm_head

    # ------------------------------------------------------------------

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # [B, T, D]
        kw_hidden:      torch.Tensor,   # [B, K, D]
        kw_mask:        torch.Tensor,   # [B, K]  bool  True=real KW
        kw_scores:      torch.Tensor,   # [B, K]  float scores
    ):
        """Forward pass.

        Returns
        -------
        enhanced_logits : torch.Tensor  [B, T, V]
        kw_attn_weights : torch.Tensor  [B, T, K]
        gate_values     : torch.Tensor  [B, T]
        """
        if self.lm_head is None:
            raise RuntimeError(
                "KeywordAttentionLayer.lm_head is not set. "
                "Call set_lm_head(t5_model.lm_head) before using the model."
            )

        B, T, D = decoder_hidden.shape
        K = kw_hidden.size(1)

        # ── 1. Cross-attention: decoder → keywords ────────────────────
        #   Q = decoder_hidden  [B, T, D]
        #   K = V = kw_hidden   [B, K, D]
        #   key_padding_mask: True = padding keyword → ignore
        kw_pad_mask = ~kw_mask                            # [B, K]

        kw_attended, raw_attn = self.kw_cross_attn(
            query=decoder_hidden,
            key=kw_hidden,
            value=kw_hidden,
            key_padding_mask=kw_pad_mask,
        )
        # raw_attn: [B, T, K]  (averaged over heads, default behaviour)
        kw_attended = self.kw_norm(decoder_hidden + self.dropout(kw_attended))  # [B, T, D]

        # ── Score-weighted attention (optional, improves focus on top KW) ──
        # Re-weight raw attention by kw_scores to amplify high-ranked KWs.
        # scores: [B, K] → broadcast to [B, T, K]
        kw_scores_masked = kw_scores.masked_fill(~kw_mask, 0.0)   # [B, K]
        score_weights = kw_scores_masked.unsqueeze(1).expand(B, T, K)  # [B, T, K]
        # Multiply raw attention by score weights and re-normalise
        weighted_attn = raw_attn * score_weights                   # [B, T, K]
        attn_sum = weighted_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        kw_attn_weights = weighted_attn / attn_sum                 # [B, T, K]

        # ── 2. Gated fusion ───────────────────────────────────────────
        gate_input = torch.cat([decoder_hidden, kw_attended], dim=-1)  # [B, T, 2D]
        gate       = torch.sigmoid(self.gate_proj(gate_input))          # [B, T, D]

        enhanced = gate * kw_attended + (1.0 - gate) * decoder_hidden  # [B, T, D]
        gate_values = gate.mean(dim=-1)                                 # [B, T]

        # ── 3. FFN ────────────────────────────────────────────────────
        enhanced = self.ffn_norm(enhanced + self.dropout(self.ffn(enhanced)))  # [B, T, D]

        # ── 4. LM Head (shared weights) ───────────────────────────────
        enhanced_logits = self.lm_head(enhanced)                        # [B, T, V]

        return enhanced_logits, kw_attn_weights, gate_values
