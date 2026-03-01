"""
fusion_layer.py — FusionLayer module.

Fuses the full document token sequence with keyword representations via
bidirectional cross-attention and a gated mechanism.

Processing pipeline
-------------------
1. merge_windows()  : stitch overlapping windows into one sequence  [B, L, D]
2. Self-attention   : contextualise doc tokens w.r.t. each other    [B, L, D]
3a. Cross-attn Doc→KW : each doc position attends to keywords       [B, L, D]
3b. Cross-attn KW→Doc : each keyword attends to doc positions       [B, K, D]
4. Gated fusion (doc): gate ∈ (0,1) blends enhanced vs original     [B, L, D]
5. FFN (doc)        : position-wise transform of doc_fused           [B, L, D]
6. FFN (kw)         : position-wise transform of kw_fused            [B, K, D]
7. Concatenate      : [doc_fused ‖ kw_fused]                        [B, L+K, D]

Output
------
encoder_hidden_states  [B, L+K, D]   combined doc+kw hidden states for decoder
encoder_attention_mask [B, L+K]      combined boolean attention mask
gate_values            [B, L]        per-position gate scalar (mean over dim D)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """Bidirectional cross-attention fusion of document and keyword representations.

    Parameters
    ----------
    hidden_size   : int — model hidden dimension (default 768).
    num_heads     : int — attention heads (default 12).
    ffn_dim       : int — inner FFN dimension (default 3072).
    window_overlap: int — token overlap between consecutive windows (default 128).
                    Used by merge_windows to de-duplicate overlapping tokens.
    max_src_len   : int — maximum merged sequence length fed to the decoder
                    (0 = no cap; default 4096 ≈ 8 windows × stride).
    dropout       : float — dropout probability (default 0.1).
    """

    def __init__(
        self,
        hidden_size:    int = 768,
        num_heads:      int = 12,
        ffn_dim:        int = 3072,
        window_overlap: int = 128,
        max_src_len:    int = 4096,
        dropout:        float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size    = hidden_size
        self.window_overlap = window_overlap
        self.max_src_len    = max_src_len

        # ── Self-attention on doc sequence ────────────────────────────
        self.doc_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.doc_norm_sa = nn.LayerNorm(hidden_size)

        # ── Cross-attention: Doc → KW ─────────────────────────────────
        self.doc_to_kw_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.doc_norm_ca = nn.LayerNorm(hidden_size)

        # ── Cross-attention: KW → Doc ─────────────────────────────────
        self.kw_to_doc_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.kw_norm_ca = nn.LayerNorm(hidden_size)

        # ── Gated fusion (document side) ──────────────────────────────
        # Input: concat(doc_ctx, doc_enhanced) → [B, L, 2*D]
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)

        # ── FFN (document) ────────────────────────────────────────────
        self.doc_ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.doc_norm_ffn = nn.LayerNorm(hidden_size)

        # ── FFN (keywords) ────────────────────────────────────────────
        self.kw_ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.kw_norm_ffn = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        full_sequence:          torch.Tensor,   # [B, W, S, D]
        window_attention_mask:  torch.Tensor,   # [B, W, S]
        kw_embeddings:          torch.Tensor,   # [B, K, D]
        kw_mask:                torch.Tensor,   # [B, K]  bool  True=real KW
    ):
        """Run the fusion forward pass.

        Returns
        -------
        encoder_hidden_states   : torch.Tensor  [B, L+K, D]
        encoder_attention_mask  : torch.Tensor  [B, L+K]   (1=attend, 0=ignore)
        gate_values             : torch.Tensor  [B, L]
        """
        # ── 1. Merge overlapping windows into one flat sequence ───────
        doc_seq, doc_mask = self._merge_windows(full_sequence, window_attention_mask)
        # doc_seq:  [B, L, D]   doc_mask: [B, L]  (1=real, 0=pad)

        # Padding mask for attention (True = ignore)
        doc_pad  = (doc_mask == 0)    # [B, L]
        kw_pad   = ~kw_mask           # [B, K]

        # ── 2. Self-attention on doc sequence ─────────────────────────
        sa_out, _ = self.doc_self_attn(
            doc_seq, doc_seq, doc_seq,
            key_padding_mask=doc_pad,
        )
        doc_ctx = self.doc_norm_sa(doc_seq + self.dropout(sa_out))  # [B, L, D]

        # ── 3a. Cross-attention: Doc → KW ─────────────────────────────
        #   Q = doc_ctx  [B, L, D]
        #   K = V = kw_embeddings  [B, K, D]
        doc_enh, _ = self.doc_to_kw_attn(
            query=doc_ctx,
            key=kw_embeddings,
            value=kw_embeddings,
            key_padding_mask=kw_pad,
        )
        doc_enhanced = self.doc_norm_ca(doc_ctx + self.dropout(doc_enh))  # [B, L, D]

        # ── 3b. Cross-attention: KW → Doc ─────────────────────────────
        #   Q = kw_embeddings  [B, K, D]
        #   K = V = doc_ctx    [B, L, D]
        kw_enh, _ = self.kw_to_doc_attn(
            query=kw_embeddings,
            key=doc_ctx,
            value=doc_ctx,
            key_padding_mask=doc_pad,
        )
        kw_enhanced = self.kw_norm_ca(kw_embeddings + self.dropout(kw_enh))  # [B, K, D]

        # ── 4. Gated fusion (document) ────────────────────────────────
        gate_input = torch.cat([doc_ctx, doc_enhanced], dim=-1)  # [B, L, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))          # [B, L, D]
        doc_fused = gate * doc_enhanced + (1.0 - gate) * doc_ctx  # [B, L, D]
        gate_values = gate.mean(dim=-1)                            # [B, L]

        # ── 5. FFN (doc) ──────────────────────────────────────────────
        doc_fused = self.doc_norm_ffn(
            doc_fused + self.dropout(self.doc_ffn(doc_fused))
        )                                                          # [B, L, D]

        # ── 6. FFN (kw) ───────────────────────────────────────────────
        kw_fused = self.kw_norm_ffn(
            kw_enhanced + self.dropout(self.kw_ffn(kw_enhanced))
        )                                                          # [B, K, D]

        # ── 7. Concatenate doc and kw representations ─────────────────
        encoder_hidden_states  = torch.cat([doc_fused, kw_fused], dim=1)   # [B, L+K, D]

        kw_mask_long = kw_mask.long()                              # [B, K]  0/1
        encoder_attention_mask = torch.cat([doc_mask, kw_mask_long], dim=1)  # [B, L+K]

        return encoder_hidden_states, encoder_attention_mask, gate_values

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_windows(
        self,
        full_sequence:         torch.Tensor,   # [B, W, S, D]
        window_attention_mask: torch.Tensor,   # [B, W, S]
    ):
        """Stitch overlapping windows into a single contiguous sequence.

        Window 0 contributes all S tokens.
        Window k (k ≥ 1) contributes only its stride tokens
        (tokens [overlap:]), skipping the repeated overlap region.

        If max_src_len > 0, the merged sequence is truncated to that length.

        Returns
        -------
        merged_seq  : [B, merged_len, D]
        merged_mask : [B, merged_len]   long  (1=real, 0=pad)
        """
        B, W, S, D = full_sequence.shape
        overlap = self.window_overlap

        if W == 1:
            return full_sequence[:, 0], window_attention_mask[:, 0].long()

        seq_parts  = [full_sequence[:, 0]]                  # [B, S, D]
        mask_parts = [window_attention_mask[:, 0].long()]   # [B, S]

        for w in range(1, W):
            seq_parts.append(full_sequence[:, w, overlap:])              # [B, stride, D]
            mask_parts.append(window_attention_mask[:, w, overlap:].long())  # [B, stride]

        merged_seq  = torch.cat(seq_parts,  dim=1)  # [B, S + (W-1)*stride, D]
        merged_mask = torch.cat(mask_parts, dim=1)  # [B, S + (W-1)*stride]

        # Optionally cap the sequence length
        if self.max_src_len and merged_seq.size(1) > self.max_src_len:
            merged_seq  = merged_seq[:,  :self.max_src_len]
            merged_mask = merged_mask[:, :self.max_src_len]

        return merged_seq, merged_mask
