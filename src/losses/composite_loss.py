"""
composite_loss.py — Composite loss for the Dual-Encoder Summarizer.

Four components
---------------
L_gen   (λ=0.65) : Cross-entropy with label smoothing.
L_cover (λ=0.15) : Keyword coverage — MSE between time-averaged
                   decoder→keyword attention and normalised target KW scores.
L_bert  (λ=0.15) : Soft BERTScore in T5 embedding space
                   (fully differentiable, no external model).
L_gate  (λ=0.05) : Hinge loss that penalises under-active gate values.

Total
-----
L = λ_gen·L_gen + λ_cover·L_cover + λ_bert·L_bert + λ_gate·L_gate
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. Generative loss  (cross-entropy + label smoothing)
# ═══════════════════════════════════════════════════════════════════════════

class GenerativeLoss(nn.Module):
    """Cross-entropy loss with optional label smoothing.

    Parameters
    ----------
    label_smoothing : float — smoothing factor applied during training
                      (default 0.1).
    ignore_index    : int   — token id excluded from loss computation
                      (default -100, consistent with HuggingFace convention).
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        ignore_index:    int   = -100,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits: torch.Tensor,   # [B, T, V]
        labels: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """
        Returns
        -------
        Scalar cross-entropy loss averaged over non-masked positions.
        """
        B, T, V = logits.shape
        return self.ce(logits.reshape(B * T, V), labels.reshape(B * T))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Keyword coverage loss  (MSE between attention distribution and scores)
# ═══════════════════════════════════════════════════════════════════════════

class KeywordCoverageLoss(nn.Module):
    """MSE between time-averaged decoder→keyword attention and target scores.

    The loss encourages the decoder to distribute attention over keywords in
    proportion to their importance scores.

    Parameters
    ----------
    ignore_index : int   — label pad value used to mask invalid decoder
                   positions when averaging attention (default -100).
    eps          : float — clamp for division stability (default 1e-8).
    """

    def __init__(
        self,
        ignore_index: int   = -100,
        eps:          float = 1e-8,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(
        self,
        kw_attn_weights: torch.Tensor,          # [B, T, K]
        kw_scores:       torch.Tensor,          # [B, K]
        kw_mask:         torch.Tensor,          # [B, K]  bool  True=real KW
        labels:          Optional[torch.Tensor] = None,  # [B, T]
    ) -> torch.Tensor:
        """
        Returns
        -------
        Scalar MSE loss.
        """
        B, T, K = kw_attn_weights.shape

        # ── Token-level validity mask (exclude label-padding positions) ──
        if labels is not None:
            tok_valid = (labels != self.ignore_index).float()   # [B, T]
        else:
            tok_valid = torch.ones(B, T, device=kw_attn_weights.device)

        # ── Average attention over valid T positions ──────────────────
        tok_exp  = tok_valid.unsqueeze(-1)                       # [B, T, 1]
        sum_attn = (kw_attn_weights * tok_exp).sum(dim=1)       # [B, K]
        count    = tok_exp.sum(dim=1).clamp(min=self.eps)        # [B, 1]
        pred     = sum_attn / count                              # [B, K]

        # ── Normalise target keyword scores ──────────────────────────
        scores_m = kw_scores.masked_fill(~kw_mask, float("-inf"))  # [B, K]
        target   = torch.softmax(scores_m, dim=-1)                 # [B, K]
        target   = torch.nan_to_num(target, nan=0.0)               # all-pad guard

        # ── Mask padding keyword slots ────────────────────────────────
        kw_float  = kw_mask.float()
        pred_m    = pred   * kw_float
        target_m  = target * kw_float

        return F.mse_loss(pred_m, target_m, reduction="mean")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Soft BERTScore loss  (differentiable, T5 embedding space)
# ═══════════════════════════════════════════════════════════════════════════

class SoftBERTScoreLoss(nn.Module):
    """Differentiable BERTScore F1 computed in T5 embedding space.

    Prediction embeddings
        soft_pred = softmax(logits) @ E   ∈ ℝ^{B×T×D}

    are compared to reference (hard) embeddings

        hard_ref  = E[label_ids]           ∈ ℝ^{B×T×D}

    via greedy cosine-similarity matching (BERTScore definition).
    Loss = 1 − F1, averaged over the batch.

    Parameters
    ----------
    ignore_index : int — label pad value excluded from the comparison
                   (default -100).
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits:           torch.Tensor,   # [B, T, V]
        labels:           torch.Tensor,   # [B, T]
        embedding_matrix: torch.Tensor,   # [V, D]
    ) -> torch.Tensor:
        """
        Returns
        -------
        Scalar loss = 1 − mean(BERTScore-F1).
        """
        B, T, V = logits.shape
        device   = logits.device

        # ── Soft prediction embeddings ────────────────────────────────
        probs     = torch.softmax(logits.float(), dim=-1)        # [B, T, V]
        soft_pred = probs @ embedding_matrix.float()             # [B, T, D]

        # ── Hard reference embeddings ─────────────────────────────────
        safe_labels = labels.clone()
        safe_labels[safe_labels == self.ignore_index] = 0
        hard_ref = embedding_matrix[safe_labels].float()         # [B, T, D]

        # ── Per-sample validity mask ──────────────────────────────────
        valid = (labels != self.ignore_index)                    # [B, T]

        losses: list[torch.Tensor] = []
        for b in range(B):
            p = soft_pred[b][valid[b]]   # [Tp, D]
            r = hard_ref[b][valid[b]]    # [Tp, D]  (same positions)

            if p.shape[0] == 0:
                # No valid tokens; contribute zero (no gradient)
                losses.append(torch.tensor(0.0, device=device))
                continue

            p_n = F.normalize(p, dim=-1)   # [Tp, D]
            r_n = F.normalize(r, dim=-1)   # [Tp, D]

            # Cosine similarity matrix  [Tp, Tp]
            sim = p_n @ r_n.T

            # BERTScore Precision: for each pred token, best-matching ref
            precision = sim.max(dim=1).values.mean()
            # BERTScore Recall:    for each ref  token, best-matching pred
            recall    = sim.max(dim=0).values.mean()

            denom = precision + recall
            f1 = torch.where(
                denom > 0,
                2.0 * precision * recall / denom.clamp(min=1e-8),
                torch.zeros_like(denom),
            )
            losses.append(1.0 - f1)

        return torch.stack(losses).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Gate loss  (hinge: penalise under-active gates)
# ═══════════════════════════════════════════════════════════════════════════

class GateLoss(nn.Module):
    """Bilateral hinge loss that keeps gate values inside [low, high].

    L_gate = mean(
        ReLU(low  − mean_gate)      ← penalise under-active gates
      + ReLU(mean_gate − high)      ← penalise over-active gates
    )

    The loss is zero when low ≤ mean_gate ≤ high, and grows linearly
    outside the corridor.

    Applied separately to the fusion gate and, optionally, the KAL gate.

    Parameters
    ----------
    threshold_low  : float — minimum desired average gate value (default 0.2).
    threshold_high : float — maximum desired average gate value (default 0.5).
    """

    def __init__(
        self,
        threshold_low:  float = 0.2,
        threshold_high: float = 0.5,
    ) -> None:
        super().__init__()
        self.threshold_low  = threshold_low
        self.threshold_high = threshold_high

    def forward(
        self,
        fusion_gate_values: torch.Tensor,                      # [B, L]
        kal_gate_values:    Optional[torch.Tensor] = None,     # [B, T]
    ) -> torch.Tensor:
        """
        Returns
        -------
        Scalar hinge loss.
        """
        components: list[torch.Tensor] = [
            F.relu(self.threshold_low  - fusion_gate_values.mean())
            + F.relu(fusion_gate_values.mean() - self.threshold_high),
        ]
        if kal_gate_values is not None:
            components.append(
                F.relu(self.threshold_low  - kal_gate_values.mean())
                + F.relu(kal_gate_values.mean() - self.threshold_high),
            )
        return torch.stack(components).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Composite loss
# ═══════════════════════════════════════════════════════════════════════════

class CompositeLoss(nn.Module):
    """Weighted sum of the four training objectives.

    L = λ_gen·L_gen + λ_cover·L_cover + λ_bert·L_bert + λ_gate·L_gate

    Parameters
    ----------
    lambda_gen     : float — weight for L_gen   (default 0.65).
    lambda_cover   : float — weight for L_cover (default 0.15).
    lambda_bert    : float — weight for L_bert  (default 0.15).
    lambda_gate    : float — weight for L_gate  (default 0.05).
    label_smoothing: float — smoothing for L_gen (default 0.1).
    gate_threshold_low  : float — min desired gate value (default 0.2).
    gate_threshold_high : float — max desired gate value (default 0.5).
    ignore_index   : int   — padding label id (default -100).
    """

    def __init__(
        self,
        lambda_gen:          float = 0.65,
        lambda_cover:        float = 0.15,
        lambda_bert:         float = 0.15,
        lambda_gate:         float = 0.05,
        label_smoothing:     float = 0.1,
        gate_threshold_low:  float = 0.2,
        gate_threshold_high: float = 0.5,
        ignore_index:        int   = -100,
    ) -> None:
        super().__init__()
        self.lambda_gen   = lambda_gen
        self.lambda_cover = lambda_cover
        self.lambda_bert  = lambda_bert
        self.lambda_gate  = lambda_gate

        self.gen_loss   = GenerativeLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        self.cover_loss = KeywordCoverageLoss(ignore_index=ignore_index)
        self.bert_loss  = SoftBERTScoreLoss(ignore_index=ignore_index)
        self.gate_loss  = GateLoss(
            threshold_low=gate_threshold_low,
            threshold_high=gate_threshold_high,
        )

    def forward(
        self,
        logits:             torch.Tensor,          # [B, T, V]
        labels:             torch.Tensor,          # [B, T]
        embedding_matrix:   torch.Tensor,          # [V, D]
        kw_attn_weights:    torch.Tensor,          # [B, T, K]
        kw_scores:          torch.Tensor,          # [B, K]
        kw_mask:            torch.Tensor,          # [B, K]  bool
        fusion_gate_values: torch.Tensor,          # [B, L]
        kal_gate_values:    Optional[torch.Tensor] = None,  # [B, T]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns
        -------
        total_loss : scalar Tensor (differentiable)
        components : dict with keys 'l_gen', 'l_cover', 'l_bert', 'l_gate'
        """
        l_gen   = self.gen_loss(logits, labels)
        l_cover = self.cover_loss(kw_attn_weights, kw_scores, kw_mask, labels)
        l_bert  = self.bert_loss(logits, labels, embedding_matrix)
        l_gate  = self.gate_loss(fusion_gate_values, kal_gate_values)

        total = (
            self.lambda_gen   * l_gen
            + self.lambda_cover * l_cover
            + self.lambda_bert  * l_bert
            + self.lambda_gate  * l_gate
        )

        components: Dict[str, float] = {
            "l_gen":   l_gen.item(),
            "l_cover": l_cover.item(),
            "l_bert":  l_bert.item(),
            "l_gate":  l_gate.item(),
        }
        return total, components
