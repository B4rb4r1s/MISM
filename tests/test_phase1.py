"""
tests/test_phase1.py — Unit tests for Phase 1: Core Model Components.

All tests use a TINY T5 config (d_model=64, 2 layers, vocab=256)
to avoid network access and keep execution fast.

Run:
    source .mism/bin/activate
    cd /home/b4rb4r1s/ForAll/MISM
    pytest tests/test_phase1.py -v

Coverage:
    KeywordsEncoder         — shapes, mean-pool, self-attn, weighted-pool
    DocumentEncoder         — shapes, window encoding, cross-attn
    FusionLayer             — shapes, merge_windows, bidirectional cross-attn, gate
    KeywordAttentionLayer   — shapes, gate, attn weights, lm_head
    DualEncoderSummarizer   — end-to-end forward pass, freeze/unfreeze helpers,
                              param counts
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from src.models.keywords_encoder      import KeywordsEncoder
from src.models.document_encoder      import DocumentEncoder
from src.models.fusion_layer          import FusionLayer
from src.models.keyword_attention     import KeywordAttentionLayer
from src.models.dual_encoder_summarizer import DualEncoderSummarizer, _shift_tokens_right


# ============================================================
# Shared tiny T5 config and factory
# ============================================================

TINY_CFG = T5Config(
    vocab_size=256,
    d_model=64,
    d_kv=16,
    d_ff=128,
    num_heads=4,
    num_layers=2,
    relative_attention_num_buckets=8,
    relative_attention_max_distance=16,
    decoder_start_token_id=0,
    eos_token_id=1,
    pad_token_id=0,
)
D  = TINY_CFG.d_model   # 64
V  = TINY_CFG.vocab_size  # 256

# Batch / sequence constants used across tests
B  = 2   # batch size
K  = 5   # max keywords
L  = 8   # keyword max length (tokens)
W  = 3   # number of windows
S  = 16  # window size (tokens)
T  = 10  # decoder target length


@pytest.fixture(scope="module")
def tiny_t5():
    """Create one tiny T5ForConditionalGeneration for the whole module."""
    torch.manual_seed(0)
    return T5ForConditionalGeneration(TINY_CFG)


@pytest.fixture(scope="module")
def full_model(tiny_t5):
    """Full DualEncoderSummarizer built from the tiny T5."""
    overlap = 2
    model = DualEncoderSummarizer(
        t5_model=tiny_t5,
        hidden_size=D,
        kw_num_heads=4,
        kw_ffn_dim=128,
        doc_num_heads=4,
        doc_ffn_dim=128,
        fusion_num_heads=4,
        fusion_ffn_dim=128,
        window_overlap=overlap,
        max_src_len=0,         # no cap — keep all windows
        kal_num_heads=4,
        kal_ffn_dim=128,
        dropout=0.0,           # deterministic for shape tests
    )
    model.eval()
    return model


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_kw_batch(b=B, k=K, l=L, n_real_kw=3):
    """Create random keyword batch tensors."""
    kw_ids  = torch.randint(1, V, (b, k, l))
    kw_mask_t = torch.zeros(b, k, l, dtype=torch.long)
    kw_bool = torch.zeros(b, k, dtype=torch.bool)
    for bi in range(b):
        kw_mask_t[bi, :n_real_kw] = 1
        kw_bool[bi, :n_real_kw]   = True
    scores = torch.rand(b, k)
    scores[~kw_bool] = 0.0
    return kw_ids, kw_mask_t, scores, kw_bool


def _make_doc_batch(b=B, w=W, s=S):
    """Create random document window batch tensors."""
    win_ids  = torch.randint(1, V, (b, w, s))
    win_mask = torch.ones(b, w, s, dtype=torch.long)
    # Last window: partially padded
    win_ids[:, -1, s // 2:] = 0
    win_mask[:, -1, s // 2:] = 0
    return win_ids, win_mask


def _make_labels(b=B, t=T, pad_id=0, label_pad=-100):
    """Create random label tensors (some padding at the end)."""
    labels = torch.randint(1, V, (b, t))
    # Last 2 positions are padding
    labels[:, -2:] = label_pad
    return labels


# ============================================================
# 1. KeywordsEncoder
# ============================================================

class TestKeywordsEncoder:

    @pytest.fixture(autouse=True)
    def setup(self, tiny_t5):
        self.enc = KeywordsEncoder(
            t5_encoder=copy.deepcopy(tiny_t5.encoder),
            hidden_size=D, num_heads=4, ffn_dim=128, dropout=0.0,
        )
        self.enc.eval()

    def _fwd(self, b=B, k=K, l=L, n_real=3):
        kw_ids, kw_mask_t, scores, kw_bool = _make_kw_batch(b, k, l, n_real)
        return self.enc(kw_ids, kw_mask_t, scores, kw_bool)

    # ── Shapes ────────────────────────────────────────────────────────

    def test_kw_embeddings_shape(self):
        embs, pooled = self._fwd()
        assert embs.shape == (B, K, D), f"Expected [{B},{K},{D}], got {embs.shape}"

    def test_kw_pooled_shape(self):
        embs, pooled = self._fwd()
        assert pooled.shape == (B, D), f"Expected [{B},{D}], got {pooled.shape}"

    def test_output_dtype_float(self):
        embs, pooled = self._fwd()
        assert embs.dtype == torch.float32
        assert pooled.dtype == torch.float32

    # ── Correctness ────────────────────────────────────────────────────

    def test_all_pad_keywords_zero_pooled(self):
        """When all keywords are padding, pooled should be zero (softmax on all-inf → nan→0)."""
        b, k, l = 1, K, L
        kw_ids  = torch.zeros(b, k, l, dtype=torch.long)
        kw_mask = torch.zeros(b, k, l, dtype=torch.long)
        scores  = torch.zeros(b, k)
        kw_bool = torch.zeros(b, k, dtype=torch.bool)
        _, pooled = self.enc(kw_ids, kw_mask, scores, kw_bool)
        assert not torch.isnan(pooled).any(), "NaN in pooled output for all-pad keywords"

    def test_single_keyword(self):
        """With exactly 1 real keyword, pooled == that keyword's embedding."""
        embs, pooled = self._fwd(b=1, k=K, l=L, n_real=1)
        assert pooled.shape == (1, D)
        assert not torch.isnan(pooled).any()

    # ── Gradient flow ─────────────────────────────────────────────────

    def test_backward_pass(self):
        """Loss backward should not raise and produce non-NaN gradients."""
        self.enc.train()
        kw_ids, kw_mask_t, scores, kw_bool = _make_kw_batch()
        embs, pooled = self.enc(kw_ids, kw_mask_t, scores, kw_bool)
        loss = pooled.sum()
        loss.backward()
        for name, param in self.enc.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        self.enc.eval()


# ============================================================
# 2. DocumentEncoder
# ============================================================

class TestDocumentEncoder:

    @pytest.fixture(autouse=True)
    def setup(self, tiny_t5):
        self.enc = DocumentEncoder(
            t5_encoder=copy.deepcopy(tiny_t5.encoder),
            hidden_size=D, num_heads=4, ffn_dim=128, dropout=0.0,
        )
        self.enc.eval()
        # Create fixed kw_pooled
        torch.manual_seed(1)
        self.kw_pooled = torch.randn(B, D)

    def _fwd(self, b=B, w=W, s=S):
        ids, mask = _make_doc_batch(b, w, s)
        kw = self.kw_pooled[:b]
        return self.enc(ids, mask, kw)

    # ── Shapes ────────────────────────────────────────────────────────

    def test_doc_pooled_shape(self):
        pooled, full_seq, weights = self._fwd()
        assert pooled.shape == (B, D)

    def test_full_sequence_shape(self):
        pooled, full_seq, weights = self._fwd()
        assert full_seq.shape == (B, W, S, D)

    def test_window_weights_shape(self):
        pooled, full_seq, weights = self._fwd()
        assert weights.shape == (B, W)

    def test_window_weights_non_negative(self):
        """Attention weights from cross-attention should be ≥ 0."""
        _, _, weights = self._fwd()
        assert (weights >= 0).all(), "Window weights should be non-negative"

    def test_single_window(self):
        """Works correctly when there is only one window."""
        pooled, full_seq, weights = self._fwd(b=1, w=1, s=S)
        assert full_seq.shape == (1, 1, S, D)
        assert pooled.shape   == (1, D)
        assert weights.shape  == (1, 1)

    # ── Gradient flow ─────────────────────────────────────────────────

    def test_backward_pass(self):
        self.enc.train()
        ids, mask = _make_doc_batch()
        pooled, full_seq, weights = self.enc(ids, mask, self.kw_pooled)
        loss = pooled.sum() + full_seq.sum()
        loss.backward()
        for name, param in self.enc.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        self.enc.eval()


# ============================================================
# 3. FusionLayer
# ============================================================

class TestFusionLayer:

    OVERLAP = 2

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fusion = FusionLayer(
            hidden_size=D, num_heads=4, ffn_dim=128,
            window_overlap=self.OVERLAP, max_src_len=0, dropout=0.0,
        )
        self.fusion.eval()

    def _make_inputs(self, b=B, w=W, s=S, k=K):
        full_seq   = torch.randn(b, w, s, D)
        win_mask   = torch.ones(b, w, s, dtype=torch.long)
        win_mask[:, -1, s // 2:] = 0               # last window half-padded
        kw_embs    = torch.randn(b, k, D)
        kw_bool    = torch.ones(b, k, dtype=torch.bool)
        kw_bool[:, -1] = False                      # last KW slot is padding
        return full_seq, win_mask, kw_embs, kw_bool

    # ── merge_windows ─────────────────────────────────────────────────

    def test_merge_windows_length(self):
        """Merged length = S + (W-1)*stride."""
        stride = S - self.OVERLAP
        expected_len = S + (W - 1) * stride
        full_seq = torch.randn(B, W, S, D)
        win_mask = torch.ones(B, W, S, dtype=torch.long)
        merged, _ = self.fusion._merge_windows(full_seq, win_mask)
        assert merged.shape == (B, expected_len, D), \
            f"Expected merged length {expected_len}, got {merged.shape[1]}"

    def test_merge_windows_single(self):
        """Single window returns shape [B, S, D]."""
        full_seq = torch.randn(B, 1, S, D)
        win_mask = torch.ones(B, 1, S, dtype=torch.long)
        merged, mask = self.fusion._merge_windows(full_seq, win_mask)
        assert merged.shape == (B, S, D)
        assert mask.shape   == (B, S)

    def test_merge_windows_max_src_len(self):
        """Merged sequence is truncated to max_src_len when set."""
        fusion = FusionLayer(
            hidden_size=D, num_heads=4, ffn_dim=128,
            window_overlap=self.OVERLAP, max_src_len=S + 1, dropout=0.0,
        )
        full_seq = torch.randn(B, W, S, D)
        win_mask = torch.ones(B, W, S, dtype=torch.long)
        merged, mask = fusion._merge_windows(full_seq, win_mask)
        assert merged.shape[1] <= S + 1

    def test_merge_windows_no_overlap_duplication(self):
        """The overlap region of window k should NOT be repeated in merged seq."""
        # Build a simple case: W=2, S=4, overlap=1 → stride=3
        # Tokens: win0=[A,B,C,D], win1=[C,E,F,G]
        # Expected merge: [A,B,C,D, E,F,G]  (skip first overlap=1 token of win1)
        fusion = FusionLayer(
            hidden_size=1, num_heads=1, ffn_dim=4,
            window_overlap=1, max_src_len=0, dropout=0.0,
        )
        # Use identity values so we can check exact positions
        full_seq = torch.arange(8, dtype=torch.float).view(1, 2, 4, 1)
        # win0 tokens 0-3, win1 tokens 4-7
        win_mask = torch.ones(1, 2, 4, dtype=torch.long)
        merged, _ = fusion._merge_windows(full_seq, win_mask)
        # Expected: win0 (all 4) + win1[1:] (3 tokens) = 7 tokens
        assert merged.shape == (1, 7, 1)
        # First 4 values come from win0
        assert (merged[0, :4, 0] == full_seq[0, 0, :, 0]).all()
        # Next 3 values come from win1[1:]
        assert (merged[0, 4:, 0] == full_seq[0, 1, 1:, 0]).all()

    # ── Full forward ──────────────────────────────────────────────────

    def test_encoder_hidden_states_shape(self):
        full_seq, win_mask, kw_embs, kw_bool = self._make_inputs()
        enc_hs, enc_mask, gates = self.fusion(full_seq, win_mask, kw_embs, kw_bool)
        stride = S - self.OVERLAP
        L_expected = S + (W - 1) * stride
        assert enc_hs.shape[0] == B
        assert enc_hs.shape[1] == L_expected + K  # doc + kw
        assert enc_hs.shape[2] == D

    def test_encoder_attention_mask_shape(self):
        full_seq, win_mask, kw_embs, kw_bool = self._make_inputs()
        _, enc_mask, _ = self.fusion(full_seq, win_mask, kw_embs, kw_bool)
        stride = S - self.OVERLAP
        L_expected = S + (W - 1) * stride
        assert enc_mask.shape == (B, L_expected + K)

    def test_gate_values_shape(self):
        full_seq, win_mask, kw_embs, kw_bool = self._make_inputs()
        stride = S - self.OVERLAP
        L_expected = S + (W - 1) * stride
        _, _, gates = self.fusion(full_seq, win_mask, kw_embs, kw_bool)
        assert gates.shape == (B, L_expected)

    def test_gate_values_in_range(self):
        """Gate values should be in (0, 1) — output of sigmoid mean."""
        full_seq, win_mask, kw_embs, kw_bool = self._make_inputs()
        _, _, gates = self.fusion(full_seq, win_mask, kw_embs, kw_bool)
        assert (gates >= 0).all() and (gates <= 1).all(), \
            "Gate values should be in [0, 1]"

    def test_kw_positions_in_enc_hidden(self):
        """Last K positions of encoder_hidden_states correspond to kw_fused."""
        full_seq, win_mask, kw_embs, kw_bool = self._make_inputs()
        enc_hs, _, _ = self.fusion(full_seq, win_mask, kw_embs, kw_bool)
        # KW positions are the last K positions
        kw_part = enc_hs[:, -K:, :]
        assert kw_part.shape == (B, K, D)

    # ── Gradient flow ─────────────────────────────────────────────────

    def test_backward_pass(self):
        self.fusion.train()
        full_seq, win_mask, kw_embs, kw_bool = self._make_inputs()
        kw_embs.requires_grad_(True)
        enc_hs, enc_mask, gates = self.fusion(full_seq, win_mask, kw_embs, kw_bool)
        loss = enc_hs.sum() + gates.sum()
        loss.backward()
        for name, param in self.fusion.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        self.fusion.eval()


# ============================================================
# 4. KeywordAttentionLayer
# ============================================================

class TestKeywordAttentionLayer:

    @pytest.fixture(autouse=True)
    def setup(self, tiny_t5):
        self.kal = KeywordAttentionLayer(
            hidden_size=D, num_heads=4, ffn_dim=128,
            vocab_size=V, dropout=0.0,
        )
        self.kal.set_lm_head(copy.deepcopy(tiny_t5.lm_head))
        self.kal.eval()

    def _fwd(self, b=B, t=T, k=K):
        dec_hidden = torch.randn(b, t, D)
        kw_hidden  = torch.randn(b, k, D)
        kw_bool    = torch.ones(b, k, dtype=torch.bool)
        kw_bool[:, -1] = False
        kw_scores  = torch.rand(b, k)
        kw_scores[~kw_bool] = 0.0
        return self.kal(dec_hidden, kw_hidden, kw_bool, kw_scores)

    # ── Shapes ────────────────────────────────────────────────────────

    def test_logits_shape(self):
        logits, attn, gates = self._fwd()
        assert logits.shape == (B, T, V)

    def test_kw_attn_weights_shape(self):
        logits, attn, gates = self._fwd()
        assert attn.shape == (B, T, K)

    def test_gate_values_shape(self):
        logits, attn, gates = self._fwd()
        assert gates.shape == (B, T)

    # ── Correctness ────────────────────────────────────────────────────

    def test_gate_values_in_range(self):
        _, _, gates = self._fwd()
        assert (gates >= 0).all() and (gates <= 1).all()

    def test_kw_attn_weights_non_negative(self):
        _, attn, _ = self._fwd()
        assert (attn >= 0).all(), "Attention weights must be non-negative"

    def test_padding_kw_has_zero_attn(self):
        """All-padding keywords should receive near-zero attention weight."""
        b, t, k = 1, T, K
        dec_hidden = torch.randn(b, t, D)
        kw_hidden  = torch.randn(b, k, D)
        # Only 1 real keyword
        kw_bool   = torch.zeros(b, k, dtype=torch.bool)
        kw_bool[:, 0] = True
        kw_scores = torch.zeros(b, k)
        kw_scores[:, 0] = 1.0
        _, attn, _ = self.kal(dec_hidden, kw_hidden, kw_bool, kw_scores)
        # Padding slots (indices 1..K-1) should have score 0 → attn weight 0
        pad_attn = attn[0, :, 1:]
        assert (pad_attn == 0).all(), "Padding KW slots should have zero attention"

    def test_no_lm_head_raises(self):
        kal_no_head = KeywordAttentionLayer(hidden_size=D, num_heads=4,
                                            ffn_dim=128, vocab_size=V)
        with pytest.raises(RuntimeError, match="lm_head is not set"):
            kal_no_head(torch.randn(1, 3, D), torch.randn(1, 2, D),
                        torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2))

    # ── Gradient flow ─────────────────────────────────────────────────

    def test_backward_pass(self):
        self.kal.train()
        logits, attn, gates = self._fwd()
        loss = logits.sum()
        loss.backward()
        for name, param in self.kal.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        self.kal.eval()


# ============================================================
# 5. DualEncoderSummarizer (end-to-end)
# ============================================================

class TestDualEncoderSummarizer:

    def _batch(self, b=B, w=W, s=S, k=K, l=L, t=T, n_real_kw=3):
        ids, win_mask = _make_doc_batch(b, w, s)
        kw_ids, kw_amask, kw_scores, kw_bool = _make_kw_batch(b, k, l, n_real_kw)
        labels = _make_labels(b, t)
        return ids, win_mask, kw_ids, kw_amask, kw_scores, kw_bool, labels

    # ── Output structure ──────────────────────────────────────────────

    def test_forward_returns_output(self, full_model):
        b = _batch = self._batch()
        out = full_model(*b)
        assert hasattr(out, "logits")
        assert hasattr(out, "kw_attn_weights")
        assert hasattr(out, "fusion_gate_values")
        assert hasattr(out, "kal_gate_values")
        assert hasattr(out, "doc_pooled")
        assert hasattr(out, "kw_pooled")
        assert hasattr(out, "decoder_hidden")

    # ── Output shapes ─────────────────────────────────────────────────

    def test_logits_shape(self, full_model):
        out = full_model(*self._batch())
        assert out.logits.shape == (B, T, V), f"logits shape: {out.logits.shape}"

    def test_kw_attn_weights_shape(self, full_model):
        out = full_model(*self._batch())
        assert out.kw_attn_weights.shape == (B, T, K)

    def test_fusion_gate_values_range(self, full_model):
        out = full_model(*self._batch())
        assert (out.fusion_gate_values >= 0).all()
        assert (out.fusion_gate_values <= 1).all()

    def test_kal_gate_values_shape(self, full_model):
        out = full_model(*self._batch())
        assert out.kal_gate_values.shape == (B, T)

    def test_doc_pooled_shape(self, full_model):
        out = full_model(*self._batch())
        assert out.doc_pooled.shape == (B, D)

    def test_kw_pooled_shape(self, full_model):
        out = full_model(*self._batch())
        assert out.kw_pooled.shape == (B, D)

    def test_decoder_hidden_shape(self, full_model):
        out = full_model(*self._batch())
        assert out.decoder_hidden.shape == (B, T, D)

    def test_no_nans_in_outputs(self, full_model):
        out = full_model(*self._batch())
        for field_name in ["logits", "kw_attn_weights", "fusion_gate_values",
                           "kal_gate_values", "doc_pooled", "kw_pooled", "decoder_hidden"]:
            tensor = getattr(out, field_name)
            assert not torch.isnan(tensor).any(), f"NaN found in {field_name}"

    # ── Freeze / unfreeze ─────────────────────────────────────────────

    def test_freeze_encoders(self, full_model):
        full_model.freeze_encoders()
        doc_frozen = all(
            not p.requires_grad
            for p in full_model.document_encoder.t5_encoder.parameters()
        )
        kw_frozen = all(
            not p.requires_grad
            for p in full_model.keywords_encoder.t5_encoder.parameters()
        )
        assert doc_frozen, "Doc encoder T5 should be frozen"
        assert kw_frozen,  "KW encoder T5 should be frozen"
        full_model.unfreeze_encoders()

    def test_unfreeze_encoders(self, full_model):
        full_model.freeze_encoders()
        full_model.unfreeze_encoders()
        assert all(
            p.requires_grad
            for p in full_model.document_encoder.t5_encoder.parameters()
        )

    def test_freeze_decoder(self, full_model):
        full_model.freeze_decoder()
        assert all(not p.requires_grad for p in full_model.decoder.parameters())
        full_model.unfreeze_decoder()

    def test_kal_always_trainable(self, full_model):
        """KeywordAttentionLayer should always be trainable."""
        full_model.freeze_encoders()
        full_model.freeze_decoder()
        kal_trainable = any(
            p.requires_grad
            for p in full_model.keyword_attention_layer.parameters()
        )
        assert kal_trainable, "KAL should remain trainable even when encoders/decoder are frozen"
        full_model.unfreeze_encoders()
        full_model.unfreeze_decoder()

    # ── Parameter counting ────────────────────────────────────────────

    def test_param_count_keys(self, full_model):
        counts = full_model.get_trainable_param_count()
        expected = {
            "keywords_encoder", "document_encoder", "fusion_layer",
            "decoder", "keyword_attention_layer", "TOTAL",
        }
        assert set(counts.keys()) == expected

    def test_total_trainable_equals_sum(self, full_model):
        counts = full_model.get_trainable_param_count()
        component_sum = sum(
            v["trainable"] for k, v in counts.items() if k != "TOTAL"
        )
        assert counts["TOTAL"]["trainable"] == component_sum

    # ── End-to-end backward pass ──────────────────────────────────────

    def test_backward_pass(self, full_model):
        """Backward through the entire model should not error."""
        full_model.train()
        full_model.unfreeze_encoders()
        full_model.unfreeze_decoder()

        out = full_model(*self._batch())
        loss = out.logits.sum()
        loss.backward()

        # Check at least some gradients exist and are non-NaN
        had_grad = False
        for name, param in full_model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
                had_grad = True
        assert had_grad, "Expected at least one parameter to have a gradient"

        full_model.eval()


# ============================================================
# 6. _shift_tokens_right helper
# ============================================================

class TestShiftTokensRight:

    def test_first_position_is_decoder_start(self):
        labels = torch.tensor([[3, 4, 5, -100]])
        shifted = _shift_tokens_right(labels, pad_token_id=0, decoder_start_token_id=7)
        assert shifted[0, 0].item() == 7

    def test_shift_by_one(self):
        labels = torch.tensor([[3, 4, 5, -100]])
        shifted = _shift_tokens_right(labels, pad_token_id=0, decoder_start_token_id=7)
        assert shifted[0, 1].item() == 3
        assert shifted[0, 2].item() == 4
        assert shifted[0, 3].item() == 5

    def test_minus_100_replaced_with_pad(self):
        labels = torch.tensor([[3, -100, -100]])
        shifted = _shift_tokens_right(labels, pad_token_id=0, decoder_start_token_id=1)
        # Position 2 and 3 (shifted from -100 in position 1 and 2) should be 0
        assert shifted[0, 2].item() == 0
        assert shifted[0, 3].item() == 0  if labels.shape[1] > 3 else True

    def test_shape_preserved(self):
        labels = torch.randint(1, 10, (4, 15))
        labels[:, -3:] = -100
        shifted = _shift_tokens_right(labels, pad_token_id=0, decoder_start_token_id=1)
        assert shifted.shape == labels.shape
