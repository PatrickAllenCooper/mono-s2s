"""
Tests for make_model_monotonic variant dispatch and sweep_aggregate logic.

Uses a tiny synthetic GPTNeoX-style model so no GPU / HuggingFace download
is required.
"""

import json
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

# Add parent directory so imports work without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.common_utils import make_model_monotonic, MONOTONIC_VARIANT_PATTERNS


# ---------------------------------------------------------------------------
# Minimal synthetic model mimicking GPTNeoX MLP + attention naming
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    """Mirrors GPTNeoX MLP: dense_h_to_4h + act + dense_4h_to_h."""
    def __init__(self, d=8):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(d, d * 4)
        self.dense_4h_to_h = nn.Linear(d * 4, d)

    def forward(self, x):
        return self.dense_4h_to_h(torch.relu(self.dense_h_to_4h(x)))


class TinyAttention(nn.Module):
    """Minimal attention block; 'dense' is the output projection."""
    def __init__(self, d=8):
        super().__init__()
        self.query_key_value = nn.Linear(d, d * 3)
        self.dense = nn.Linear(d, d)

    def forward(self, x):
        return self.dense(x)


class TinyLayer(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.attention = TinyAttention(d)
        self.mlp = TinyMLP(d)

    def forward(self, x):
        return self.mlp(self.attention(x))


class TinyGPTNeoX(nn.Module):
    """Two-layer miniature of the GPTNeoX naming structure."""
    def __init__(self, d=8, n_layers=2):
        super().__init__()
        self.gpt_neox = nn.ModuleDict({
            "layers": nn.ModuleList([TinyLayer(d) for _ in range(n_layers)])
        })

    def forward(self, x):
        for layer in self.gpt_neox["layers"]:
            x = layer(x)
        return x

    def named_modules(self):
        # Replicate the dotted-name structure HuggingFace uses
        yield from super().named_modules()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constrained_names(model):
    """Return set of module names that have a softplus parametrization."""
    import torch.nn.utils.parametrize as P
    names = set()
    for name, module in model.named_modules():
        if P.is_parametrized(module):
            names.add(name)
    return names


def _make_fresh():
    return TinyGPTNeoX(d=8, n_layers=2)


# ---------------------------------------------------------------------------
# Variant: mlp_in
# ---------------------------------------------------------------------------

class TestVariantMlpIn:
    def test_only_dense_h_to_4h_constrained(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in")
        constrained = _constrained_names(model)
        # dense_h_to_4h in both layers
        assert any("dense_h_to_4h" in n for n in constrained)
        # dense_4h_to_h must NOT be constrained
        assert not any("dense_4h_to_h" in n for n in constrained)
        # attention.dense must NOT be constrained
        assert not any(n.endswith("attention.dense") for n in constrained)

    def test_count_matches_num_layers(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in")
        constrained = _constrained_names(model)
        # 2 layers * 1 MLP-in projection = 2
        assert len(constrained) == 2

    def test_weights_non_negative_after_application(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in")
        for name, module in model.named_modules():
            if "dense_h_to_4h" in name and isinstance(module, nn.Linear):
                assert (module.weight >= 0).all(), f"{name} has negative weights"

    def test_forward_pass_unchanged_shape(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in")
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 8)

    def test_env_var_default_selects_mlp_in(self, monkeypatch):
        monkeypatch.delenv("MONOTONIC_VARIANT", raising=False)
        model = _make_fresh()
        model = make_model_monotonic(model)   # no variant kwarg
        constrained = _constrained_names(model)
        assert not any("dense_4h_to_h" in n for n in constrained)


# ---------------------------------------------------------------------------
# Variant: mlp_both
# ---------------------------------------------------------------------------

class TestVariantMlpBoth:
    def test_both_mlp_projections_constrained(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_both")
        constrained = _constrained_names(model)
        assert any("dense_h_to_4h" in n for n in constrained)
        assert any("dense_4h_to_h" in n for n in constrained)

    def test_attention_not_constrained(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_both")
        constrained = _constrained_names(model)
        assert not any(n.endswith("attention.dense") for n in constrained)

    def test_count_matches_num_layers(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_both")
        constrained = _constrained_names(model)
        # 2 layers * 2 MLP projections = 4
        assert len(constrained) == 4

    def test_forward_pass_unchanged_shape(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_both")
        x = torch.randn(2, 8)
        assert model(x).shape == (2, 8)


# ---------------------------------------------------------------------------
# Variant: mlp_in_attn_out
# ---------------------------------------------------------------------------

class TestVariantMlpInAttnOut:
    def test_mlp_in_and_attention_dense_constrained(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in_attn_out")
        constrained = _constrained_names(model)
        assert any("dense_h_to_4h"  in n for n in constrained)
        assert any("attention.dense" in n for n in constrained), \
            f"attention.dense not found in {constrained}"

    def test_mlp_out_not_constrained(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in_attn_out")
        constrained = _constrained_names(model)
        assert not any("dense_4h_to_h" in n for n in constrained)

    def test_count_matches_num_layers(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in_attn_out")
        constrained = _constrained_names(model)
        # 2 layers * (1 MLP-in + 1 attn-out) = 4
        assert len(constrained) == 4

    def test_forward_pass_unchanged_shape(self):
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in_attn_out")
        x = torch.randn(2, 8)
        assert model(x).shape == (2, 8)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_variant_raises(self):
        model = _make_fresh()
        with pytest.raises(ValueError, match="Unknown monotonic variant"):
            make_model_monotonic(model, variant="nonexistent_variant")

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("MONOTONIC_VARIANT", "mlp_both")
        model = _make_fresh()
        model = make_model_monotonic(model)
        constrained = _constrained_names(model)
        # mlp_both should constrain dense_4h_to_h too
        assert any("dense_4h_to_h" in n for n in constrained)

    def test_kwarg_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("MONOTONIC_VARIANT", "mlp_both")
        model = _make_fresh()
        model = make_model_monotonic(model, variant="mlp_in")
        constrained = _constrained_names(model)
        assert not any("dense_4h_to_h" in n for n in constrained)


# ---------------------------------------------------------------------------
# sweep_aggregate decision logic
# ---------------------------------------------------------------------------

class TestSweepAggregateDecision:
    """Test the winner-selection logic in sweep_aggregate.py."""

    def setup_method(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from sweep_aggregate import pick_winner, load_cell
        self.pick_winner = pick_winner

    def _make_row(self, variant, seed, ppl_ratio, sr_drop):
        return {
            "variant": variant, "seed": seed,
            "ppl_ratio": ppl_ratio,
            "hf_sr_drop": sr_drop,
            "hf_sr_baseline": 0.7, "hf_sr_monotonic": 0.7 - sr_drop,
        }

    def test_picks_best_sr_drop_within_ceiling(self):
        rows = [
            self._make_row("mlp_both",         42, 1.8, 0.10),
            self._make_row("mlp_in_attn_out",   42, 1.5, 0.30),
        ]
        winner = self.pick_winner(rows, ppl_ceiling=2.0,
                                  variants=["mlp_both", "mlp_in_attn_out"])
        assert winner == "mlp_in_attn_out"

    def test_rejects_variant_above_ppl_ceiling(self):
        rows = [
            self._make_row("mlp_both",         42, 2.5, 0.40),
            self._make_row("mlp_in_attn_out",   42, 1.5, 0.10),
        ]
        winner = self.pick_winner(rows, ppl_ceiling=2.0,
                                  variants=["mlp_both", "mlp_in_attn_out"])
        assert winner == "mlp_in_attn_out"

    def test_no_winner_when_all_exceed_ceiling(self):
        rows = [
            self._make_row("mlp_both",         42, 3.0, 0.40),
            self._make_row("mlp_in_attn_out",   42, 2.5, 0.30),
        ]
        winner = self.pick_winner(rows, ppl_ceiling=2.0,
                                  variants=["mlp_both", "mlp_in_attn_out"])
        assert winner is None

    def test_no_winner_when_sr_drop_negative(self):
        rows = [self._make_row("mlp_both", 42, 1.5, -0.05)]
        winner = self.pick_winner(rows, ppl_ceiling=2.0, variants=["mlp_both"])
        assert winner is None

    def test_multi_seed_all_must_satisfy_ceiling(self):
        rows = [
            self._make_row("mlp_in_attn_out", 42,   1.5, 0.20),
            self._make_row("mlp_in_attn_out", 1337, 2.5, 0.20),  # violates ceiling
        ]
        winner = self.pick_winner(rows, ppl_ceiling=2.0, variants=["mlp_in_attn_out"])
        assert winner is None

    def test_relaxed_ceiling_rescues_winner(self):
        rows = [
            self._make_row("mlp_in_attn_out", 42,   2.4, 0.20),
            self._make_row("mlp_in_attn_out", 1337, 2.6, 0.20),
        ]
        # Fails at 2.0x
        assert self.pick_winner(rows, 2.0, ["mlp_in_attn_out"]) is None
        # Succeeds at 3.0x
        assert self.pick_winner(rows, 3.0, ["mlp_in_attn_out"]) == "mlp_in_attn_out"
