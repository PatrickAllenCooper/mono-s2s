"""Gated-FFN monotonic variants on a tiny LlamaConfig model (no download)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.common_utils import make_model_monotonic, MONOTONIC_VARIANT_PATTERNS


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
    )
    return LlamaForCausalLM(config)


def _constrained_names(model):
    names = set()
    for name, module in model.named_modules():
        if P.is_parametrized(module):
            names.add(name)
    return names


def test_gated_patterns_are_registered():
    assert "gated_updown" in MONOTONIC_VARIANT_PATTERNS
    assert "gated_all" in MONOTONIC_VARIANT_PATTERNS
    assert MONOTONIC_VARIANT_PATTERNS["gated_updown"] == ["up_proj", "down_proj"]
    assert MONOTONIC_VARIANT_PATTERNS["gated_all"] == [
        "gate_proj", "up_proj", "down_proj",
    ]


def test_gated_updown_constrains_up_and_down_only():
    model = _tiny_llama()
    model = make_model_monotonic(model, variant="gated_updown")
    constrained = _constrained_names(model)
    assert any("up_proj" in n for n in constrained)
    assert any("down_proj" in n for n in constrained)
    assert not any("gate_proj" in n for n in constrained)
    assert not any("o_proj" in n for n in constrained)
    # 2 layers * (up + down)
    assert len(constrained) == 4


def test_gated_all_constrains_three_projections():
    model = _tiny_llama()
    model = make_model_monotonic(model, variant="gated_all")
    constrained = _constrained_names(model)
    assert any("gate_proj" in n for n in constrained)
    assert any("up_proj" in n for n in constrained)
    assert any("down_proj" in n for n in constrained)
    assert not any("o_proj" in n for n in constrained)
    # 2 layers * (gate + up + down)
    assert len(constrained) == 6


def test_gated_updown_weights_nonnegative():
    model = _tiny_llama()
    model = make_model_monotonic(model, variant="gated_updown")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "up_proj" in name or "down_proj" in name:
            assert (module.weight >= 0).all(), f"{name} has negative weights"
        if "gate_proj" in name:
            assert (module.weight < 0).any(), f"{name} should remain unconstrained"


def test_gated_updown_forward_pass():
    model = _tiny_llama()
    model = make_model_monotonic(model, variant="gated_updown")
    model.eval()
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        out = model(input_ids)
    assert out.logits.shape == (2, 8, 32)


def test_mlp_both_does_not_match_llama():
    model = _tiny_llama()
    with pytest.raises(RuntimeError, match="No layers matched"):
        make_model_monotonic(model, variant="mlp_both")
