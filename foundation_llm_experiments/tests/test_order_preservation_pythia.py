"""Pythia Stage 11 reuses T5 probe math (loaded by file path)."""
import importlib.util
from pathlib import Path

import numpy as np

_MATH_PATH = Path(__file__).parent.parent.parent / "hpc_version" / "utils" / "order_preservation_math.py"
_spec = importlib.util.spec_from_file_location("order_preservation_math", _MATH_PATH)
_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)


def test_stage_11_constants_match_t5_instrument():
    assert _math.NUM_PROBE_DIRECTIONS == 64
    assert _math.ORDER_EPS > 0


def test_reused_math_recovers_order_on_synthetic_decoder_states():
    rng = np.random.RandomState(1)
    d = 16
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)

    fit_pairs = []
    hidden = {"_text_count": 0}
    for i in range(40):
        weak = f"w{i}"
        strong = f"s{i}"
        h_w = rng.normal(size=d) * 0.05
        h_s = h_w + 0.5 * direction
        hidden[_math.text_key(weak)] = {"last": [h_w.tolist(), h_w.tolist()]}
        hidden[_math.text_key(strong)] = {"last": [h_s.tolist(), h_s.tolist()]}
        fit_pairs.append({"weaker": weak, "stronger": strong, "pair_id": i, "axis": "x"})

    A = _math.fit_probe_directions(
        fit_pairs, hidden, layer_idx=1, pooling="last", num_directions=8, seed=0
    )
    summary, _ = _math.measure_order_preservation(fit_pairs, hidden, A, "last", 2)
    assert summary[1]["mean"] > 0.8
