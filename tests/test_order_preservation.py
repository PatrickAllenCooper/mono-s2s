"""
Tests for the order-preservation measurement math in
hpc_version/scripts/stage_8_order_preservation.py.

These functions are pure numpy (no model, no GPU), so they are exercised
directly against synthetic hidden states rather than through a real T5
encoder. GPU/model-dependent parts of stage_8 (extraction, main()) are
covered separately by CURC integration runs; this file targets the
statistics that determine what "order preservation" means (Ah <= Ah').
"""
import sys
import os
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version" / "scripts"))

from stage_8_order_preservation import (
    _text_key,
    _collect_unique_texts,
    _fit_logreg_direction,
    fit_probe_directions,
    _bootstrap_ci,
    measure_order_preservation,
    NUM_PROBE_DIRECTIONS,
    ORDER_EPS,
)


class TestTextKey:
    """Hashing helper used to key the hidden-state cache by sentence text."""

    def test_deterministic(self):
        assert _text_key("hello world") == _text_key("hello world")

    def test_distinguishes_different_texts(self):
        assert _text_key("hello world") != _text_key("hello, world")

    def test_returns_fixed_length_hex_string(self):
        key = _text_key("some arbitrary sentence.")
        assert len(key) == 24
        int(key, 16)  # should not raise: must be valid hex


class TestCollectUniqueTexts:
    def test_deduplicates_across_chains(self):
        chains = [
            {"sequence": ["a", "b", "c"]},
            {"sequence": ["b", "c", "d"]},
        ]
        texts = _collect_unique_texts(chains)
        assert texts == sorted({"a", "b", "c", "d"})

    def test_empty_chains(self):
        assert _collect_unique_texts([]) == []


class TestFitLogregDirection:
    """The minimal dependency-free logistic-regression probe fitter."""

    def test_recovers_separating_direction_on_linearly_separable_data(self):
        rng = np.random.RandomState(0)
        d = 8
        true_w = rng.normal(size=d)
        true_w /= np.linalg.norm(true_w)

        n = 200
        X_pos = rng.normal(size=(n, d)) * 0.1 + 2.0 * true_w
        X_neg = rng.normal(size=(n, d)) * 0.1 - 2.0 * true_w

        direction = _fit_logreg_direction(X_pos, X_neg, iters=500, lr=0.5, rng=rng)

        assert direction.shape == (d,)
        assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)
        # The fitted direction should score positives higher than negatives
        # on essentially every example, since the classes are well separated.
        scores_pos = X_pos @ direction
        scores_neg = X_neg @ direction
        assert np.mean(scores_pos.mean() > scores_neg.mean()) == 1.0
        separation_accuracy = np.mean(
            [s > np.median(np.concatenate([scores_pos, scores_neg])) for s in scores_pos]
        )
        assert separation_accuracy > 0.85

    def test_direction_is_unit_norm(self):
        rng = np.random.RandomState(1)
        X_pos = rng.normal(size=(30, 5)) + 1.0
        X_neg = rng.normal(size=(30, 5)) - 1.0
        direction = _fit_logreg_direction(X_pos, X_neg, rng=rng)
        assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)


class TestFitProbeDirections:
    def test_returns_requested_number_of_directions(self):
        rng = np.random.RandomState(2)
        d = 4
        num_layers = 3
        pooling = "mean"

        fit_pairs = []
        hidden_cache = {}
        for i in range(20):
            weaker_text = f"weak_{i}"
            stronger_text = f"strong_{i}"
            base = rng.normal(size=d)
            hidden_cache[_text_key(weaker_text)] = {
                pooling: [(base - 1.0).tolist() for _ in range(num_layers)]
            }
            hidden_cache[_text_key(stronger_text)] = {
                pooling: [(base + 1.0).tolist() for _ in range(num_layers)]
            }
            fit_pairs.append({"weaker": weaker_text, "stronger": stronger_text})

        directions = fit_probe_directions(
            fit_pairs, hidden_cache, layer_idx=num_layers - 1, pooling=pooling,
            num_directions=8, seed=42,
        )
        assert directions.shape == (8, d)

    def test_deterministic_given_seed(self):
        rng = np.random.RandomState(3)
        d = 4
        fit_pairs, hidden_cache = _make_synthetic_fit_pairs(rng, d, n=15, num_layers=2)

        d1 = fit_probe_directions(fit_pairs, hidden_cache, 1, "mean", 6, seed=7)
        d2 = fit_probe_directions(fit_pairs, hidden_cache, 1, "mean", 6, seed=7)
        np.testing.assert_allclose(d1, d2)


def _make_synthetic_fit_pairs(rng, d, n, num_layers, pooling="mean"):
    fit_pairs = []
    hidden_cache = {}
    for i in range(n):
        weaker_text = f"weak_{i}"
        stronger_text = f"strong_{i}"
        base = rng.normal(size=d)
        hidden_cache[_text_key(weaker_text)] = {
            pooling: [(base - 1.0).tolist() for _ in range(num_layers)]
        }
        hidden_cache[_text_key(stronger_text)] = {
            pooling: [(base + 1.0).tolist() for _ in range(num_layers)]
        }
        fit_pairs.append({"weaker": weaker_text, "stronger": stronger_text})
    return fit_pairs, hidden_cache


class TestBootstrapCI:
    def test_ci_brackets_the_mean(self):
        values = [0.9, 0.95, 1.0, 0.85, 0.92, 0.88]
        mean, lo, hi = _bootstrap_ci(values, n_boot=500, seed=0)
        assert lo <= mean <= hi

    def test_empty_values_returns_zeros(self):
        assert _bootstrap_ci([]) == (0.0, 0.0, 0.0)

    def test_constant_values_gives_zero_width_ci(self):
        mean, lo, hi = _bootstrap_ci([0.7] * 10, n_boot=200, seed=0)
        assert mean == pytest.approx(0.7)
        assert lo == pytest.approx(0.7)
        assert hi == pytest.approx(0.7)


class TestMeasureOrderPreservation:
    """The core Ah <= Ah' computation the paper's title refers to."""

    def test_perfect_order_preservation_gives_fraction_one(self):
        """If every probe coordinate scores the stronger sentence higher at
        every layer, order preservation must be measured as 1.0 everywhere."""
        num_layers = 3
        p = 4  # probe dimension
        A = np.eye(p)

        eval_pairs = [{"pair_id": 0, "axis": "severity", "weaker": "w", "stronger": "s"}]
        hidden_cache = {
            _text_key("w"): {"mean": [[0.0] * p] * num_layers},
            _text_key("s"): {"mean": [[1.0] * p] * num_layers},
        }

        per_layer_summary, per_pair_records = measure_order_preservation(
            eval_pairs, hidden_cache, A, "mean", num_layers,
        )
        for layer in range(num_layers):
            assert per_layer_summary[layer]["mean"] == pytest.approx(1.0)
        assert per_pair_records[0]["layer_0_frac"] == pytest.approx(1.0)

    def test_total_order_violation_gives_fraction_zero(self):
        """If the stronger sentence scores strictly lower everywhere, the
        fraction of preserved coordinates must be 0."""
        num_layers = 2
        p = 4
        A = np.eye(p)

        eval_pairs = [{"pair_id": 0, "axis": "severity", "weaker": "w", "stronger": "s"}]
        hidden_cache = {
            _text_key("w"): {"mean": [[1.0] * p] * num_layers},
            _text_key("s"): {"mean": [[0.0] * p] * num_layers},
        }

        per_layer_summary, _ = measure_order_preservation(
            eval_pairs, hidden_cache, A, "mean", num_layers,
        )
        for layer in range(num_layers):
            assert per_layer_summary[layer]["mean"] == pytest.approx(0.0)

    def test_partial_order_preservation(self):
        """Half the probe coordinates preserved should measure ~0.5."""
        num_layers = 1
        A = np.eye(4)
        eval_pairs = [{"pair_id": 0, "axis": "x", "weaker": "w", "stronger": "s"}]
        hidden_cache = {
            _text_key("w"): {"mean": [[0.0, 0.0, 1.0, 1.0]]},
            # Coordinates 0,1 increase (preserved); coordinates 2,3 decrease (violated).
            _text_key("s"): {"mean": [[1.0, 1.0, 0.0, 0.0]]},
        }
        per_layer_summary, _ = measure_order_preservation(
            eval_pairs, hidden_cache, A, "mean", num_layers,
        )
        assert per_layer_summary[0]["mean"] == pytest.approx(0.5)

    def test_order_eps_tolerates_numerical_ties(self):
        """Coordinates that are equal (within ORDER_EPS) should count as
        preserved, not as violations, to avoid float-noise false negatives."""
        num_layers = 1
        A = np.eye(2)
        eval_pairs = [{"pair_id": 0, "axis": "x", "weaker": "w", "stronger": "s"}]
        tie_value = 1.0
        hidden_cache = {
            _text_key("w"): {"mean": [[tie_value, tie_value]]},
            _text_key("s"): {"mean": [[tie_value - ORDER_EPS / 2, tie_value]]},
        }
        per_layer_summary, _ = measure_order_preservation(
            eval_pairs, hidden_cache, A, "mean", num_layers,
        )
        assert per_layer_summary[0]["mean"] == pytest.approx(1.0)

    def test_reports_ci_bounds_per_layer(self):
        num_layers = 2
        A = np.eye(3)
        rng = np.random.RandomState(0)
        eval_pairs = []
        hidden_cache = {}
        for i in range(30):
            w, s = f"w{i}", f"s{i}"
            base = rng.normal(size=3)
            hidden_cache[_text_key(w)] = {"mean": [base.tolist()] * num_layers}
            # Randomly preserve or violate order per example, per coordinate.
            delta = rng.choice([-1.0, 1.0], size=3)
            hidden_cache[_text_key(s)] = {"mean": [(base + delta).tolist()] * num_layers}
            eval_pairs.append({"pair_id": i, "axis": "x", "weaker": w, "stronger": s})

        per_layer_summary, per_pair_records = measure_order_preservation(
            eval_pairs, hidden_cache, A, "mean", num_layers,
        )
        assert len(per_pair_records) == 30
        for layer in range(num_layers):
            summary = per_layer_summary[layer]
            assert 0.0 <= summary["ci_low"] <= summary["mean"] <= summary["ci_high"] <= 1.0
