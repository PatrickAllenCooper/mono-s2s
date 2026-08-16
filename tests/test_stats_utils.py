"""Unit tests for paired t-tests, Bonferroni, Cohen's d, and bootstrap CIs."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))

from utils.stats_utils import (
    paired_t_test,
    bonferroni_correct,
    bootstrap_mean_ci,
    per_example_paired_tests,
    extract_metric_vectors,
)


class TestPairedTTest:
    def test_identical_samples_have_zero_effect(self):
        x = [1.0, 2.0, 3.0, 4.0]
        result = paired_t_test(x, x)
        assert result["n"] == 4
        assert result["cohens_d"] == pytest.approx(0.0, abs=1e-12)
        # Zero-variance differences make the t statistic undefined.
        assert np.isnan(result["t_stat"]) or result["t_stat"] == pytest.approx(0.0)

    def test_systematic_shift_is_detected(self):
        rng = np.random.RandomState(0)
        a = rng.normal(size=30)
        b = a + 0.4 + 0.05 * rng.normal(size=30)
        result = paired_t_test(a, b)
        assert result["p_value"] < 0.01
        assert result["cohens_d"] < 0
        assert result["n"] == 30

    def test_too_few_points_returns_nan(self):
        result = paired_t_test([1.0], [2.0])
        assert np.isnan(result["p_value"])
        assert result["n"] == 1


class TestBonferroni:
    def test_multiplies_by_family_size_and_clips_at_one(self):
        tests = {
            "a": {"p_value": 0.04, "t_stat": 1.0},
            "b": {"p_value": 0.80, "t_stat": 0.1},
        }
        corrected = bonferroni_correct(tests)
        assert corrected["a"]["p_value_bonferroni"] == pytest.approx(0.08)
        assert corrected["b"]["p_value_bonferroni"] == pytest.approx(1.0)
        assert corrected["a"]["bonferroni_m"] == 2

    def test_nan_p_stays_nan(self):
        corrected = bonferroni_correct({"x": {"p_value": float("nan")}})
        assert np.isnan(corrected["x"]["p_value_bonferroni"])


class TestBootstrapCI:
    def test_interval_covers_mean_of_symmetric_sample(self):
        rng = np.random.RandomState(0)
        values = rng.normal(loc=10.0, scale=1.0, size=200)
        ci = bootstrap_mean_ci(values, n_bootstrap=500, seed=0)
        assert ci["lower"] < ci["mean"] < ci["upper"]
        assert ci["n"] == 200
        assert abs(ci["mean"] - 10.0) < 0.3

    def test_empty_input(self):
        ci = bootstrap_mean_ci([])
        assert np.isnan(ci["mean"])
        assert ci["n"] == 0


class TestPerExampleHelpers:
    def test_extract_and_paired_family(self):
        rows_a = [
            {"rouge1": 0.40, "rouge2": 0.20},
            {"rouge1": 0.55, "rouge2": 0.31},
            {"rouge1": 0.48, "rouge2": 0.22},
            {"rouge1": 0.61, "rouge2": 0.35},
        ]
        rows_b = [
            {"rouge1": 0.31, "rouge2": 0.11},
            {"rouge1": 0.44, "rouge2": 0.19},
            {"rouge1": 0.39, "rouge2": 0.18},
            {"rouge1": 0.50, "rouge2": 0.27},
        ]
        vec_a = extract_metric_vectors(rows_a, ["rouge1", "rouge2"])
        vec_b = extract_metric_vectors(rows_b, ["rouge1", "rouge2"])
        tests = per_example_paired_tests(vec_a, vec_b)
        assert set(tests) == {"rouge1", "rouge2"}
        assert tests["rouge1"]["bonferroni_m"] == 2
        assert tests["rouge1"]["p_value_bonferroni"] >= tests["rouge1"]["p_value"]
