"""
Statistical helpers claimed in the paper appendix.

Paired t-tests on matched examples or seed-level means, Cohen's d for
paired samples, Bonferroni correction across a family of tests, and
percentile bootstrap confidence intervals. These functions are pure
NumPy / SciPy so they can be unit-tested without GPU or model weights.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def paired_t_test(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, float]:
    """Paired t-test and paired Cohen's d for two aligned samples."""
    arr_a = np.asarray(values_a, dtype=float)
    arr_b = np.asarray(values_b, dtype=float)
    if arr_a.shape != arr_b.shape or arr_a.size < 2:
        return {
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "cohens_d": float("nan"),
            "n": int(arr_a.size),
        }

    t_stat, p_value = stats.ttest_rel(arr_a, arr_b)
    diff = arr_a - arr_b
    denom = np.std(diff, ddof=1)
    cohens_d = float(np.mean(diff) / denom) if denom > 0 else 0.0
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "n": int(arr_a.size),
    }


def bonferroni_correct(
    tests: Dict[str, Dict[str, float]],
    p_key: str = "p_value",
) -> Dict[str, Dict[str, float]]:
    """Multiply each p-value by the number of tests (Bonferroni)."""
    n = len(tests)
    corrected = {}
    for name, result in tests.items():
        updated = dict(result)
        p = result.get(p_key, float("nan"))
        if p is None or (isinstance(p, float) and np.isnan(p)):
            updated["p_value_bonferroni"] = float("nan")
        else:
            updated["p_value_bonferroni"] = float(min(1.0, p * n))
        updated["bonferroni_m"] = n
        corrected[name] = updated
    return corrected


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}

    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "n": 1}

    rng = np.random.RandomState(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boot[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    alpha = 1.0 - confidence
    return {
        "mean": mean,
        "lower": float(np.percentile(boot, alpha / 2 * 100)),
        "upper": float(np.percentile(boot, (1 - alpha / 2) * 100)),
        "n": int(arr.size),
    }


def per_example_paired_tests(
    scores_a: Dict[str, Sequence[float]],
    scores_b: Dict[str, Sequence[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Paired t-tests on per-example metric vectors, Bonferroni-corrected
    across the shared metric names.
    """
    tests = {}
    for metric in scores_a:
        if metric not in scores_b:
            continue
        tests[metric] = paired_t_test(scores_a[metric], scores_b[metric])
    return bonferroni_correct(tests)


def extract_metric_vectors(
    all_scores: List[Dict[str, float]],
    metrics: Sequence[str],
) -> Dict[str, List[float]]:
    """Turn a list of per-example score dicts into aligned metric vectors."""
    vectors = {m: [] for m in metrics}
    for row in all_scores:
        for m in metrics:
            if m in row:
                vectors[m].append(float(row[m]))
    return vectors
