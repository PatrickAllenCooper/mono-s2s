"""Pure NumPy probe-fit / Ah <= Ah' math shared by T5 Stage 8 and Pythia Stage 11."""

import hashlib
import numpy as np

NUM_PROBE_DIRECTIONS = 64
ORDER_EPS = 1e-6


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def collect_unique_texts(chains):
    texts = set()
    for chain in chains:
        texts.update(chain["sequence"])
    return sorted(texts)


def fit_logreg_direction(X_pos, X_neg, iters=300, lr=0.5, l2=1e-2, rng=None):
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sigma

    if rng is not None:
        w = rng.normal(scale=0.01, size=X.shape[1])
    else:
        w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iters):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = Xn.T @ (p - y) / n + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b

    direction = w / sigma
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm
    return direction


def fit_probe_directions(fit_pairs, hidden_cache, layer_idx, pooling, num_directions, seed):
    rng = np.random.RandomState(seed)
    n = len(fit_pairs)
    sample_size = max(20, int(0.7 * n))
    directions = []
    for _ in range(num_directions):
        idx = rng.choice(n, size=sample_size, replace=True)
        sampled = [fit_pairs[i] for i in idx]
        X_pos = np.stack([
            np.array(hidden_cache[text_key(p["stronger"])][pooling][layer_idx]) for p in sampled
        ])
        X_neg = np.stack([
            np.array(hidden_cache[text_key(p["weaker"])][pooling][layer_idx]) for p in sampled
        ])
        directions.append(fit_logreg_direction(X_pos, X_neg, rng=rng))
    return np.stack(directions)


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return float(values.mean()), lo, hi


def measure_order_preservation(eval_pairs, hidden_cache, A, pooling, num_layers):
    per_layer_values = {layer: [] for layer in range(num_layers)}
    per_pair_records = []
    for pair in eval_pairs:
        h_weak_layers = hidden_cache[text_key(pair["weaker"])][pooling]
        h_strong_layers = hidden_cache[text_key(pair["stronger"])][pooling]
        record = {"pair_id": pair["pair_id"], "axis": pair["axis"]}
        for layer in range(num_layers):
            s_weak = A @ np.array(h_weak_layers[layer])
            s_strong = A @ np.array(h_strong_layers[layer])
            frac = float(np.mean(s_strong >= s_weak - ORDER_EPS))
            per_layer_values[layer].append(frac)
            record[f"layer_{layer}_frac"] = frac
        per_pair_records.append(record)

    per_layer_summary = {}
    for layer in range(num_layers):
        mean, lo, hi = bootstrap_ci(per_layer_values[layer], seed=layer)
        per_layer_summary[layer] = {"mean": mean, "ci_low": lo, "ci_high": hi}
    return per_layer_summary, per_pair_records
