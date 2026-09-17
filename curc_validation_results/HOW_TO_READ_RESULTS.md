# How to Read These Results

This is a field-by-field guide to every JSON schema in this bundle, with real
examples pulled from the data itself. See `README.md` for what's in each
folder; this file is about how to interpret what's inside each file.

Three models appear throughout: **Standard** (pretrained, not fine-tuned),
**Baseline** (fine-tuned, unconstrained FFNs — the fair control), and
**Monotonic** (fine-tuned, FFN weights constrained to W ≥ 0 via the softplus
reparametrization). For Pythia there is no "Standard" arm, only
**baseline_pythia** / **monotonic_pythia**.

---

## `evaluation_results.json` (T5 — clean ROUGE performance)

```json
"cnn_dm": {
  "baseline_t5": {
    "rouge_scores": {
      "rouge1": {"mean": 0.3993, "lower": 0.3971, "upper": 0.4015},
      ...
    },
    "prediction_length_stats": {...},
    "brevity_penalty": {...}
  },
  "monotonic_t5": {...},
  "standard_t5": {...},
  "paired_tests_baseline_vs_monotonic": {
    "rouge1": {"t_stat": 25.25, "p_value": 5.7e-137, "p_value_bonferroni": 1.7e-136, "cohens_d": 0.236}
  }
}
```
- `mean`/`lower`/`upper` is the ROUGE score with its **95% bootstrap CI**
  (1,000 resamples) — read `lower`/`upper` as "we're 95% confident the true
  score is in this range," not a single point estimate.
- `paired_tests_baseline_vs_monotonic` is a **paired** t-test (same examples,
  both models) with **Bonferroni correction** across the three ROUGE variants
  and Cohen's *d* for effect size. Small `p_value_bonferroni` + small `d`
  (as seen throughout the T5-small results, d≈0.05-0.24) means the gap is
  statistically real but practically modest — that's the expected, reported
  trade-off, not a red flag.
- `evaluation_results_{cnn_dm,xsum,samsum}.json` are per-dataset shards of the
  exact same computation (an artifact of how the full test set was split
  across parallel CURC jobs). Use the merged `evaluation_results.json`.
- `_metadata`/`metadata` carry the run's timestamp, seed, and decoding config
  — useful for provenance, not for analysis.

## `uat_results.json` (T5 — Universal Adversarial Trigger attack)

```json
"results": {
  "monotonic_t5": {
    "trigger_text": "recherche» shift flexible travail",
    "rouge_deltas": {"rouge1": 0.0026, "rouge2": 0.0023, "rougeLsum": 0.0018},
    "nll_increase": 0.0038,
    "clean_rouge": {...}, "attack_rouge": {...}
  }
}
```
- Each model gets **its own trigger**, optimized against it specifically
  (coordinate-ascent search, see `attack_config` for the search budget).
- `rouge_deltas` = attack_rouge − clean_rouge (small/negative = the trigger
  barely hurt performance; UAT is a weak attack against T5 overall — the
  paper notes this explicitly and relies on HotFlip for the stronger effect).
- `nll_increase` is the increase in negative log-likelihood under the
  trigger — a smaller number here means more robust.
- `transfer_matrix[source][target]` = ROUGE-L delta when the trigger learned
  on `source` is replayed against `target`. **The diagonal is the
  same-model attack** (matches `rouge_deltas.rougeLsum` above for that
  model); off-diagonal cells test whether a trigger generalizes across
  models.
- `learned_triggers.csv` is the same `learned_triggers`/`trigger_ids` data,
  just flattened to one row per model for quick reading — the trigger text
  is nonsense by design (it's optimized for loss, not fluency).

## `hotflip_results.json` (T5 — gradient-based token-flip attack)

```json
"results": {
  "monotonic_t5": {
    "avg_degradation": 0.0491, "std_degradation": 0.078,
    "success_rate": 0.17, "avg_orig_loss": 3.39, "avg_attack_loss": 3.53
  }
},
"statistical_tests": {
  "baseline_vs_monotonic": {"t_stat": 9.04, "p_value": 6.9e-18, "significant": "***"}
}
```
- `avg_degradation` is the mean **relative** loss increase under attack
  (0.049 = +4.9%). This is the pipeline's main robustness signal — HotFlip is
  a much stronger attack than UAT, and this is where the monotonicity effect
  is large (seed 2024: standard 19.9% → baseline 15.1% → monotonic 4.9%).
- `success_rate` = fraction of examples where degradation exceeded the 10%
  threshold (an "attack succeeded" call, not a continuous score).
- `statistical_tests` are **unpaired** t-tests between models' degradation
  distributions; `significant` is a conventional star rating (`***` = p<0.001).

## `order_preservation_results.json` (T5 and Pythia)

```json
"per_layer_order_preservation": {
  "monotonic_t5": {
    "3": {"mean": 0.452, "ci_low": 0.440, "ci_high": 0.463}
  },
  "baseline_t5_last_token": {...}
}
```
- This tests a *semantic* claim distinct from the FFN-level nonnegativity
  constraint: for pairs of prompts that progressively strengthen along some
  axis (severity, certainty, etc. — see `config.num_fit_pairs`/
  `num_eval_pairs`), does the *hidden representation* preserve that order
  along a linear probe direction, layer by layer?
- Each layer key (`"0"`..`"N"`) is that encoder/decoder-residual-stream
  layer's **fraction of held-out pairs** (out of `num_eval_pairs`) that
  satisfy the order relation, with a bootstrap CI. Closer to 1.0 = more
  order-preserving at that depth. Compare `monotonic_t5` vs `baseline_t5` at
  the same layer index to see whether the FFN constraint changes this.
- The `_last_token` variants use last-token pooling instead of mean pooling
  as a robustness check on the same underlying measurement.
- `order_preservation_depth.png` plots exactly this (mean ± CI vs. layer
  depth, one line per model) — open it directly for the fastest read.
- `hotflip_disruption_correlation` (present but empty in this batch) would
  cross-reference stage_6b's persisted HotFlip substitutions against this
  probe, when that optional cross-check is run.

## Pythia `uat_results.json` / `uat_transfer_results.json`

Same idea as T5's UAT, but Pythia is a **language model, not
summarization** — so the metric is held-out Pile **perplexity/NLL**, not
ROUGE:
```json
"baseline_pythia": {
  "clean_loss": 2.007, "attacked_loss": 2.318,
  "nll_increase": 0.155, "nll_increase_percent": 15.46
}
```
`nll_increase_percent` is the headline number — lower is more robust.
`uat_transfer_results.json`'s `transfer_matrix[crafted_on][evaluated_on]`
works exactly like T5's UAT transfer matrix above, just in NLL-increase-%
units instead of ROUGE-L-delta.

## Pythia `hotflip_transfer_results.json`

```json
"substitution_transfer": {
  "monotonic_pythia": {          // crafted_on
    "baseline_pythia": {...},    // evaluated_on: does monotonic's attack transfer TO baseline?
    "monotonic_pythia": {...}    // evaluated_on: same-model (the "real" HotFlip result)
  }
},
"random_control": {...},
"query_attack": {...}
```
- `substitution_transfer[crafted_on][evaluated_on]`: the persisted token
  substitutions from a HotFlip attack crafted against one model, replayed as
  fixed input against the other. `crafted_on == evaluated_on` is the direct
  (non-transfer) HotFlip result, comparable to T5's `hotflip_results.json`.
- `random_control`: the same number of token flips at **random** positions
  (no gradient, no optimization) — a sanity floor. If a model's real attack
  `avg_degradation` isn't well above its random-control degradation, the
  "attack" isn't doing much beyond noise.
- `query_attack`: a gradient-free greedy search scored directly against the
  target model's own loss — rules out "gradient masking" as the explanation
  for any robustness gap (a model could look robust to gradient-based
  HotFlip while still being vulnerable to a attack that doesn't need
  gradients at all; this control checks for that).

## Training histories (`baseline_training_history.json` / `monotonic_training_history.json`)

```json
{"train_losses": [...], "val_losses": [...], "best_val_loss": ..., "num_epochs": 7}
```
One entry per epoch. **What to sanity-check first on any new run**: does the
loss decrease smoothly and land somewhere reasonable (T5-small monotonic:
~2.2-2.6; T5-base baseline: ~1.5-2 range), or does it drop sharply once and
then flatline? A flatline after epoch 1 at a value much higher than the
unconstrained baseline (we saw ~7.5 on T5-base before the bf16 fix, against
baseline's normal ~1.5-2) is the signature of the softplus-reparametrization
precision collapse described in `README.md` — a dead giveaway something's
wrong with that specific run, not a subtle numeric detail to skip past.

## `data_statistics.json` / `setup_complete.json`

Bookkeeping: exact train/val/test split sizes and environment/checkpoint
provenance for that run. Useful for the paper's reproducibility section, not
for any actual analysis.
