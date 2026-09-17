# CURC Validation Results — Post-ICML Submission

Generated: 2026-09-16

This bundle contains the multi-seed validation results produced on CU Boulder's
CURC Alpine cluster since the original ICML submission (which reported only
seed-42, n=200-subset numbers). Everything here uses the full test sets
(CNN/DailyMail 11,490 / XSUM 11,334 / SAMSum 819) unless noted otherwise.

## Contents

### `t5-small/` — T5-small, 5-seed full-test-set validation
Seeds: 42, 1337, 2024, 8888, 12345 (matches the five seeds named in the paper's
experimental protocol). Per seed:
- `evaluation_results.json` — merged ROUGE-1/2/L (Standard/Baseline/Monotonic
  T5) with bootstrap 95% CIs, on all three datasets, plus paired t-tests
  (Bonferroni-corrected) and Cohen's d for Baseline vs. Monotonic.
  `evaluation_results_{cnn_dm,xsum,samsum}.json` are the same data as
  per-dataset shards (an artifact of how the full-test-set eval was split
  across parallel jobs to fit CURC's walltime limits; the merged file is the
  one to use).
- `uat_results.json` — Universal Adversarial Trigger attack results (present
  for seeds 1337/2024/8888/12345; seed 42's run is still in progress).
- `hotflip_results.json` — HotFlip gradient-based attack results (present for
  seed 2024 only; the other seeds' runs are still in progress).
- `order_preservation_results.json` / `order_preservation_depth.png` — the
  per-layer order-preservation probe (all 5 seeds complete).
- `learned_triggers.csv` — the human-readable learned UAT triggers.
- `baseline_training_history.json` / `monotonic_training_history.json` —
  per-epoch train/val loss curves.

Also included: the two attribution-ablation arms at seed 42 —
`seed_42_sign_frozen/` (W = sign(W_pretrained) * softplus(V), tests whether
nonnegativity itself is the causal factor) and `seed_42_abs_init_free/`
(V initialized from |W_pretrained| but trained with no constraint, tests
whether the initialization disruption alone explains the effect). Both just
completed monotonic training (clean, normal convergence); evaluation for
these two has not yet run.

### `pythia-1.4b/` — Pythia-1.4B (decoder-only), 3-seed attack/transfer results
Seeds: 42, 1337, 2024. Per seed:
- `uat_results.json` — UAT attack results (Pile held-out perplexity under
  learned triggers).
- `uat_transfer_results.json` — cross-model UAT trigger transfer matrix
  (baseline-trained vs. monotonic-trained triggers, evaluated against both
  models).
- `hotflip_transfer_results.json` — HotFlip substitution transfer + control
  results.
- `learned_triggers.csv` — human-readable learned UAT triggers.
- `order_preservation_results.json` / `order_preservation_depth.png` — seed 42
  only; the other two seeds' runs are still in progress.

## Explicitly NOT included

- **T5-base / T5-large scale-up.** T5-base's monotonic training hit a bf16
  numerical-precision bug (the softplus reparametrization's initialization
  collapsed under bfloat16, discussed in the accompanying email) that produced
  invalid near-zero-ROUGE results; those were discarded and T5-base is
  currently re-running under a fix. T5-large was deprioritized given the
  submission timeline. Neither is represented here.
- **Pythia-2.8B / 6.9B / Llama-3 scale-up.** Same reasoning — deprioritized
  given the timeline; still in progress, not included.
- Any `partial/` subdirectories on CURC (incremental/resumable optimizer
  scratch state, not final results).

## Provenance

Pulled from `/scratch/alpine/paco0228/mono_s2s_results/` and
`/scratch/alpine/paco0228/foundation_llm_results_seed*/` on CURC Alpine via
`scp` on 2026-09-16. Produced by the `mono-s2s` pipeline
(github.com/PatrickAllenCooper/mono-s2s), commit `3623032` and later.
