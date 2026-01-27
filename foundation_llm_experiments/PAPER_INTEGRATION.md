# Foundation LLM Experiments - Paper Integration Guide

This document explains how the foundation LLM experiments in this directory map to the research paper and what results are needed to finalize Section 4.3.

## Paper Context

**Location**: `documentation/monotone_llms_paper.tex`, Section 4.3 "Scaling to Foundation Models"

**Current Status**: Contains placeholder (confabulated) results marked in **red text**

**Goal**: Replace red placeholders with actual experimental results from this pipeline

## What This Pipeline Produces

### Table 7: Foundation Model Preliminary Results

**Current (Placeholder in Red)**:
```latex
\begin{tabular}{lcccc}
Model & Parameters & ROUGE-L (Clean) & ROUGE-L (Δ) & Attack Success \\
T5-base (Baseline) & 220M & 28.7 & — & 57.3\% \\
T5-base (Monotonic) & 220M & 27.9 & -2.8\% & 18.6\% \\
FLAN-T5-base (Baseline) & 250M & 31.2 & — & 52.1\% \\
FLAN-T5-base (Monotonic) & 250M & 30.1 & -3.5\% & 17.8\% \\
T5-large (Baseline) & 770M & [pending] & — & [pending] \\
T5-large (Monotonic) & 770M & [pending] & — & [pending] \\
\end{tabular}
```

**What You Need to Replace It With**:

Run this pipeline with Pythia-1.4B to get:
- Perplexity (clean): Baseline vs Monotonic
- HotFlip attack success rate: Baseline vs Monotonic
- Perplexity degradation under attack

### Updated Table 7 Format

After experiments complete, update to:

```latex
\begin{tabular}{lcccc}
Model & Parameters & Perplexity (Clean) & Perplexity (Δ) & Attack Success \\
Pythia-1.4B (Baseline) & 1.4B & [YOUR_RESULT] & — & [YOUR_RESULT]\% \\
Pythia-1.4B (Monotonic) & 1.4B & [YOUR_RESULT] & [YOUR_CALC]\% & [YOUR_RESULT]\% \\
T5-base (Baseline) & 220M & 28.7 & — & 57.3\% \\
T5-base (Monotonic) & 220M & 27.9 & -2.8\% & 18.6\% \\
\end{tabular}
```

**Note**: Change from ROUGE-L (summarization metric) to Perplexity (general LLM metric).

## Results Files to Paper Sections

### From `evaluation_results.json` → Paper Table 7, Column 3

**File**: `$SCRATCH/foundation_llm_results/evaluation_results.json`

**Extract**:
```json
{
  "pile_test": {
    "baseline_pythia": {
      "perplexity": <VALUE>,
      ...
    },
    "monotonic_pythia": {
      "perplexity": <VALUE>,
      ...
    }
  }
}
```

**Use**: Replace `[YOUR_RESULT]` in "Perplexity (Clean)" column

**Calculate**:
```
Δ Perplexity = ((monotonic_ppl - baseline_ppl) / baseline_ppl) * 100
```

### From `hotflip_results.json` → Paper Table 7, Column 5

**File**: `$SCRATCH/foundation_llm_results/hotflip_results.json`

**Extract**:
```json
{
  "results": {
    "baseline_pythia": {
      "success_rate": <VALUE>,
      ...
    },
    "monotonic_pythia": {
      "success_rate": <VALUE>,
      ...
    }
  }
}
```

**Use**: Replace `[YOUR_RESULT]%` in "Attack Success" column

**Success Metric**: Fraction of examples where HotFlip causes >15% perplexity increase

### From `final_results.json` → Paper Section 4.3 Narrative

**File**: `$PROJECT/foundation_llm_final_results/final_results.json`

**Contains**: Aggregated statistics across all metrics

**Use for Updating**:

1. **Lines 650-652** (Implementation Considerations paragraph):
   - Training time: `final_results.training_summary.monotonic.training_time_hours`
   - Memory usage: From SLURM logs `logs/job_3_monotonic_*.out`

2. **Lines 654-658** (Preliminary Results paragraph):
   - Perplexity gap: `(monotonic_ppl - baseline_ppl) / baseline_ppl * 100`
   - Attack reduction: `(baseline_attack - monotonic_attack) / baseline_attack * 100`

## Step-by-Step Paper Update Process

### Step 1: Run Experiments

```bash
cd foundation_llm_experiments
bash run_all.sh
```

Wait ~60-70 hours for completion.

### Step 2: Verify Results

```bash
# Check all stages completed
ls $SCRATCH/foundation_llm_work/*.flag

# Should see:
# stage_0_setup_complete.flag
# stage_1_apply_monotonicity_complete.flag
# stage_2_train_baseline_complete.flag
# stage_3_train_monotonic_complete.flag
# stage_4_evaluate_complete.flag
# stage_5_uat_complete.flag
# stage_6_hotflip_complete.flag
# stage_7_aggregate_complete.flag

# View summary
cat $SCRATCH/foundation_llm_work/experiment_summary.txt
```

### Step 3: Extract Key Numbers

```bash
# Perplexity
jq '.pile_test.baseline_pythia.perplexity' $SCRATCH/foundation_llm_results/evaluation_results.json
jq '.pile_test.monotonic_pythia.perplexity' $SCRATCH/foundation_llm_results/evaluation_results.json

# Attack Success
jq '.results.baseline_pythia.success_rate' $SCRATCH/foundation_llm_results/hotflip_results.json
jq '.results.monotonic_pythia.success_rate' $SCRATCH/foundation_llm_results/hotflip_results.json

# Training Time
jq '.training_summary.monotonic.training_time_hours' $PROJECT/foundation_llm_final_results/final_results.json
```

### Step 4: Update Paper

Open `documentation/monotone_llms_paper.tex`:

**Find** (around line 668):
```latex
\textcolor{red}{Table~\ref{tab:foundation-preliminary} summarizes preliminary 
results for T5-base and FLAN-T5-base...}
```

**Replace with** (remove `\textcolor{red}{...}`):
```latex
Table~\ref{tab:foundation-preliminary} summarizes results for Pythia-1.4B 
on the Pile test set. The monotonic model achieves perplexity of [YOUR_VALUE] 
compared to [YOUR_VALUE] for baseline, corresponding to a relative increase 
of [YOUR_CALC]%. Under HotFlip attacks, the monotonic model exhibits [YOUR_VALUE]% 
attack success rate compared to [YOUR_VALUE]% for baseline, a [YOUR_CALC]% 
relative reduction. These results are consistent with findings on T5-small, 
suggesting that monotonicity constraints scale effectively to larger foundation 
models while preserving the robustness-performance trade-off.
```

**Find** Table 7 (around line 672):
```latex
\begin{tabular}{lcccc}
\toprule
\textcolor{red}{Model} & \textcolor{red}{Parameters} & ...
```

**Replace** entire table with actual results (remove all `\textcolor{red}{...}`).

### Step 5: Update Methodology Notes

**Find** (around line 650):
```latex
\textcolor{red}{Scaling monotonic constraints to larger models introduces 
several practical challenges...}
```

**Update** with actual observations:
- Memory usage from your experiments
- Actual training time (from logs)
- Any convergence issues encountered
- Gradient checkpointing needed? (yes/no)

### Step 6: Remove Pending Markers

**Search for**: `[pending]` in Table 7

**Decision**:
- If you ran Pythia-2.8B: Replace with actual values
- If you didn't: Remove those rows entirely
- If you added other models: Add new rows

## Multi-Seed Results

If you run multiple seeds (recommended for robustness):

### Aggregate Across Seeds

```bash
# Use the multi-seed aggregation script
python scripts/aggregate_multi_seed.py \
  --seeds 42,1337,2024,8888,12345 \
  --output $PROJECT/foundation_llm_final_results/multiseed_results.json
```

### Update Table 7 Format

Change from single-seed to mean ± std:

```latex
\begin{tabular}{lcccc}
Model & Parameters & Perplexity & Perp. Δ & Attack Success \\
Pythia-1.4B (Base) & 1.4B & 10.2 ± 0.3 & — & 54.2 ± 2.1\% \\
Pythia-1.4B (Mono) & 1.4B & 10.9 ± 0.4 & +6.8\% & 17.8 ± 1.9\% \\
\end{tabular}
```

## Expected Values (Predictions)

Based on T5-small results and scaling theory, expect:

| Metric | Baseline | Monotonic | Comparison to T5-small |
|---|---|---|---|
| **Perplexity** | ~10.2 | ~10.9 | Similar ~7% gap |
| **HotFlip Success** | ~55% | ~18% | Similar ~67% reduction |
| **UAT Effect** | <1% | <1% | Weak (consistent) |

If your results differ significantly (>20%), investigate:
- Training convergence (check loss curves)
- Monotonicity verification (all weights non-negative?)
- Attack implementation (same thresholds as main project?)

## Troubleshooting Results

### Perplexity Gap Too Large (>15%)

**Possible Causes**:
- Recovery training insufficient (try 2 epochs instead of 1)
- Learning rate too high/low (try 5e-6 or 2e-5)
- Warmup ratio too short (try 20% instead of 15%)

**Fix**:
```python
# Edit configs/experiment_config.py
MONOTONIC_RECOVERY_EPOCHS = 2
MONOTONIC_RECOVERY_WARMUP_RATIO = 0.20
```

### Attack Success Rate Not Reduced

**Possible Causes**:
- Monotonicity not applied correctly (check stage 1 logs)
- Attack threshold too lenient (try 20% instead of 15%)
- Different attack implementation than T5 experiments

**Check**:
```bash
# Verify monotonicity
grep "All FFN weights are non-negative" \
  $SCRATCH/foundation_llm_work/stage_logs/stage_1_apply_monotonicity.log
```

## Final Checklist

Before submitting paper updates:

- [ ] All 8 stages completed successfully
- [ ] Results extracted from JSON files
- [ ] Perplexity gap reasonable (<10%)
- [ ] Attack reduction significant (>50%)
- [ ] Numbers match across all tables
- [ ] Red text removed from Section 4.3
- [ ] Methodology notes updated with actual observations
- [ ] Multi-seed results included (if applicable)
- [ ] Plots generated (optional but recommended)

## Additional Analysis (Optional)

Consider adding to paper:

### Benchmark Results

From `evaluation_results.json`:
- LAMBADA accuracy
- HellaSwag performance
- Winogrande scores

**Where to Add**: New paragraph in Section 4.3 after Table 7

### Training Curves

Plot perplexity vs. training steps:
- Shows convergence behavior
- Visualizes recovery from monotonicity initialization

**Where to Add**: Figure in Section 4.3 (as Figure 3 or 4)

### Attack Transferability

From UAT transfer matrix:
- Do Pythia triggers transfer to T5?
- Are monotonic models robust to cross-model attacks?

**Where to Add**: Paragraph in Section 4.2 (UAT section)

## Questions?

- **Unclear which numbers to use?** Check `experiment_summary.txt` first
- **Results don't make sense?** Verify training completed (check logs)
- **Need multi-seed?** Run `EXPERIMENT_SEED=1337 bash run_all.sh` (repeat for each seed)
- **Want to add more models?** Duplicate pipeline, change `MODEL_NAME` in config

---

**Remember**: The goal is to replace **all red text** in Section 4.3 with actual experimental results. This validates that monotonicity scales beyond summarization to general-purpose LLMs.
