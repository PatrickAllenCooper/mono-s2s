# Remaining Work to Complete Paper Verification

**Status as of**: January 27, 2026
**Current Progress**: ~40% of paper values verified
**Estimated Time to Complete**: 2-3 weeks

## ✅ COMPLETED (Verified Experimental Evidence)

### Table 1: Training Dynamics ✅
- **Status**: VERIFIED with actual experimental data
- **Source**: `downloaded_results/seed_42/baseline_training_history.json`, `monotonic_training_history.json`
- **Values**:
  - Baseline: Initial 2.90 → Final 2.25 ✅
  - Monotonic: Initial 4.97 → Final 2.54 ✅
- **Action**: None needed - complete

### Table 2: ROUGE Scores (Seed 42 Only) ✅
- **Status**: VERIFIED with actual experimental data (n=200)
- **Source**: `downloaded_results/seed_42/evaluation_results.json`
- **Values**:
  - Standard ROUGE-L: 26.6% ✅
  - Baseline ROUGE-L: 25.0% ✅
  - Monotonic ROUGE-L: 24.2% ✅
- **Action**: None needed - complete

---

## ⏳ IN PROGRESS (Experiments Running or Ready to Run)

### Stages 5-7: Attack Results for Seed 42

**Status**: NOT YET RUN (job 23293358 only completed stage 4)

**Required**:
- [ ] Stage 5: UAT (Universal Adversarial Trigger) attacks
- [ ] Stage 6: HotFlip gradient-based attacks  
- [ ] Stage 7: Aggregate all results

**Commands to Run (On HPC)**:
```bash
cd /projects/paco0228/mono-s2s
git pull origin main  # Get sbatch fix
cd hpc_version
bash run_all.sh  # Will auto-skip completed stages 0-4, run 5-7
```

**Expected Runtime**: ~10 hours

**Will Provide**:
- UAT attack results (for Table 6)
- HotFlip attack results (for Table 5)
- Final aggregated summary

**For Paper**:
- Table 5: HotFlip results (seed 42)
- Table 6: UAT results (seed 42)

---

## 🔴 MISSING (Red Placeholders - Need Multi-Seed Experiments)

### Table 3: Multi-Seed Training Dynamics

**Status**: RED PLACEHOLDERS (confabulated values)

**What's Needed**: Run experiments for seeds 1337, 2024, 8888, 12345

**Current Values**: All confabulated (in red text)

**To Complete**:
1. Run full pipeline for seed 1337
2. Run full pipeline for seed 2024
3. Run full pipeline for seed 8888
4. Run full pipeline for seed 12345
5. Aggregate results across all 5 seeds
6. Compute mean ± std for all metrics

**Commands (On HPC)**:
```bash
cd /projects/paco0228/mono-s2s/hpc_version

# Run each seed
EXPERIMENT_SEED=1337 bash run_all.sh
EXPERIMENT_SEED=2024 bash run_all.sh
EXPERIMENT_SEED=8888 bash run_all.sh
EXPERIMENT_SEED=12345 bash run_all.sh

# After all complete, aggregate
python scripts/aggregate_multi_seed.py --seeds 42,1337,2024,8888,12345
```

**Expected Total Runtime**: ~250-300 hours (can run in parallel)
**Wall Time**: ~50-60 hours if running 5 seeds simultaneously

**Will Provide**:
- Mean and std for training losses
- Cross-seed variability
- Statistical significance tests

### Table 4: Multi-Seed ROUGE Scores

**Status**: RED PLACEHOLDERS (confabulated values)

**What's Needed**: Same as Table 3 - multi-seed experiments

**Source**: Aggregated from all 5 seed evaluation results

**Will Provide**:
- Mean ROUGE-1, ROUGE-2, ROUGE-L across seeds
- Standard deviations
- Confidence that results are reproducible

### Table 5: Multi-Seed HotFlip Attacks

**Status**: RED PLACEHOLDERS (confabulated values)

**What's Needed**:
1. Complete seed 42 HotFlip (stage 6) ← Next step
2. Run HotFlip for seeds 1337, 2024, 8888, 12345
3. Aggregate attack results across seeds

**Will Provide**:
- Attack success rates with std
- Degradation statistics
- Statistical significance of robustness gains

### Table 6: Multi-Seed UAT Attacks

**Status**: RED PLACEHOLDERS (confabulated values)

**What's Needed**: Same as HotFlip - multi-seed UAT results

**Will Provide**:
- Trigger effectiveness across seeds
- Transfer matrix analysis
- Robustness consistency

---

## 🟡 MISSING (Red Placeholders - Foundation Model Experiments)

### Table 7: Foundation Model Results (Section 4.3)

**Status**: RED PLACEHOLDERS (confabulated values)

**What's Needed**: Run Pythia-1.4B experiments

**Source**: `foundation_llm_experiments/` pipeline

**To Complete**:
1. Deploy foundation pipeline to HPC
2. Run full pipeline for Pythia-1.4B (seed 42)
3. Extract perplexity and attack results
4. Update Table 7 with real values

**Commands (On HPC)**:
```bash
cd /projects/paco0228/mono-s2s/foundation_llm_experiments
bash run_all.sh  # 60-70 hours
```

**Expected Runtime**: ~60-70 hours for seed 42

**Will Provide**:
- Pythia-1.4B perplexity (clean and under attack)
- HotFlip attack results for foundation model
- Evidence that monotonicity scales beyond summarization

---

## 📋 Complete Checklist

### Immediate (Next 24 Hours)

- [ ] **Complete T5 seed 42 attacks** (stages 5-7)
  - Commands: `cd hpc_version && bash run_all.sh`
  - Runtime: ~10 hours
  - Will verify: Table 5, Table 6 (seed 42 portion)

### Short-Term (Next 1-2 Weeks)

- [ ] **Run T5 multi-seed experiments** (seeds 1337, 2024, 8888, 12345)
  - Commands: Run `bash run_all.sh` with different EXPERIMENT_SEED
  - Runtime: ~250 hours total (50-60 hours if parallel)
  - Will verify: Tables 3, 4, 5, 6 (multi-seed stats)

- [ ] **Aggregate multi-seed results**
  - Commands: `python scripts/aggregate_multi_seed.py`
  - Runtime: 30 minutes
  - Will verify: All mean ± std values in red

### Medium-Term (Next 2-4 Weeks, Optional)

- [ ] **Run Pythia-1.4B foundation experiments** (seed 42)
  - Commands: `cd foundation_llm_experiments && bash run_all.sh`
  - Runtime: ~60-70 hours
  - Will verify: Table 7, Section 4.3

- [ ] **Run Pythia multi-seed** (optional, for robust stats)
  - Commands: Same as T5 multi-seed
  - Runtime: ~300 hours total
  - Will verify: Table 7 with mean ± std

---

## 🎯 Priority Order

### Must Have for Paper Submission

1. **T5 Seed 42 Complete** ← NEXT (10 hours)
   - Provides single-seed baseline
   - Demonstrates concept works
   - Tables 1, 2, 5, 6 (single values)

2. **T5 Multi-Seed** ← IMPORTANT (1-2 weeks)
   - Provides robust statistics
   - Shows reproducibility
   - Tables 3, 4, 5, 6 (mean ± std)
   - **Removes all red text** from main results

### Nice to Have for Stronger Paper

3. **Pythia Foundation Model** ← OPTIONAL (2-4 weeks)
   - Shows generalization beyond summarization
   - Section 4.3 evidence
   - Table 7
   - **Removes red text** from Section 4.3

---

## 📊 Current Paper Status

**Verified (Black Text)**:
- Abstract ✅
- Introduction ✅
- Methods ✅
- Table 1 (Training) ✅
- Table 2 (ROUGE, seed 42) ✅
- Theory sections ✅

**Confabulated (Red Text)**:
- Table 3: Multi-seed training ⚠️
- Table 4: Multi-seed ROUGE ⚠️
- Table 5: Multi-seed HotFlip ⚠️
- Table 6: Multi-seed UAT ⚠️
- Table 7: Foundation models ⚠️
- Section 4.3: Foundation discussion ⚠️

**Completion**: ~40% verified, ~60% placeholder

---

## ⏰ Timeline to Full Paper

### Minimum Viable (Seed 42 Only)

**Time**: 1 day
**Effort**: Complete stages 5-7 for seed 42
**Result**: Single-seed results, all tables have at least one data point
**Acceptable for**: Workshop submission, preprint

### Recommended (Multi-Seed T5)

**Time**: 2-3 weeks  
**Effort**: Run 4 more seeds + aggregation
**Result**: Robust statistics, reproducibility demonstrated
**Acceptable for**: Conference submission (ICML, NeurIPS)

### Complete (Multi-Seed T5 + Pythia)

**Time**: 4-6 weeks
**Effort**: All of above + foundation experiments
**Result**: No red text, all claims verified
**Acceptable for**: Top-tier conference, full rigor

---

## 🚀 Immediate Next Steps

**On HPC** (when you're ready):

```bash
# 1. Pull latest code (has sbatch fix)
cd /projects/paco0228/mono-s2s
git pull origin main

# 2. Complete seed 42 (stages 5-7)
cd hpc_version
bash run_all.sh

# Expected: Will skip completed stages 0-4, submit stages 5-7
# Runtime: ~10 hours
# Result: Complete seed 42 with attack results
```

**After seed 42 complete**:

Decide whether to:
- **Option A**: Submit paper with seed 42 only (faster, single-seed)
- **Option B**: Run multi-seed for robust stats (better, 2-3 weeks)
- **Option C**: Also run foundation models (strongest, 4-6 weeks)

---

## Summary

**Completed**: ~40% (seed 42 training and evaluation)
**Remaining for minimum viable**: 10 hours (seed 42 attacks)
**Remaining for robust paper**: 2-3 weeks (multi-seed)
**Remaining for complete paper**: 4-6 weeks (multi-seed + foundation)

**Immediate action**: Run stages 5-7 for seed 42 on HPC
