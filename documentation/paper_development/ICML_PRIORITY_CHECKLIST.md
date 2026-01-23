# ICML Submission Priority Checklist

## For Strong ICML 2025 Acceptance

### 🚨 CRITICAL (Already Fixed)

- [x] **Fair comparison:** Both models train 7 epochs (was 5 vs 7)
- [x] **Sample size:** Use full test sets n=11,490 (was 200)
- [x] **Config committed** and pushed to git

### 🔴 MUST HAVE (Will Cause Rejection If Missing)

#### Experiments:
- [ ] **Clean performance table:** ROUGE-1/2/L for all models (no attack)
- [ ] **UAT results table:** Currently described but no results shown
- [ ] **Multi-dataset results:** Run on CNN/DM + XSUM + SAMSum (infrastructure exists)
- [ ] **Ablation: Equal training budget:** Compare baseline-5epoch vs baseline-7epoch

#### Paper:
- [ ] **Fix statistical tests:** Change to paired t-tests + Bonferroni correction
- [ ] **Add dataset table:** Complete statistics for train/val/test splits
- [ ] **Add hyperparameter table:** All settings for both models
- [ ] **Add computational cost:** Training time, inference time, memory

**Timeline:** Next pipeline run (Week 1)

---

### 🟡 HIGHLY RECOMMENDED (Significantly Strengthens Paper)

#### Experiments:
- [ ] **Multi-seed runs:** Run with 5 seeds, report mean ± std
- [ ] **Transfer attack matrix:** Cross-model attack evaluation
- [ ] **Attack budget sensitivity:** Vary trigger length {1,3,5,7,10}
- [ ] **T5-base scaling:** Show approach works on 220M param model
- [ ] **Ablation: Constraint location:** FFN-only vs Attention-only vs Both

#### Analysis:
- [ ] **Gradient norm analysis:** Why gradient attacks less effective?
- [ ] **Weight distribution analysis:** How do learned weights differ?
- [ ] **Computational cost analysis:** Overhead of softplus parameterization

#### Paper:
- [ ] **Discussion section:** Why monotonicity helps, limitations, future work
- [ ] **Error analysis:** Failure modes, when monotonicity doesn't help
- [ ] **Comparison to other defenses:** Adversarial training, smoothing, etc.

**Timeline:** Weeks 2-3

---

### 🟢 NICE TO HAVE (Makes Paper Excellent)

#### Experiments:
- [ ] **T5-large scaling:** 770M params (if resources available)
- [ ] **Ablation: Parameterization:** Softplus vs Projection vs Exponential
- [ ] **Ablation: Initialization:** Pretrained vs Random vs Zero
- [ ] **Additional attack types:** TextFooler, BERT-Attack, paraphrasing
- [ ] **Certified robustness:** Compute provable bounds (if tractable)

#### Analysis:
- [ ] **Lipschitz constant analysis:** Bound on output change
- [ ] **Loss landscape visualization:** Show smoothness differences
- [ ] **Attention pattern analysis:** Stability under perturbations
- [ ] **Per-example analysis:** Which examples benefit most?

#### Paper:
- [ ] **Theoretical robustness bounds:** Prove Lipschitz properties
- [ ] **Comparison to vision:** Connect to monotonic CNNs
- [ ] **Deployment considerations:** Production readiness discussion

**Timeline:** Weeks 4-5

---

## Immediate Action Items (This Week)

### 1. Run Full Pipeline with Fixes
```bash
cd hpc_version
./run_all.sh  # Uses new config: 7 epochs, full test sets
```

**Collect:**
- Clean ROUGE for all models (Table 1)
- HotFlip results (update Table 1)
- UAT results (new Table 2)
- Results on all 3 datasets

### 2. Quick Configuration Updates

Enable analysis features:
```python
# In experiment_config.py (already added):
TRACK_TRAINING_TIME = True
TRACK_INFERENCE_TIME = True
COMPUTE_GRADIENT_NORMS = True
```

### 3. Create Results Tables for Paper

After pipeline completes, create:
- `results/table1_clean_performance.tex`
- `results/table2_uat_results.tex`
- `results/table3_transfer_matrix.tex`
- `results/table4_dataset_statistics.tex`

---

## What Makes ICML Papers Strong (Based on Research)

### ICML Reviewers Look For:

1. **Clear claims backed by evidence** ✓ (you have this)
2. **Appropriate benchmarks** ✓ (CNN/DM standard for summarization)
3. **Rigorous evaluation** ← IMPROVED (full test sets, fair comparison)
4. **Ablation studies** ← NEED TO ADD
5. **Generalization evidence** ← ADD (multi-dataset, multi-seed)
6. **Scalability** ← ADD (T5-base at minimum)
7. **Computational feasibility** ← MEASURE AND REPORT
8. **Novel insights** ✓ (monotonicity for LLM robustness is novel)

### What Gets Papers Rejected:

1. ❌ Unfair comparisons ← FIXED
2. ❌ Insufficient evidence (small n) ← FIXED
3. ❌ Missing ablations ← NEED TO ADD
4. ❌ No comparison to baselines ← NEED TO ADD
5. ❌ Lack of generalization ← EASY FIX (run on XSUM/SAMSum)
6. ❌ Poor writing/clarity ← NEEDS REVISION
7. ❌ Incremental contributions ← YOUR WORK IS NOVEL (good!)

---

## Quick Wins (Implement Now)

### Configuration Changes (Low Effort, High Impact):

```python
# Already in your pipeline, just need to enable:

# 1. Multi-dataset evaluation
# Your TEST_DATASETS already includes XSUM and SAMSum!
# Just make sure results are reported in paper

# 2. Transfer matrix
# Already computed in stage_5_uat_attacks.py!
# Just need to add to paper

# 3. Computational tracking
# Add simple timer wrapper to training loop
```

### Analysis Scripts to Create:

```python
# hpc_version/scripts/compute_gradient_norms.py
"""
Load baseline and monotonic models
For 100 random examples:
    Compute ||∂Loss/∂input_embeddings||
    Compare baseline vs monotonic
Report: mean, std, histogram
"""

# hpc_version/scripts/analyze_weight_distributions.py
"""
Load baseline and monotonic checkpoints
Extract FFN weights
Plot histograms:
    - Baseline: weights span negative/positive
    - Monotonic: all positive
Report statistics: mean, std, min, max, sparsity
"""

# hpc_version/scripts/measure_computational_cost.py
"""
Measure:
    - Training time per epoch
    - Inference time per example
    - Peak GPU memory
    - FLOPs (if feasible)
Compare baseline vs monotonic
"""
```

---

## Paper Revision Priority

### Methods Section (Must Expand):

**Add 4 subsections:**

1. **Model Architecture** (T5-small details, parameter count)
2. **Datasets** (complete table with train/val/test splits)
3. **Training Protocol** (complete hyperparameter table)
4. **Evaluation Protocol** (ROUGE details, statistical testing, attack details)

**Target:** Expand from 0.5 pages to 2.5 pages

### Results Section (Must Expand):

**Add 5 subsections:**

1. **Clean Performance** (Table: ROUGE scores, no attacks)
2. **HotFlip Robustness** (expand current half-paragraph)
3. **UAT Robustness** (new: Table + analysis)
4. **Multi-Dataset Generalization** (CNN/DM, XSUM, SAMSum)
5. **Ablation Studies** (training budget, constraint location)

**Target:** Expand from 0.5 pages to 3-4 pages

### Discussion Section (Currently Missing):

**Add:**
1. Why does monotonicity help? (mechanistic insights)
2. Performance-robustness tradeoffs
3. Computational costs
4. Limitations and scope
5. Broader impact

**Target:** 1-1.5 pages

---

## Expected Paper Structure (ICML-Ready)

```
Abstract (0.25 pages)
1. Introduction (1.5 pages) ✓ already good
2. Related Work (1.5 pages) ✓ already good
3. Preliminaries (1 page) - could be shortened
4. Methods (2.5 pages) ← EXPAND from 0.5
5. Results (3.5 pages) ← EXPAND from 0.5
6. Discussion (1.5 pages) ← ADD
7. Conclusion (0.25 pages) ← ADD
References (1+ pages)
---
Total: ~8 pages (ICML limit)

Appendix:
A. Complete hyperparameter tables
B. Dataset statistics
C. Additional ablation results
D. Reproducibility details
E. Failure case examples
```

---

## Realistic Timeline

### Optimistic (2 weeks):
- Week 1: Run pipeline with fixes, collect core results
- Week 2: Run key ablations, write paper revisions
- **Result:** Submittable but not maximally strong

### Realistic (4 weeks):
- Week 1: Core experiments (fair comparison, full eval)
- Week 2: Ablations (training budget, constraint location, multi-seed)
- Week 3: Analysis (gradients, weights, computational cost)
- Week 4: Paper writing and revision
- **Result:** Strong submission with good acceptance chances

### Ideal (6 weeks):
- Weeks 1-3: As above
- Week 4: Scaling to T5-base, additional attack types
- Week 5: Comparison to other defenses, theoretical analysis
- Week 6: Paper polishing, comprehensive appendix
- **Result:** Very strong submission, excellent acceptance chances

---

## What to Prioritize Based on ICML Criteria

**ICML Values (in priority order):**

1. **Correctness** ← Fair comparison CRITICAL (fixed ✅)
2. **Evidence quality** ← Sample size CRITICAL (fixed ✅)
3. **Generalization** ← Multi-dataset (easy win)
4. **Ablations** ← Isolate contribution (important)
5. **Scalability** ← T5-base (recommended)
6. **Reproducibility** ← Details (easy to add)

**Your Strengths:**
- Novel idea (monotonicity for LLM robustness)
- Solid theory (definitions, propositions)
- Good writing (intro, related work)
- Practical approach (works with pretrained models)

**Your Weaknesses (Now):**
- Thin empirical evaluation (only 1 dataset result shown)
- Missing ablations
- No mechanistic analysis
- No scaling evidence

**After Implementing Suggestions:**
- Strong empirical evaluation (3 datasets, multi-seed)
- Complete ablations
- Mechanistic insights
- Scaling to T5-base

**Acceptance Probability:**
- Current draft: ~20% (missing too much)
- With must-haves: ~50% (meets bar)
- With highly recommended: ~70% (strong submission)
- With nice-to-haves: ~85% (excellent submission)

---

## Summary of Recommendations

### Already Implemented ✅:
1. Fair comparison (7 epochs both)
2. Full test sets (n=11,490)
3. Analysis infrastructure (tracking flags)

### Easy Wins (Use Existing Infrastructure):
4. Multi-dataset results (XSUM, SAMSum already in pipeline)
5. Transfer matrix (already computed, just add to paper)
6. Training time tracking (just log timestamps)

### Important Additions (New Experiments):
7. Multi-seed runs (5 seeds)
8. Baseline-10epoch ablation
9. Gradient norm analysis
10. Weight distribution analysis

### Aspirational (If Time/Resources):
11. T5-base scaling
12. Additional ablations (parameterization, constraint location)
13. Comparison to other defenses
14. Theoretical robustness bounds

**Recommendation: Focus on items 1-10 for strong ICML submission.**
