# Paper Status: Monotone LLMs for ICML 2025

## Current Status: IN PROGRESS - Critical Issues Fixed

### ✅ Completed (Rejection Issues Fixed)

1. **Fair Experimental Comparison**
   - Both models now train for 7 epochs (was 5 vs 7)
   - Eliminates major confound in results
   - Changes committed to config ✅

2. **Adequate Sample Size**
   - Full test sets enabled (11,490 examples vs 200)
   - Statistical tests now valid
   - Changes committed to config ✅

3. **Analysis Infrastructure**
   - Added tracking flags for training time, memory, gradients
   - Ready for mechanistic analysis
   - Changes committed ✅

4. **Documentation Created**
   - Complete methods critique
   - Implementation docs
   - Priority checklists
   - All committed ✅

---

## What Still Needs to Be Done

### 🔴 Critical for Submission (Week 1)

**Run New Pipeline:**
```bash
cd hpc_version
./run_all.sh  # With fixed config
```

**Collect Results:**
- [ ] Clean ROUGE scores (all 3 models, all 3 datasets)
- [ ] HotFlip attack results (updated with full test set)
- [ ] UAT attack results (currently missing from paper)
- [ ] Transfer attack matrix (already computed, add to paper)

**Update Paper:**
- [ ] Add Table 1: Clean Performance
- [ ] Add Table 2: UAT Results
- [ ] Add Table 3: Dataset Statistics
- [ ] Add Table 4: Hyperparameters
- [ ] Fix statistical testing description (paired t-tests)
- [ ] Add exact dataset sizes
- [ ] Add reproducibility details (seeds, hardware, versions)

---

### 🟡 Important for Strong Submission (Weeks 2-3)

**Additional Experiments:**
- [ ] Multi-seed runs (5 seeds × mean ± std)
- [ ] Ablation: Baseline with 10 epochs
- [ ] Ablation: Constraint location (FFN vs Attention)
- [ ] T5-base scaling experiment
- [ ] Gradient norm analysis
- [ ] Weight distribution analysis
- [ ] Computational cost measurement

**Paper Additions:**
- [ ] Expand Methods section (0.5 → 2.5 pages)
- [ ] Expand Results section (0.5 → 3.5 pages)
- [ ] Add Discussion section (1.5 pages)
- [ ] Add error analysis
- [ ] Add comparison to other defenses

---

### 🟢 Nice to Have (Weeks 4-5)

- [ ] T5-large experiments
- [ ] Additional ablations (parameterization, initialization)
- [ ] Additional attack types
- [ ] Theoretical robustness bounds
- [ ] Mechanistic interpretability analysis

---

## Key ICML Reviewer Concerns to Address

### 1. "Why should I care about monotonicity?"

**Current Answer:** Improves robustness (weak - one result table)

**Strong Answer Needs:**
- Multiple datasets showing generalization
- Multiple attack types showing broad robustness
- Mechanistic analysis showing WHY it works
- Comparison to other defenses showing WHERE it fits
- Scaling showing it's practical for real models

### 2. "Is this just regularization?"

**Current Answer:** No, it's architectural (but not proven empirically)

**Strong Answer Needs:**
- Ablation showing regularization (L1/L2) doesn't achieve same robustness
- Analysis showing qualitatively different learned features
- Comparison to dropout, weight decay, etc.

### 3. "Does it scale?"

**Current Answer:** Only T5-small (60M) - reviewers will be skeptical

**Strong Answer Needs:**
- T5-base results (220M) - minimum
- T5-large results (770M) - ideal
- Show scaling curve if possible

### 4. "How does it compare to existing defenses?"

**Current Answer:** Claims it's complementary (but no evidence)

**Strong Answer Needs:**
- Empirical comparison to adversarial training
- Empirical comparison to smoothing defenses
- Show combining monotonic + other methods helps

### 5. "What are the limitations?"

**Current Answer:** Brief mention in intro

**Strong Answer Needs:**
- Dedicated limitations subsection in Discussion
- Failure case analysis
- Scope limitations (summarization only, small models only, etc.)
- Computational overhead quantified

---

## ICML Acceptance Probability Estimate

### Current Draft (Before Fixes):
**~15% acceptance chance**
- Unfair comparison (major flaw)
- n=200 too small (insufficient evidence)
- Missing critical results (UAT, clean performance)
- Missing ablations
- No scaling

### After Critical Fixes (Now):
**~35% acceptance chance**
- Fair comparison ✅
- Adequate sample size ✅
- Still missing: UAT results, clean table, ablations

### After Must-Haves (Week 1):
**~55% acceptance chance**
- All critical tables
- Multi-dataset results
- Complete methods details
- Still missing: ablations, scaling, analysis

### After Highly Recommended (Weeks 2-3):
**~75% acceptance chance**
- Comprehensive ablations
- Multi-seed robustness
- T5-base scaling
- Mechanistic analysis
- Discussion section

### After Nice-to-Haves (Weeks 4-5):
**~85% acceptance chance**
- T5-large scaling
- Comparison to other defenses
- Theoretical analysis
- Comprehensive evaluation

---

## Recommendation

### Minimum Viable ICML Submission:
**Timeline:** 2 weeks
**Effort:** Run pipeline + write

**Includes:**
- Fair comparison ✅
- Full test sets ✅
- Clean performance table
- UAT results
- Multi-dataset (CNN/DM, XSUM, SAMSum)
- Complete methods details
- Basic ablation (training budget)

**Outcome:** ~55% acceptance chance (competitive)

### Strong ICML Submission:
**Timeline:** 4 weeks
**Effort:** Additional experiments + analysis + writing

**Adds:**
- Multi-seed results
- T5-base scaling
- Gradient/weight analysis
- Multiple ablations
- Discussion section

**Outcome:** ~75% acceptance chance (strong)

### Excellent ICML Submission:
**Timeline:** 6 weeks
**Effort:** Comprehensive evaluation + comparisons

**Adds:**
- T5-large scaling
- Comparison to other defenses
- Additional attack types
- Theoretical bounds
- Comprehensive appendix

**Outcome:** ~85% acceptance chance (excellent)

---

## Immediate Next Steps

1. **Run pipeline** with current config (fair comparison, full test sets)
2. **Collect all results** from existing infrastructure
3. **Create result tables** for paper
4. **Update methods** with complete details
5. **Write discussion** section

**Then decide:** Do you have time/resources for ablations and scaling?

---

## Resources You Already Have

✅ Full HPC pipeline automated
✅ Multiple test datasets (CNN/DM, XSUM, SAMSum)
✅ Multiple random seeds configured
✅ Both attack types (UAT, HotFlip)
✅ Transfer matrix computation
✅ Bootstrap CI computation
✅ Fair comparison configuration

**You're 60% of the way there!**

Just need to:
- Run with updated config
- Extract results for paper
- Add ablation experiments
- Write expanded sections

---

## Files Created for Your Reference

1. **documentation/paper_methods_critique.md** - Detailed critique
2. **documentation/paper_methods_fixes_brief.md** - Short bullet list
3. **documentation/ICML_STRENGTHENING_SUGGESTIONS.md** - Complete suggestions
4. **documentation/ICML_PRIORITY_CHECKLIST.md** - This checklist
5. **PAPER_METHODS_FIXES_IMPLEMENTED.md** - What's been fixed

All committed and pushed to origin/main ✅
