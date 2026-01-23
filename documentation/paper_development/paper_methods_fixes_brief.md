# Methods Fixes - Quick Reference

## CRITICAL - Must Fix Before Submission

### Fair Comparison Issues
- Baseline: 5 epochs vs Monotonic: 7 epochs → **Train both for same epochs**
- Extended warmup for monotonic (15%) vs baseline (10%) → **Must justify or equalize**
- Different training = confounded results → **Robustness gains could just be more training**

### Sample Size
- Current: 200 examples → **Use full test set (11,490) or minimum 1,000**
- Bootstrap CIs unreliable with n=200
- Statistical power too low for significance claims

### Missing Results
- No clean performance table → **Add ROUGE scores for all models (no attack)**
- UAT attacks described but no results shown → **Add UAT results table**
- No transfer matrix → **Add cross-model attack transfer results**

### Statistical Testing
- Using independent t-tests → **Should use paired t-tests (same examples)**
- No multiple comparison correction → **Apply Bonferroni or FDR correction**
- Missing effect sizes → **Report Cohen's d alongside p-values**

## HIGH PRIORITY

### Dataset Details Missing
- Exact dataset sizes unclear → **Specify N for each train/val split**
- How datasets combined? → **Describe concatenation/sampling strategy**
- Data preprocessing? → **Specify tokenization, filtering, cleaning steps**
- CNN/DM in training? → **Explicitly state it's held-out for evaluation**

### Hyperparameter Table Needed
- Scatter details in text → **Create comprehensive hyperparameter table**
- Include: lr, weight decay, batch size, epochs, warmup, gradient clip, optimizer
- Show baseline vs monotonic side-by-side

### Attack Protocol Details
- **UAT:** Trigger length? Vocabulary for candidates? Optimization objective formula?
- **HotFlip:** Position selection method? Candidate vocabulary? Single pass or restarts?
- **Metrics:** Exact formulas for degradation, success rate threshold justification

### Reproducibility Gaps
- Missing random seeds → **Specify seeds for Python, NumPy, PyTorch**
- Missing hardware specs → **Report GPU model, memory, compute time**
- Missing software versions → **PyTorch version, Transformers version, Python version**
- No code availability statement → **Add anonymized code/checkpoint availability**

## MEDIUM PRIORITY

### Ablation Studies Missing
- Does baseline improve with 7 epochs? → **Train baseline-7epoch variant**
- Projection vs softplus? → **Compare optimization methods**
- Different constraint locations? → **Try constraining attention layers**
- Initialization sensitivity? → **Test with random vs pretrained init**

### Decoding Details Incomplete
- max_new_tokens vs max_length? → **Clarify generation length limit**
- min_length setting? → **Specify minimum generation length**
- no_repeat_ngram_size? → **Report n-gram blocking setting**
- early_stopping? → **Specify beam search termination**

### Training Protocol
- Convergence criteria? → **Early stopping patience? Validation frequency?**
- Checkpoint selection? → **Best validation loss? Last epoch? Average of last K?**
- Gradient accumulation? → **Specify if used**
- Mixed precision? → **fp32, fp16, or bf16?**

### Evaluation Metrics
- ROUGE library/version? → **Specify: rouge-score v0.1.2**
- Stemming enabled? → **Porter stemmer on/off?**
- Lowercase normalization? → **Specify normalization steps**
- Bootstrap method? → **Percentile, BCa, or studentized?**

## STRUCTURAL IMPROVEMENTS

### Expand Methods Section
- Current: ~0.5 pages → **Target: 2-3 pages**
- Add subsections:
  - Model Architecture (T5-small details)
  - Training Data (splits, sizes, statistics)
  - Training Procedure (optimization, convergence)
  - Evaluation Protocol (metrics, implementation)
  - Attack Protocols (detailed algorithms)
  - Statistical Analysis (tests, corrections, CIs)

### Add Missing Tables
- **Table 1:** Hyperparameters (baseline vs monotonic)
- **Table 2:** Dataset statistics (all splits)
- **Table 3:** Clean performance (ROUGE-1/2/L)
- **Table 4:** UAT attack results
- **Table 5:** Attack transfer matrix
- **Table 6:** Ablation studies

### Add Missing Figures
- **Figure 1:** Training curves (loss over epochs, both models)
- **Figure 2:** ROUGE degradation across attack budgets
- **Figure 3:** Attack success rate vs trigger length

## QUICK FIXES

### Typos/Errors
- "guaranties" → "guarantees" (abstract)
- Definition numbering inconsistent → Fix cross-references
- $\phi^{-1}$ formula needs numerical stability discussion

### Notation
- Too much notation in Prelim → Move to appendix
- Lower/upper closed sets not used → Remove if not needed
- Maximal/minimal points not used → Remove if not needed

### Clarity
- "biased toward disruptive symbols" too vague → **Define explicitly**
- "approximately 50K examples each" → **Give exact numbers**
- "extended warmup phase" → **Justify why monotonic needs more warmup**

## CHECKLIST FOR REVISION

### Experimental Design
- [ ] Train baseline and monotonic with identical epochs
- [ ] Use n≥1000 for evaluation (ideally full test set)
- [ ] Run ablation: baseline with 7 epochs
- [ ] Run ablation: different constraint locations
- [ ] Run ablation: projection vs softplus

### Results to Add
- [ ] Clean performance table (ROUGE without attacks)
- [ ] UAT attack results table
- [ ] Transfer attack matrix
- [ ] Ablation study results
- [ ] Training curves figure

### Methods Details to Add
- [ ] Complete dataset table with exact splits/sizes
- [ ] Complete hyperparameter table
- [ ] Exact ROUGE implementation details
- [ ] Complete attack algorithm descriptions
- [ ] Statistical testing methodology
- [ ] Computational cost analysis

### Statistical Rigor
- [ ] Switch to paired t-tests
- [ ] Add multiple comparison correction
- [ ] Report effect sizes (Cohen's d)
- [ ] Report actual p-values and CIs
- [ ] Increase bootstrap samples if n is small

### Reproducibility
- [ ] Add random seed specification
- [ ] Add hardware/software details
- [ ] Add training time/cost
- [ ] Add code availability statement
- [ ] Add checkpoint availability statement

## ONE-SENTENCE SUMMARY

**The paper needs fair comparison (same epochs), larger samples (n≥1000), clean performance results, UAT results, proper statistical testing, and 10x more methodological detail to be acceptable at ICML.**
