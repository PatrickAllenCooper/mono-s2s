# Methods Critique: Monotone LLMs Paper

## Critical Methods Issues to Address

### 1. EXPERIMENTAL DESIGN - CRITICAL ISSUES

#### Missing Fair Comparison Baseline
**PROBLEM:** The paper compares monotonic model against "Standard" (pretrained) and "Baseline" but doesn't clearly specify what "Baseline" means.

**FIX NEEDED:**
```
Current: "Standard" vs "Baseline" vs "Monotonic"
Should be: 
- Standard T5 (pretrained, NOT fine-tuned) - reference only
- Baseline T5 (fine-tuned, unconstrained) - FAIR CONTROL
- Monotonic T5 (fine-tuned, constrained) - TREATMENT
```

The baseline MUST be fine-tuned with identical data/hyperparameters as monotonic to avoid confounding fine-tuning effects with constraint effects.

#### Hyperparameter Asymmetry
**PROBLEM:** "The baseline model is trained for 5 epochs, while the monotone model is trained for 7 epochs"

**ISSUE:** This is NOT a fair comparison. You're giving monotonic model MORE training, so improved robustness could just be from more training, not from monotonicity.

**FIX:** Either:
1. Train both for same epochs (recommended for fair comparison)
2. Train baseline until convergence, match monotonic to same validation loss
3. Explicitly justify and ablate this choice

**CURRENT METHODS VIOLATE FAIR COMPARISON PRINCIPLES**

### 2. SAMPLE SIZE - CRITICAL WEAKNESS

**PROBLEM:** "evaluation subsets of 200 examples"

**ISSUE:** 200 examples is FAR too small for:
- Reliable ROUGE scores (need 1000+ for stable estimates)
- Statistical significance testing
- Bootstrap confidence intervals (unreliable with n=200)
- Adversarial attack evaluation

**FIX:**
- Use FULL test sets (CNN/DM has 11,490 test examples)
- If computational constraints, use AT LEAST 1,000 examples
- Report sample size justification based on power analysis
- Current n=200 undermines all statistical claims

### 3. MISSING METHODOLOGICAL DETAILS

#### Training Data
**MISSING:**
- Exact dataset splits used (train/val/test)
- How many examples from each dataset?
- Data preprocessing steps
- How were datasets combined?
- Any data filtering or cleaning?

**NEEDED:**
```latex
\paragraph{Training Data.}
We train on DialogSum (train split, N=12,460), HighlightSum 
(train split, N=XXX), and arXiv abstracts (train split, N=XXX),
yielding a total of XXX training examples and XXX validation examples.
We ensure no overlap between training and evaluation sets by using
CNN/DailyMail for evaluation only (not in training data).
```

#### Evaluation Protocol
**MISSING:**
- Exact ROUGE implementation (rouge-score library? which version?)
- Stemming settings
- Tokenization details
- How bootstrap CIs were computed (percentile? BCa?)
- Significance testing details

**NEEDED:**
```latex
\paragraph{Evaluation Metrics.}
ROUGE scores are computed using the rouge-score Python library (v0.1.2)
with stemming enabled. We report bootstrap 95% confidence intervals
using the percentile method with 1,000 resamples. Statistical
significance is assessed using Welch's t-test with Bonferroni 
correction for multiple comparisons.
```

#### Reproducibility
**MISSING:**
- Random seeds
- Hardware specifications
- Training time
- Convergence criteria
- Early stopping details (if any)

**NEEDED:**
```latex
\paragraph{Reproducibility.}
All experiments use fixed random seeds (Python=42, NumPy=42, 
PyTorch=42) and deterministic algorithms where available. 
Training is conducted on a single NVIDIA A100 GPU (40GB). 
We train until validation loss converges (patience=3 epochs) 
or maximum epochs is reached. Training the baseline model 
requires approximately 8 hours; the monotonic model requires 
11 hours due to additional epochs.
```

### 4. ATTACK EVALUATION - INSUFFICIENT DETAIL

#### Universal Adversarial Triggers
**MISSING:**
- Trigger length (you mention 5 tokens for HotFlip, what about UAT?)
- Exact optimization objective
- Vocabulary size for candidate tokens
- "biased toward disruptive symbols" - WHAT DOES THIS MEAN?
- How is "disruptive" defined?
- Position of trigger (prepended, appended, inserted?)

**NEEDED:**
```latex
\paragraph{Universal Adversarial Trigger Optimization.}
Triggers consist of 5 tokens optimized to maximize average 
cross-entropy loss when prepended to input texts. We optimize 
using greedy coordinate ascent over a vocabulary of 1,000 
high-frequency tokens, with 3 random restarts and 50 iterations 
each. Triggers are optimized on CNN/DM validation set (n=500) 
and evaluated on test set (n=1000, disjoint from optimization).
```

#### HotFlip Details
**MISSING:**
- Which embedding layer gradients?
- How are the 5 positions selected? (top-5 gradient magnitude?)
- Candidate vocabulary for replacements
- Multiple restarts or single pass?
- Position constraints (can't flip special tokens?)

#### Statistical Testing
**PROBLEM:** "statistical significance assessed via independent $t$-tests"

**ISSUES:**
- Independent t-tests assume independence - BUT you're comparing same examples
- Should use PAIRED t-tests
- Need multiple comparison correction
- Need to report effect sizes, not just p-values
- Need to report actual p-values, not just "significant"

**FIX:**
```latex
Statistical significance is assessed using paired t-tests 
(since we evaluate the same examples across models) with 
Bonferroni correction for 3 pairwise comparisons (α=0.05/3). 
We report Cohen's d effect sizes alongside p-values.
```

### 5. MISSING CRITICAL EXPERIMENTS

#### Ablation Studies
**MISSING:**
- What if you apply constraints to attention instead of FFN?
- What if you use projection instead of softplus?
- What if you don't use pretrained initialization?
- What if you train baseline for 7 epochs too?

These are ESSENTIAL for a methods paper.

#### Clean Performance Comparison
**MISSING:** No results on clean (non-adversarial) performance!

**CRITICAL ADDITION NEEDED:**
```latex
\begin{table}[t]
\caption{Clean task performance on CNN/DailyMail test set.}
\begin{tabular}{lccc}
\toprule
Model & R-1 & R-2 & R-L \\
\midrule
Standard  & XX.X & XX.X & XX.X \\
Baseline  & XX.X & XX.X & XX.X \\
Monotonic & XX.X & XX.X & XX.X \\
\bottomrule
\end{tabular}
\end{table}
```

Without this, readers can't assess the performance-robustness trade-off!

#### UAT Results
**MISSING:** You describe UAT attacks but show NO results! Only HotFlip results in Table 1.

**NEEDED:**
- UAT results table
- Transfer matrix results
- Comparison of UAT vs HotFlip

### 6. DEFINITION/NOTATION ISSUES

#### Bias Terms
**PROBLEM:** Proposition 1 requires $W_i \succeq 0$ but doesn't address bias terms $b_i$.

**ISSUE:** If $b_i$ can be negative, the network may not be monotone even with $W_i \succeq 0$.

**FIX:**
Add clarification:
```latex
\begin{remark}
Proposition~\ref{Prop1} addresses weight matrices only. 
For strict monotonicity, bias terms must also satisfy 
$b_i \succeq 0$, or be eliminated. In our implementation, 
we constrain weight matrices only and leave bias terms 
unconstrained, resulting in quasi-monotone behavior.
\end{remark}
```

#### Definition Numbering
**PROBLEM:** "Definition~3" referenced but definitions aren't numbered consistently.

**FIX:** Ensure all definitions are numbered and cross-referenced correctly.

### 7. INITIALIZATION FORMULA ERROR

**PROBLEM:** 
```latex
V_{\mathrm{init}} = \phi^{-1}\bigl(|W_{\mathrm{pre}}| + \epsilon\bigr)
```

**ISSUE:** The inverse softplus is:
$$\phi^{-1}(x) = \log(\exp(x) - 1)$$

But this is undefined for $x \leq 0$! Taking absolute value helps but you need to be more careful about numerical stability.

**FIX:**
```latex
V_{\mathrm{init}} = \log\bigl(\exp(|W_{\mathrm{pre}}| + \epsilon) - 1\bigr)
```

And add discussion of numerical stability for small weights.

### 8. MISSING DETAILS ON DATA SPLITS

**CRITICAL:** You must specify:
- Training used: DialogSum (train), HighlightSum (train), arXiv (train)
- Validation used: DialogSum (val), HighlightSum (val), arXiv (val)
- Testing used: CNN/DM (test) - NEVER SEEN DURING TRAINING

**MUST EXPLICITLY STATE:**
"CNN/DailyMail is held out entirely from training to ensure evaluation measures generalization to unseen distributions."

### 9. DECODING PARAMETERS - INCOMPLETE

**MISSING:**
- max_length or max_new_tokens?
- min_length constraints?
- no_repeat_ngram_size?
- early_stopping settings?
- temperature (if used)?

These affect generation quality and must be reported for reproducibility.

### 10. MISSING BASELINE COMPARISONS

**NEEDED:**
- Compare to adversarial training baseline
- Compare to certified defense methods
- Compare to smoothing defenses (SmoothLLM from related work)
- Show that monotonicity is complementary to these

## METHODS SECTION STRUCTURE RECOMMENDATIONS

### Current Structure (Weak)
```
3. Preliminaries (heavy on theory, light on connection to LLMs)
4. Methods (too brief, missing critical details)
5. Results (only HotFlip, no clean performance, no UAT)
```

### Recommended Structure
```
3. Background
   3.1 Monotone Neural Networks (brief theory)
   3.2 Transformer Architecture (T5 overview)
   3.3 Adversarial Attacks on LLMs

4. Methods
   4.1 Monotone Transformer Architecture
       - Constraint scope (which layers)
       - Softplus parameterization
       - Initialization from pretrained
   
   4.2 Experimental Setup
       - Model Architecture (T5-small details)
       - Training Data (exact splits, sizes)
       - Hyperparameters (complete table)
       - Evaluation Datasets
   
   4.3 Training Protocol
       - Optimization details
       - Convergence criteria
       - Computational requirements
   
   4.4 Evaluation Metrics
       - ROUGE implementation details
       - Statistical testing procedures
       - Confidence interval methods
   
   4.5 Adversarial Attack Protocols
       4.5.1 Universal Adversarial Triggers
       4.5.2 HotFlip Attacks
       4.5.3 Attack Evaluation Procedure

5. Results
   5.1 Clean Task Performance
   5.2 HotFlip Robustness
   5.3 UAT Robustness
   5.4 Transfer Attack Analysis
   5.5 Ablation Studies
   
6. Discussion
   6.1 Performance-Robustness Tradeoffs
   6.2 Limitations
   6.3 Future Work
```

## CRITICAL ADDITIONS NEEDED

### Table 1: Hyperparameters (ESSENTIAL)
```latex
\begin{table}[t]
\caption{Training hyperparameters for all models.}
\begin{tabular}{lcc}
\toprule
Hyperparameter & Baseline & Monotonic \\
\midrule
Learning rate & 5e-5 & 5e-5 \\
Weight decay & 0.01 & 0.01 \\
Batch size & 4 & 4 \\
Epochs & 5 & 7 \\
Warmup ratio & 10\% & 15\% \\
Gradient clip & 1.0 & 1.0 \\
Optimizer & AdamW & AdamW \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 2: Dataset Statistics
```latex
\begin{table}[t]
\caption{Dataset statistics.}
\begin{tabular}{lrr}
\toprule
Dataset & Split & Size \\
\midrule
\multicolumn{3}{c}{\textit{Training}} \\
DialogSum & train & XX,XXX \\
HighlightSum & train & XX,XXX \\
arXiv & train & XX,XXX \\
\midrule
\multicolumn{3}{c}{\textit{Evaluation}} \\
CNN/DM & test & 11,490 \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 3: Clean Performance (MISSING!)
MUST add this table with ROUGE scores for all models on clean data.

### Table 4: UAT Results (MISSING!)
Currently UAT attacks are described but no results shown.

### Figure 1: Training Curves (Recommended)
Show loss curves for baseline vs monotonic to demonstrate:
- Both converge properly
- Monotonic converges to similar loss despite constraints
- No optimization issues

## SPECIFIC CORRECTIONS

### Abstract
- "guaranties" → "guarantees" (typo)
- Add quantitative results: "reducing attack success rate by X% while maintaining Y% of clean performance"

### Introduction
- Add brief roadmap at end: "The rest of this paper is organized as follows..."

### Related Work
- Missing paragraph on "Robustness via Architecture" - should discuss other architectural approaches

### Preliminaries
- Notation section is overkill for ICML - move to appendix
- Focus preliminaries on what's needed for methods

### Methods Section

#### Current Problems:
1. Too brief (half a page for entire methods)
2. Missing dataset details
3. Missing training details
4. Missing evaluation details
5. No ablation study description

#### Specific Additions Needed:

```latex
\subsection{Model Architecture}
We use T5-small~\citep{raffel2020exploring} as our base 
architecture, consisting of 6 encoder layers, 6 decoder 
layers, 512-dimensional hidden states, and 2048-dimensional 
FFN intermediate activations. The model contains 60M 
parameters total, of which approximately 40M reside in FFN 
weight matrices (the constrained components in our approach).

\subsection{Training Datasets}
\textbf{Training:} We combine three summarization datasets:
\begin{itemize}
\item DialogSum~\citep{XXX}: Dialogue summarization, 
      train split (N=12,460), validation split (N=500)
\item HighlightSum~\citep{XXX}: Highlight extraction,
      train split (N=XXX), validation split (N=XXX)  
\item arXiv abstracts~\citep{XXX}: Scientific paper abstracts,
      train split (N=XXX), validation split (N=XXX)
\end{itemize}
Total: XXX training examples, XXX validation examples.

\textbf{Evaluation:} CNN/DailyMail v3.0.0~\citep{XXX}, test 
split (N=11,490), held-out from training to measure generalization.

\textbf{Critical:} CNN/DailyMail is NOT used during training, 
ensuring evaluation measures out-of-distribution generalization.

\subsection{Training Procedure}
\paragraph{Optimization.}
We fine-tune both baseline and monotonic models using AdamW 
with learning rate 5e-5, weight decay 0.01, and batch size 4 
with gradient accumulation over 1 step. Gradients are clipped 
to maximum norm 1.0. We use a linear warmup schedule over 
the first 10% (baseline) or 15% (monotonic) of training steps,
followed by linear decay to zero.

\paragraph{Convergence.}
The baseline model trains for 5 epochs. The monotonic model 
receives 7 epochs and extended warmup (15% vs 10%) to account 
for the constrained parameter space, which we observe empirically 
to require additional optimization steps to reach comparable 
validation loss. Both models are selected based on best 
validation loss.

\textbf{Note:} This hyperparameter difference should be addressed
as it confounds the comparison. Ideally, train both to convergence
or use early stopping with identical patience.

\subsection{Decoding}
All models use identical decoding parameters: beam search with 
4 beams, length penalty 1.2, minimum length 10 tokens, maximum 
length 128 tokens, and no_repeat_ngram_size=3 to prevent 
repetition. These parameters are fixed across all models and 
all evaluation sets to ensure fair comparison.

\subsection{Attack Evaluation Protocol}
\paragraph{Trigger Optimization.}
Universal adversarial triggers are optimized on CNN/DM validation
set (N=XXX examples, disjoint from test set). Triggers consist 
of 5 tokens selected to maximize average cross-entropy loss when
prepended to inputs. We use greedy coordinate ascent with 3 
random initializations and 50 iterations, selecting candidates 
from a vocabulary of 1,000 high-frequency tokens (excluding 
special tokens and punctuation).

\paragraph{Attack Evaluation.}
Optimized triggers are evaluated on CNN/DM test set (N=XXX, 
held-out from trigger optimization). For HotFlip attacks, we 
select up to 5 token positions with highest embedding gradient 
magnitude and replace each with the candidate token that 
maximizes dot product with the gradient direction. Candidates 
are drawn from the full vocabulary (32,000 tokens for T5).

\paragraph{Metrics.}
For each attack, we report:
\begin{itemize}
\item ROUGE-L degradation: $(ROUGE_{clean} - ROUGE_{attack}) / ROUGE_{clean}$
\item Success rate: \% of examples with >10\% degradation
\item Mean loss increase: $\mathbb{E}[\ell_{attack} - \ell_{clean}]$
\end{itemize}
Statistical significance is assessed using paired t-tests 
(same examples across models) with Bonferroni correction.
```

## REPRODUCIBILITY CHECKLIST

Add to appendix or methods:

```latex
\subsection{Reproducibility Checklist}
\begin{itemize}
\item Model architecture: T5-small, 60M parameters
\item Training data: DialogSum + HighlightSum + arXiv (XXX examples)
\item Evaluation data: CNN/DM v3.0.0 test (11,490 examples)
\item Hyperparameters: See Table~X
\item Random seeds: 42 (all libraries)
\item Hardware: 1x NVIDIA A100 40GB
\item Training time: 8h (baseline), 11h (monotonic)
\item Software: PyTorch 2.0, Transformers 4.30, Python 3.10
\item Code: Available at [ANONYMIZED FOR REVIEW]
\item Checkpoints: Available at [ANONYMIZED FOR REVIEW]
\end{itemize}
```

## STATISTICAL RIGOR ISSUES

### Confidence Intervals
**PROBLEM:** Bootstrap CIs with n=200 are unreliable.

**FIX:**
- Increase to n≥1000
- Or use analytical CIs if distributional assumptions justified
- Report CI widths to show precision

### Multiple Comparisons
**MISSING:** With 3 models and multiple metrics (R-1, R-2, R-L, multiple attacks), you're doing MANY comparisons.

**NEEDED:**
- Bonferroni or FDR correction
- Clear specification of primary vs secondary outcomes
- Possibly pre-registration of hypotheses

### Effect Sizes
**MISSING:** p-values without effect sizes are incomplete.

**NEEDED:** Report Cohen's d or similar for all comparisons.

## CRITICAL MISSING SECTIONS

### 4.X Limitations and Scope
**NEEDED:**
```latex
\subsection{Limitations}
Our evaluation is limited to T5-small (60M parameters) and 
summarization tasks. Scaling to larger models (T5-base, T5-large)
and other sequence-to-sequence tasks (translation, question 
answering) remains future work. Additionally, our evaluation 
uses 200-sample subsets rather than full test sets due to 
computational constraints, which may limit statistical power.

Furthermore, monotonicity is enforced only on FFN sublayers;
the full Transformer is not globally monotone due to attention,
LayerNorm, and residual connections. The degree to which partial
monotonicity provides robustness guarantees is an open question.
```

### 5.X Computational Cost
**NEEDED:**
```latex
\subsection{Computational Overhead}
The softplus parameterization introduces minimal computational
overhead: approximately X% increase in training time and Y% 
increase in inference latency compared to baseline, measured
across 1,000 examples on a single A100 GPU.
```

## RECOMMENDED ADDITIONS TO APPENDIX

### A. Complete Hyperparameter Tables
### B. Full Dataset Statistics
### C. Additional Attack Evaluation Details
### D. Convergence Analysis
### E. Ablation Study Results
### F. Full Statistical Test Results (all p-values, effect sizes)
### G. Error Analysis (failure case examples)

## PRIORITY RANKING

### Must Fix Before Submission:
1. ✅ Fair comparison: Train baseline for same epochs OR justify difference with ablation
2. ✅ Sample size: Use full test sets (n=11,490) or justify small sample
3. ✅ Add clean performance table (ROUGE scores without attacks)
4. ✅ Add UAT results (currently described but not shown)
5. ✅ Fix statistical testing: paired t-tests, multiple comparison correction
6. ✅ Complete dataset details: exact splits, sizes, preprocessing
7. ✅ Complete reproducibility details: seeds, hardware, software versions

### Should Fix:
8. Ablation studies (constraint location, initialization, optimization method)
9. Computational cost analysis
10. Error analysis with examples
11. Comparison to other defense methods
12. Scaling experiments (T5-base at minimum)

### Nice to Have:
13. Transfer learning analysis
14. Theoretical analysis of robustness
15. Certified robustness bounds
16. Multi-task evaluation

## SUMMARY OF METHODS REFINEMENTS

The paper has solid theoretical motivation but the empirical methods have serious gaps that would likely result in rejection at a top venue like ICML. The critical issues are:

1. **Unfair comparison** (different training epochs)
2. **Insufficient sample size** (n=200 is too small)
3. **Missing clean performance results**
4. **Missing UAT results** (described but not shown)
5. **Inadequate statistical rigor**
6. **Incomplete methodological details**

These must be addressed to make the paper publishable at a top-tier venue.

## RECOMMENDED IMMEDIATE ACTIONS

1. Re-run experiments with:
   - Same epochs for baseline and monotonic (5 or 7 for both)
   - Full test sets (n≥1000, ideally full 11,490)
   - Report clean ROUGE scores
   - Complete UAT evaluation

2. Add tables:
   - Table 1: Hyperparameters
   - Table 2: Clean performance
   - Table 3: UAT results  
   - Table 4: Transfer matrix
   - Table 5: Ablation studies

3. Expand methods section from 0.5 pages to 2-3 pages with complete details

4. Add proper statistical analysis with effect sizes and correction for multiple comparisons

Without these changes, the paper is not ready for submission to ICML or similar venues.
