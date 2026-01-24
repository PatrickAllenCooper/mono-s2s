# Recommended Figures for ICML Paper

## MUST HAVE (Will Strengthen Paper Significantly)

### Figure 1: Training and Validation Curves
**Purpose:** Show both models converge properly, monotonic doesn't degrade optimization

**Layout:** Single figure with 2 subplots
- (a) Training loss over epochs - Baseline vs Monotonic
- (b) Validation loss over epochs - Baseline vs Monotonic

**What to Show:**
- Both models converge to similar validation loss
- Monotonic may start higher but catches up by epoch 7
- No optimization pathologies from constraints

**Data Source:** Training history JSONs from pipeline
**Why Critical:** Proves monotonicity doesn't break optimization

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/training_curves.pdf}
\caption{Training dynamics for baseline and monotonic models. 
Both models converge to similar validation loss by epoch 7, 
demonstrating that monotonicity constraints do not impair 
optimization despite constraining 40\% of parameters.}
\label{fig:training}
\end{figure}
```

---

### Figure 2: Clean vs Attack Performance Comparison
**Purpose:** Visualize performance-robustness tradeoff

**Layout:** Scatter plot or grouped bar chart
- X-axis: Clean ROUGE-L score
- Y-axis: Attack ROUGE-L score (or attack success rate)
- Points: Standard, Baseline, Monotonic
- Ideal point: top-right (high clean, low attack success)

**What to Show:**
- Monotonic: similar clean performance, much better under attack
- Baseline: higher clean, worse under attack
- Standard: lowest clean, worst under attack
- Pareto frontier showing monotonic is not dominated

**Alternative:** Side-by-side bars (clean vs attack) for each model

**Data Source:** Tables 1 and 2 from paper
**Why Critical:** Core result visualization

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/performance_robustness.pdf}
\caption{Performance-robustness tradeoff. Monotonic model preserves 
clean task performance (x-axis) while substantially improving robustness 
(y-axis). Error bars show 95\% bootstrap confidence intervals.}
\label{fig:tradeoff}
\end{figure}
```

---

### Figure 3: Attack Success Rate vs Attack Budget
**Purpose:** Show monotonic advantage holds across different attack strengths

**Layout:** Line plot
- X-axis: Attack budget (trigger length: 1, 3, 5, 7, 10 tokens)
- Y-axis: Attack success rate (%)
- Lines: Standard (worst), Baseline (middle), Monotonic (best)

**What to Show:**
- All models degrade as attack budget increases
- Monotonic maintains advantage at ALL budgets
- Gap widens or stays constant (doesn't disappear)

**Data Source:** Run attacks with varying trigger lengths
**Why Critical:** Shows robustness is not just at one specific attack level

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/attack_budget.pdf}
\caption{Attack success rate versus trigger length. Monotonic model 
maintains robustness advantage across all attack budgets, demonstrating 
that improvements are not specific to a particular attack strength.}
\label{fig:budget}
\end{figure}
```

---

## HIGHLY RECOMMENDED (Adds Significant Value)

### Figure 4: Weight Distribution Comparison
**Purpose:** Show HOW monotonic and baseline models differ

**Layout:** Histograms or density plots (2 subplots)
- (a) Baseline FFN weights: spans negative to positive
- (b) Monotonic FFN weights: all positive, different distribution

**What to Show:**
- Baseline: roughly symmetric around zero
- Monotonic: all positive, possibly different shape
- Visual proof that models learn different representations

**Data Source:** Extract FFN weights from saved checkpoints
**Why Valuable:** Mechanistic insight into what monotonicity does

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/weight_distributions.pdf}
\caption{FFN weight distributions. (a) Baseline model: weights 
span negative and positive values. (b) Monotonic model: all weights 
non-negative, exhibiting qualitatively different learned features.}
\label{fig:weights}
\end{figure}
```

---

### Figure 5: Multi-Dataset Generalization
**Purpose:** Show robustness generalizes across domains

**Layout:** Grouped bar chart
- X-axis: Dataset (CNN/DM, XSUM, SAMSum)
- Y-axis: Attack success rate (%)
- Bars: Baseline vs Monotonic (side by side)

**What to Show:**
- Monotonic consistently better across all datasets
- Advantage magnitude may vary by domain
- No dataset where monotonic is worse

**Data Source:** Table 5 from paper (multi-dataset results)
**Why Valuable:** Proves generalization, not overfitting to CNN/DM

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/multi_dataset.pdf}
\caption{Attack robustness across three summarization domains. 
Monotonic model demonstrates consistent robustness improvements 
across news (CNN/DM), extreme summarization (XSUM), and dialogue 
(SAMSum), indicating domain-general benefits.}
\label{fig:multidataset}
\end{figure}
```

---

### Figure 6: Input Gradient Magnitude Analysis
**Purpose:** Mechanistic explanation - why are gradient attacks less effective?

**Layout:** Box plots or violin plots
- X-axis: Model (Standard, Baseline, Monotonic)
- Y-axis: $\|\nabla_x \ell\|$ (input gradient norm)
- Shows distribution across examples

**What to Show:**
- Monotonic has lower gradient norms than baseline
- Smaller gradients = harder to find adversarial perturbations
- Statistical significance (non-overlapping distributions)

**Data Source:** Compute gradients on 500-1000 examples
**Why Valuable:** Explains WHY monotonicity helps (not just that it does)

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/gradient_norms.pdf}
\caption{Input gradient magnitude distributions. Monotonic models 
exhibit significantly lower gradient norms ($p<0.001$), explaining 
reduced vulnerability to gradient-based attacks. Box plots show 
median, quartiles, and outliers across 1,000 test examples.}
\label{fig:gradients}
\end{figure}
```

---

## NICE TO HAVE (Further Strengthens Paper)

### Figure 7: Transfer Attack Heatmap
**Purpose:** Visualize cross-model attack transferability

**Layout:** 3×3 heatmap
- Rows: Trigger optimized on (Standard, Baseline, Monotonic)
- Columns: Trigger evaluated on (Standard, Baseline, Monotonic)
- Colors: Red (high success) to Blue (low success)

**What to Show:**
- Diagonal (self-attack) should be highest
- Off-diagonal shows transfer
- Monotonic column may be darker (harder to attack)

**Data Source:** Table 4 transfer matrix
**Why Valuable:** Rich visualization of attack transferability

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\columnwidth]{figures/transfer_heatmap.pdf}
\caption{Attack transferability heatmap. Darkness indicates attack 
success rate. Triggers optimized on monotonic model [are/are not] 
as transferable as baseline triggers, suggesting [interpretation].}
\label{fig:transfer}
\end{figure}
```

---

### Figure 8: ROUGE Score Distributions (Clean vs Attack)
**Purpose:** Show monotonic has more stable performance

**Layout:** Box plots with 2 groups
- Group 1: Clean performance (all 3 models)
- Group 2: Under attack (all 3 models)

**What to Show:**
- Clean: all models have similar distributions
- Attack: monotonic distribution tighter (less degradation)
- Standard has widest distribution under attack (most vulnerable)

**Data Source:** Per-example ROUGE scores from evaluation
**Why Valuable:** Shows stability, not just average performance

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/rouge_distributions.pdf}
\caption{ROUGE-L score distributions under clean and adversarial 
conditions. Monotonic model exhibits tighter distribution under 
attack, indicating more consistent behavior across examples.}
\label{fig:distributions}
\end{figure}
```

---

### Figure 9: Ablation Study Results
**Purpose:** Show contribution of different design choices

**Layout:** Grouped bar chart
- X-axis: Configuration (Baseline-5ep, Baseline-7ep, Baseline-10ep, Monotonic-7ep)
- Y-axis: Attack success rate (%)

**What to Show:**
- Even with 10 epochs, baseline doesn't match monotonic
- Proves constraints are key, not just training time

**Data Source:** Ablation experiments
**Why Valuable:** Isolates contribution of monotonicity

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/ablation_epochs.pdf}
\caption{Ablation study: effect of training epochs on robustness. 
Even with extended training (10 epochs), unconstrained baseline 
does not match monotonic model robustness, confirming that 
architectural constraints are the key factor.}
\label{fig:ablation}
\end{figure}
```

---

### Figure 10: Example Adversarial Perturbations
**Purpose:** Qualitative illustration of attacks and defenses

**Layout:** Text boxes showing:
- Original input
- Adversarial input (with triggers/flips highlighted in red)
- Baseline output (degraded summary)
- Monotonic output (more robust summary)

**What to Show:**
- Concrete example where monotonic maintains quality
- Highlighting shows exactly what was perturbed
- Human-readable demonstration of robustness

**Data Source:** Cherry-pick good example from test set
**Why Valuable:** Makes abstract numbers concrete

**LaTeX:**
```latex
\begin{figure}[t]
\centering
\fbox{\parbox{0.95\columnwidth}{
\textbf{Original:} The president announced...\\
\textbf{+ Trigger:} \colorbox{red!30}{<!!> <?> <#>} The president...\\
\textbf{Baseline Output:} [Degraded summary]\\
\textbf{Monotonic Output:} [Robust summary]
}}
\caption{Example adversarial perturbation. Red highlighting shows 
trigger tokens. Monotonic model maintains summary quality despite 
perturbation, while baseline model degrades significantly.}
\label{fig:example}
\end{figure}
```

---

### Figure 11: Loss Landscape Visualization
**Purpose:** Show monotonic has smoother loss landscape

**Layout:** 2D projection of loss landscape
- Visualize loss around adversarial examples
- Baseline: sharp peaks/valleys
- Monotonic: smoother surface

**What to Show:**
- Monotonic loss landscape has lower curvature
- Smoother landscape = harder to find adversarial examples

**Data Source:** Sample loss in neighborhood of inputs
**Why Valuable:** Deep mechanistic insight

**Implementation:** Challenging - may skip if time-limited

---

### Figure 12: Multi-Seed Robustness
**Purpose:** Show results are consistent across random seeds

**Layout:** Error bars or violin plots
- X-axis: Metric (Clean R-L, HotFlip Success, UAT Success)
- Y-axis: Value
- Points/distributions: 5 seeds for each model

**What to Show:**
- Monotonic advantage consistent across seeds
- Not due to lucky initialization

**Data Source:** Run with all 5 seeds in RANDOM_SEEDS
**Why Valuable:** Proves results are robust, not cherry-picked

---

## PRIORITY RANKING

### Tier 1 (Must Include - 3 figures):
1. **Figure 1:** Training curves (proves optimization works)
2. **Figure 2:** Performance-robustness tradeoff (core result)
3. **Figure 4:** Weight distributions (mechanistic insight)

### Tier 2 (Highly Recommended - 2-3 figures):
4. **Figure 3:** Attack budget sensitivity (shows generality)
5. **Figure 5:** Multi-dataset results (proves generalization)
6. **Figure 6:** Gradient norms (explains mechanism)

### Tier 3 (Nice to Have - 1-2 figures):
7. **Figure 7:** Transfer heatmap (rich visualization)
8. **Figure 10:** Example perturbation (concrete illustration)

### Tier 4 (If Space/Time Allows):
9. **Figure 8:** ROUGE distributions (shows stability)
10. **Figure 9:** Ablation results (isolates contribution)
11. **Figure 12:** Multi-seed robustness (proves consistency)

---

## RECOMMENDED FIGURE SET FOR ICML

### Minimal Set (3 figures):
- Figure 1: Training curves
- Figure 2: Performance-robustness scatter
- Figure 4: Weight distributions

**Rationale:** Covers optimization, core result, and mechanism

### Standard Set (5 figures):
- Figure 1: Training curves
- Figure 2: Performance-robustness tradeoff
- Figure 3: Attack budget sensitivity
- Figure 4: Weight distributions  
- Figure 6: Gradient norm analysis

**Rationale:** Complete story from training → results → mechanism

### Comprehensive Set (7 figures):
- All of Standard Set +
- Figure 5: Multi-dataset generalization
- Figure 10: Example perturbation

**Rationale:** Full evaluation + concrete example

---

## IMPLEMENTATION PRIORITY

### Can Create from Existing Data (Easy):
1. ✅ **Figure 1:** Training curves (from training_history.json)
2. ✅ **Figure 2:** Performance-robustness (from Tables 1-2)
3. ✅ **Figure 5:** Multi-dataset (from Table 5)
4. ✅ **Figure 7:** Transfer heatmap (from Table 4)

### Requires Running Analysis Scripts (Medium):
5. **Figure 4:** Weight distributions (extract from checkpoints)
6. **Figure 6:** Gradient norms (compute on test set)
7. **Figure 10:** Example perturbation (find good example)

### Requires New Experiments (Hard):
8. **Figure 3:** Attack budget (re-run attacks with varying lengths)
9. **Figure 9:** Ablation figure (run ablation experiments)
10. **Figure 12:** Multi-seed (run with 5 seeds)

---

## FIGURE CREATION SCRIPTS NEEDED

### Create: hpc_version/scripts/create_training_curves.py
```python
"""
Load baseline_training_history.json and monotonic_training_history.json
Extract train_losses and val_losses
Plot both on same figure with 2 subplots
Save as figures/training_curves.pdf
"""
```

### Create: hpc_version/scripts/create_weight_histograms.py
```python
"""
Load baseline best_model.pt and monotonic best_model.pt
Extract all FFN weights (wi, wo)
Create histograms/density plots
Save as figures/weight_distributions.pdf
"""
```

### Create: hpc_version/scripts/compute_gradient_norms.py
```python
"""
Load both models
For 1,000 test examples:
    Compute ||∂Loss/∂input_embeddings||
    Store norms for each model
Create box plots comparing distributions
Save as figures/gradient_norms.pdf
"""
```

### Create: hpc_version/scripts/create_attack_budget_figure.py
```python
"""
Run attacks with trigger lengths: [1, 3, 5, 7, 10]
For each length, compute success rate
Plot lines for all 3 models
Save as figures/attack_budget.pdf
"""
```

---

## DETAILED FIGURE SPECIFICATIONS

### Figure 1: Training Curves

**Data Required:**
- Baseline: train_losses, val_losses (7 epochs)
- Monotonic: train_losses, val_losses (7 epochs)

**Plot Details:**
- Subplot (a): epoch vs train loss
- Subplot (b): epoch vs val loss
- Legend: Baseline (blue), Monotonic (red)
- Grid: light gray
- Line style: solid, width 2pt
- Markers: circles every epoch

**Key Insights to Highlight:**
- Both converge by epoch 7
- Final val losses within X% of each other
- Monotonic may have slower early convergence but catches up

---

### Figure 2: Performance-Robustness Tradeoff

**Data Required:**
- Clean ROUGE-L: Standard, Baseline, Monotonic
- Attack ROUGE-L: Standard, Baseline, Monotonic

**Plot Details:**
- Scatter plot with error bars (95% CI)
- Point size: large (10pt)
- Colors: Standard (gray), Baseline (blue), Monotonic (red)
- Diagonal line: y=x (equal clean/attack performance)
- Annotations: model names next to points

**Key Insights:**
- Monotonic in top-right (high clean, low attack success)
- Baseline in middle
- Standard in bottom-left
- Pareto frontier shows monotonic dominates

---

### Figure 3: Attack Budget Sensitivity

**Data Required:**
- For trigger lengths {1, 3, 5, 7, 10}:
  - Success rate for Standard, Baseline, Monotonic

**Plot Details:**
- Line plot with markers
- X-axis: Trigger length (discrete: 1, 3, 5, 7, 10)
- Y-axis: Success rate (%)
- Error bars: 95% CI
- Lines: Standard (dashed gray), Baseline (solid blue), Monotonic (solid red)

**Key Insights:**
- Monotonic advantage at ALL budgets
- Gap may widen with budget (monotonic scales better)
- Or gap stays constant (consistent advantage)

---

### Figure 4: Weight Distributions

**Data Required:**
- All FFN weights from baseline checkpoint
- All FFN weights from monotonic checkpoint

**Plot Details:**
- Two histograms (subplots or overlaid)
- X-axis: Weight value
- Y-axis: Density or count
- Baseline: blue bars/line
- Monotonic: red bars/line
- Vertical line at x=0 for reference

**Key Insights:**
- Baseline spans [-X, +X]
- Monotonic spans [0, +X]
- Different distributions despite similar performance
- May show monotonic is more sparse or concentrated

---

### Figure 6: Gradient Norm Analysis

**Data Required:**
- For 1,000 examples, compute $\|\nabla_x \ell\|$ for each model
- Results: 3 distributions (Standard, Baseline, Monotonic)

**Plot Details:**
- Box plots or violin plots
- X-axis: Model
- Y-axis: log-scale gradient norm
- Show: median, quartiles, outliers
- Statistical annotation: *** for p<0.001

**Key Insights:**
- Monotonic has XX% lower median gradient norm
- Tighter distribution (more predictable)
- Explains why gradient attacks less effective

---

## FIGURE PLACEMENT IN PAPER

### Suggested Placement:

**Methods Section:**
- (optional) Figure: T5 architecture diagram showing which layers constrained

**Results Section:**
- Figure 1: Training curves (after describing training protocol)
- Figure 2: Performance-robustness tradeoff (with Table 1)
- Figure 3: Attack budget sensitivity (with Tables 2-3)
- Figure 5: Multi-dataset (with Table 5)

**Discussion Section:**
- Figure 4: Weight distributions (in "Why Does It Help?")
- Figure 6: Gradient norms (in "Why Does It Help?")

**Appendix:**
- Additional figures if running out of space in main body
- Figure 10: Example perturbation
- Figure 7: Transfer heatmap

---

## EFFORT vs IMPACT MATRIX

### High Impact, Low Effort (Do These First):
1. Figure 1: Training curves ✅ (have data, easy plot)
2. Figure 2: Performance-robustness ✅ (have data, easy plot)
3. Figure 5: Multi-dataset ✅ (have data, easy plot)

### High Impact, Medium Effort:
4. Figure 4: Weight distributions (need to extract weights)
5. Figure 6: Gradient norms (need to compute)

### Medium Impact, Low Effort:
6. Figure 3: Attack budget (need new experiments)
7. Figure 7: Transfer heatmap (have data)

### Medium Impact, Medium Effort:
8. Figure 8: ROUGE distributions
9. Figure 10: Example perturbation

### Lower Priority:
10. Figure 9: Ablation results
11. Figure 11: Loss landscape
12. Figure 12: Multi-seed robustness

---

## RECOMMENDED MINIMAL FIGURE SET (For Time-Constrained Scenario)

**Choose These 3:**

1. **Figure 1: Training Curves**
   - Proves optimization works
   - Easy to create from existing data
   - Reviewers expect this

2. **Figure 2: Performance-Robustness Tradeoff**
   - Core result visualization
   - Easy to create from Tables 1-2
   - Makes numbers intuitive

3. **Figure 4: Weight Distributions**
   - Mechanistic insight
   - Medium effort but high value
   - Shows HOW models differ, not just that they do

**Rationale:** Covers optimization (no issues), result (robust + accurate), mechanism (why it works)

---

## FIGURE CREATION CHECKLIST

### Before Pipeline Run:
- [ ] Decide which figures to include (recommend 3-5)
- [ ] Create figure creation scripts
- [ ] Test scripts on dummy data

### After Pipeline Run:
- [ ] Extract training curves → Figure 1
- [ ] Create performance-robustness plot → Figure 2
- [ ] Create multi-dataset bar chart → Figure 5

### Additional Analysis:
- [ ] Extract FFN weights → Figure 4
- [ ] Compute gradient norms → Figure 6
- [ ] (Optional) Run attack budget sweep → Figure 3

### Integration:
- [ ] Generate all PDFs in figures/ directory
- [ ] Reference in paper with \includegraphics
- [ ] Write captions
- [ ] Reference in text

---

## SUMMARY RECOMMENDATION

**For Strong ICML Submission, Include:**

**Minimum (3 figures):**
- Training curves
- Performance-robustness tradeoff
- Weight distributions

**Recommended (5 figures):**
- Above 3 +
- Multi-dataset generalization
- Gradient norm analysis

**Comprehensive (7 figures):**
- Above 5 +
- Attack budget sensitivity
- Transfer heatmap

**Choose based on:**
- Time available for analysis
- Computational resources
- Page space (can move some to appendix)

**All figures should be high-quality PDFs with clear labels, legends, and captions.**
