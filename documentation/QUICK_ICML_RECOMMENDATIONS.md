# Quick ICML Recommendations

## Critical Fixes (Already Done ✅)
- Baseline: 5 epochs → 7 epochs (fair comparison)
- Sample size: 200 → 11,490 (adequate power)

## Must Add (Week 1)

### Missing Results Tables:
1. **Clean performance** - ROUGE scores without attacks
2. **UAT results** - Currently described but not shown
3. **Dataset statistics** - Train/val/test splits with sizes
4. **Hyperparameters** - Complete settings table

### Methods Details:
- Change "independent t-tests" → "paired t-tests + Bonferroni"
- Add exact dataset sizes (DialogSum: 12,460 train, etc.)
- Add ROUGE implementation (rouge-score v0.1.2, stemming=True)
- Add reproducibility (seeds=42, GPU=A100, PyTorch=2.0)

## Highly Recommended (Weeks 2-3)

### Experiments:
- **Multi-seed runs** - Run with 5 seeds, report mean ± std
- **Multi-dataset** - Report CNN/DM + XSUM + SAMSum (already in pipeline!)
- **Ablation** - Baseline-10epoch to test if more training closes gap
- **T5-base** - Scale to 220M params (minimum for credibility)

### Analysis:
- **Gradient norms** - Why gradient attacks less effective?
- **Weight distributions** - How learned features differ
- **Computational cost** - Training time, inference time, memory

### Paper:
- **Discussion section** - Why it works, limitations, future work
- **Expand methods** - 0.5 pages → 2.5 pages
- **Expand results** - 0.5 pages → 3.5 pages

## Quick Wins (Use Existing Infrastructure)

You already compute these - just add to paper:
- ✅ XSUM results (in pipeline, not in paper)
- ✅ SAMSum results (in pipeline, not in paper)
- ✅ Transfer matrix (computed in stage_5, not in paper)
- ✅ Full test sets (infrastructure ready, just enable)

## ICML Acceptance Probability

- Before fixes: ~15%
- After critical fixes: ~35% ✅ (NOW)
- After must-haves: ~55% (submittable)
- After recommended: ~75% (strong)

## Timeline

**2 weeks:** Minimum viable submission (~55% chance)
**4 weeks:** Strong submission (~75% chance)
**6 weeks:** Excellent submission (~85% chance)

## Bottom Line

**Two most important things:**
1. Run pipeline and collect ALL results (clean + attacks + all datasets)
2. Expand Methods from 0.5 pages to 2.5 pages with complete details

**Everything else builds on these two foundations.**
