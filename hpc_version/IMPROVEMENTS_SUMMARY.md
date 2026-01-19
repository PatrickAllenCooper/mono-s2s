# Improvements Summary: Reducing Clean Performance Gap & Reintroducing Datasets

**Date**: December 30, 2025  
**Goal**: Reduce monotonic model clean performance gap and reintroduce XSUM/SAMSum datasets

---

## Motivation

Previous results showed:
- **Strong robustness gains**: Monotonic T5 reduced HotFlip degradation by ~50% (8.06% vs 16.35%)
- **Clean performance gap**: ROUGE-L dropped ~4% (0.2577 → 0.2473)
- **Missing datasets**: XSUM and SAMSum were disabled due to loading issues

**Objectives**:
1. Reduce or eliminate the clean performance trade-off
2. Safely reintroduce XSUM and SAMSum with robust error handling

---

## Changes Made

### 1. **Monotonic-Specific Hyperparameters** 

**Rationale**: Constrained parameter space needs more training time and careful warmup

```python
# In experiment_config.py
MONOTONIC_NUM_EPOCHS = 7          # Was: 5 (baseline)
MONOTONIC_WARMUP_RATIO = 0.15     # Was: 0.10 (baseline)
MONOTONIC_LEARNING_RATE = 5e-5    # Same as baseline
```

**Expected impact**: 
- **+40% more training** (7 vs 5 epochs) for better convergence
- **+50% more warmup** (0.15 vs 0.10) for softplus stability
- Estimated **+2-3 hours** training time (~14-15 hrs vs ~11 hrs)

---

### 2. **Improved Decoding Parameters**

**Rationale**: Better generation quality reduces ROUGE degradation

```python
# In experiment_config.py
DECODE_LENGTH_PENALTY = 1.2      # Was: 1.0 (neutral)
DECODE_MAX_NEW_TOKENS = 80       # Was: 64 (capped)
```

**Expected impact**:
- **More complete summaries** with better coverage
- **Reduces brevity penalty** issues
- **~1-2 ROUGE point gain** on all models

---

### 3. **Better Softplus Initialization**

**Rationale**: Preserve pretrained knowledge while enforcing W ≥ 0

**Implementation** (`common_utils.py`):
```python
class NonNegativeParametrization(nn.Module):
    def right_inverse(self, W):
        """Initialize V from pretrained W to preserve learned features."""
        eps = 1e-4
        W_abs = torch.abs(W) + eps
        V = torch.log(torch.exp(W_abs) - 1.0 + eps)
        return V
```

**Expected impact**:
- **Minimizes disruption** to pretrained weights
- **Faster convergence** from better initialization
- **~1-2 ROUGE point gain** through knowledge preservation

---

### 4. **Dataset Retry Logic**

**Rationale**: Robust loading prevents pipeline failures

**Implementation** (`common_utils.py`):
```python
def load_dataset_split(..., max_retries=3, retry_delay=10):
    # Retry with exponential backoff
    # Graceful fallback if DATASET_ALLOW_PARTIAL=True
    # Clear error reporting
```

**Configuration** (`experiment_config.py`):
```python
DATASET_MAX_RETRIES = 3
DATASET_RETRY_DELAY = 10  # seconds
DATASET_ALLOW_PARTIAL = True  # Continue if some datasets fail
```

**Expected impact**:
- **Handles transient HuggingFace API failures**
- **Graceful degradation** if datasets unavailable
- **No pipeline interruptions**

---

### 5. **XSUM and SAMSum Reintroduction**

**Implementation** (`stage_1_prepare_data.py`):
```python
# XSUM - with retry logic and graceful fallback
logger.log("Loading XSUM test...")
xsum_texts, xsum_sums = load_dataset_split("xsum", "test", ...)
if len(xsum_texts) == 0:
    logger.log("  ⚠️  XSUM dataset could not be loaded, will be skipped")

# SAMSum - with retry logic and graceful fallback  
logger.log("Loading SAMSum test...")
samsum_texts, samsum_sums = load_dataset_split("samsum", "test", ...)
if len(samsum_texts) == 0:
    logger.log("  ⚠️  SAMSum dataset could not be loaded, will be skipped")
```

**Expected impact**:
- **Complete test coverage** across 3 datasets (if available)
- **Scientific completeness** per README specification
- **No failures** even if datasets temporarily unavailable

---

## Expected Results

### Clean Performance Improvement

**Conservative estimate** (combining improvements):

| Metric | Baseline T5 | Monotonic T5 (OLD) | Monotonic T5 (NEW) | Improvement |
|--------|-------------|--------------------|--------------------|-------------|
| ROUGE-L | 0.2577 | 0.2473 (-4.0%) | **0.2540-0.2565** | **+2.7-3.7%** |
| ROUGE-1 | 0.3154 | 0.3002 (-4.8%) | **0.3075-0.3110** | **+2.4-3.4%** |
| ROUGE-2 | 0.1182 | 0.1049 (-11.3%) | **0.1095-0.1135** | **+4.4-8.2%** |

**Sources of improvement**:
1. Better initialization: +1-2 ROUGE points
2. More training: +0.5-1 ROUGE point  
3. Better decoding: +1-1.5 ROUGE points
4. More warmup: +0.3-0.5 ROUGE point

**Result**: Gap reduced from **-4%** to **-1.5% to -0.5%**

---

### Robustness (Should Remain Strong)

| Metric | Monotonic T5 (OLD) | Monotonic T5 (NEW) | Note |
|--------|--------------------|--------------------|------|
| HotFlip Degradation | 8.06% | **~7-9%** | Maintained |
| HotFlip Success Rate | 26.0% | **~23-28%** | Maintained |
| Loss Increase | +0.2302 | **~0.22-0.24** | Maintained |

**Longer training may slightly improve robustness further**

---

## Testing Before Full Run

**New test script**: `hpc_version/test_improvements.py`

```bash
cd hpc_version
python test_improvements.py
```

**Tests**:
1. Configuration parameters
2. Dataset loading (CNN/DM, XSUM, SAMSum) with retry
3. Softplus initialization preservation  
4. Monotonic model creation

**Run this BEFORE** submitting the full pipeline to catch issues early!

---

## Files Modified

### Configuration
- `hpc_version/configs/experiment_config.py`
  - Added `MONOTONIC_NUM_EPOCHS`, `MONOTONIC_WARMUP_RATIO`, `MONOTONIC_LEARNING_RATE`
  - Updated `DECODE_LENGTH_PENALTY` and `DECODE_MAX_NEW_TOKENS`
  - Added `DATASET_MAX_RETRIES`, `DATASET_RETRY_DELAY`, `DATASET_ALLOW_PARTIAL`
  - Reactivated XSUM and SAMSum in `TEST_DATASETS`

### Core Utilities
- `hpc_version/utils/common_utils.py`
  - Enhanced `NonNegativeParametrization` with `right_inverse()` initialization
  - Updated `load_dataset_split()` with retry logic and graceful fallback
  - Updated `make_model_monotonic()` to use improved initialization

### Training Scripts
- `hpc_version/scripts/stage_3_train_monotonic.py`
  - Updated to use `MONOTONIC_NUM_EPOCHS`, `MONOTONIC_WARMUP_RATIO`
  
### Data Preparation
- `hpc_version/scripts/stage_1_prepare_data.py`
  - Reactivated XSUM and SAMSum loading with error handling
  - Updated summary reporting

### New Files
- `hpc_version/test_improvements.py` - Pre-flight validation script
- `hpc_version/IMPROVEMENTS_SUMMARY.md` - This document

---

## Next Steps

### 1. **Test Locally** (Recommended)

```bash
cd hpc_version
python test_improvements.py
```

Expected: All tests pass, datasets load (or gracefully skip)

---

### 2. **Run Full Pipeline**

```bash
cd hpc_version
./run_all.sh 42  # Or use default seed
```

**Expected runtime changes**:
- Stage 3 (Monotonic training): ~14-15 hrs (was ~11 hrs)
- Stage 4 (Evaluation): +1-2 hrs if XSUM/SAMSum available
- Total: ~30-35 hrs (was ~28-30 hrs)

---

### 3. **Monitor Results**

**Check clean performance**:
```bash
tail -f $SCRATCH/mono_s2s_work/stage_logs/stage_4_evaluate.log
```

**Look for**:
- Monotonic T5 ROUGE-L: **Target ≥ 0.254** (currently 0.2473)
- Gap vs Baseline: **Target ≤ 1.5%** (currently 4.0%)

**Check robustness** (should remain strong):
```bash
tail -f $SCRATCH/mono_s2s_work/stage_logs/stage_6_hotflip.log
```

**Look for**:
- Monotonic degradation: **Target ≤ 10%** (currently 8.06%)
- Baseline degradation: ~16-17% (reference)

---

## Success Criteria

### Minimum Success
- Pipeline completes without errors
- Robustness gains maintained (degradation ≤ 10%)
- Clean performance gap reduced to ≤ 2.5%

### Full Success  
- All three datasets (CNN/DM, XSUM, SAMSum) evaluated
- Clean performance gap reduced to ≤ 1.5%
- Robustness still strong across all datasets
- Hypothesis confirmed with **minimal trade-off**

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| XSUM/SAMSum still fail to load | Medium | Graceful fallback, continue with CNN/DM |
| Longer training doesn't help | Low | 2 extra epochs is conservative |
| Robustness degrades | Very Low | Constraints unchanged, only training longer |
| Decoding changes hurt metrics | Low | 1.2 penalty is standard, 80 tokens reasonable |

---

## Rollback Plan

If results are worse, revert to previous config:

```python
# In experiment_config.py
MONOTONIC_NUM_EPOCHS = 5
MONOTONIC_WARMUP_RATIO = 0.1
DECODE_LENGTH_PENALTY = 1.0
DECODE_MAX_NEW_TOKENS = 64
```

And in `stage_1_prepare_data.py`, set XSUM/SAMSum to empty lists.

---

## References

- **README.md**: Project specification (3 test datasets required)
- **Previous results**: Stage 7 output showing 4% gap
- **Softplus properties**: inverse_softplus(x) = log(exp(x) - 1)
- **HuggingFace docs**: Dataset loading best practices

---

## Completion Checklist

- [x] Updated configuration with monotonic-specific hyperparameters
- [x] Improved softplus initialization in common_utils.py
- [x] Added dataset retry logic with graceful fallback
- [x] Reintroduced XSUM and SAMSum in stage_1_prepare_data.py
- [x] Updated stage_3_train_monotonic.py to use new hyperparameters
- [x] Created test_improvements.py for validation
- [x] No linter errors in modified files
- [ ] **TODO**: Run test_improvements.py on HPC
- [ ] **TODO**: Submit full pipeline
- [ ] **TODO**: Analyze new results

---

Ready to proceed with testing and full pipeline run.

