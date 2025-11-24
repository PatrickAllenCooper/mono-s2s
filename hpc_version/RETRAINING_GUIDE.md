# Retraining Guide: Improved Hyperparameters

This guide walks through examining dataset construction and retraining with improved hyperparameters.

## Changes Made

### 1. Improved Hyperparameters

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `LEARNING_RATE` | 3e-5 | **5e-5** | Better convergence |
| `NUM_EPOCHS` | 3 | **5** | More complete training |
| `DECODE_LENGTH_PENALTY` | 1.0 | **2.0** | Reduce over-generation |

### 2. New Diagnostic Tools

- `test_dataset_loading.py` - Test all datasets before training

---

## Step-by-Step Retraining Process

### Step 1: Examine Dataset Construction

First, let's understand what went wrong with XSUM and SAMSum (they showed 0 samples).

#### On Alpine:

```bash
# Navigate to your repo
cd ~/code/mono-s2s/hpc_version  # adjust path as needed

# Pull latest code with fixes
git pull origin main

# Activate your environment
conda activate mono_s2s

# Run the diagnostic script
python test_dataset_loading.py
```

**Expected output:**
```
================================================================================
DATASET LOADING DIAGNOSTIC
================================================================================
Configuration:
  USE_FULL_TEST_SETS: False
  QUICK_TEST_SIZE: 200

Testing HuggingFace datasets library...
  ✓ datasets library imported successfully

================================================================================
TESTING INDIVIDUAL DATASETS
================================================================================

================================================================================
Testing: CNN/DailyMail
================================================================================
  Dataset: cnn_dailymail
  Config: 3.0.0
  Split: test
  Fields: article → highlights

  [1/4] Loading dataset from HuggingFace...
        ✓ Loaded 11490 total samples
  [2/4] Checking fields...
        Available fields: ['article', 'highlights', 'id']
        ✓ Text field 'article' found
        ✓ Summary field 'highlights' found
  [3/4] Extracting samples...
        ✓ Extracted 200 valid samples
  [4/4] Showing first example...
        Text (first 100 chars): LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 M...
        Summary (first 100 chars): Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday...

  ✅ CNN/DailyMail: SUCCESS (200 samples)

... (similar for XSUM and SAMSum)
```

#### If Diagnostics Fail:

**Common Issues:**

1. **Network/Firewall**: HuggingFace datasets need internet access
   - Check: `curl https://huggingface.co`
   - Solution: Run on compute node instead of login node

2. **Authentication Required**: Some datasets need HF token
   - Check error message for "authentication"
   - Solution: `huggingface-cli login`

3. **Field Names Changed**: Dataset API updated
   - Check: "Field not found" in output
   - Solution: Update field names in `experiment_config.py`

4. **Dataset Deprecated/Moved**: Dataset relocated on HuggingFace
   - Check: "Dataset not found"
   - Solution: Find new dataset path on huggingface.co

---

### Step 2: Check Previous Stage 1 Logs

```bash
# Check what happened in Stage 1
cat logs/job_1_data_*.out | tail -100

# Look specifically for errors
grep -i "error\|fail\|warning" logs/job_1_data_*.out logs/job_1_data_*.err

# Check data statistics
cat /scratch/alpine/$USER/mono_s2s_results/data_statistics.json | python -m json.tool
```

Look for lines like:
```
⚠️  Error loading EdinburghNLP/xsum (test): [error message]
⚠️  Error loading samsum (test): [error message]
```

---

### Step 3: Clean and Prepare for Retraining

Once you understand the dataset issues:

```bash
# Option A: Clean everything (recommended for fresh start)
./clean_all.sh --keep-cache  # Keeps HF cache, removes experiments

# Option B: Clean just data and evaluation (keep trained models)
rm -f /scratch/alpine/$USER/mono_s2s_work/stage_1_data_prep_complete.flag
rm -f /scratch/alpine/$USER/mono_s2s_work/stage_4_evaluate_complete.flag
rm -rf /scratch/alpine/$USER/mono_s2s_work/data_cache/
```

---

### Step 4: Run Full Pipeline with New Hyperparameters

```bash
# Ensure latest code is pulled
git pull origin main

# Verify configuration
grep "NUM_EPOCHS\|LEARNING_RATE\|LENGTH_PENALTY" configs/experiment_config.py

# Expected output:
# LEARNING_RATE = 5e-5
# NUM_EPOCHS = 5
# DECODE_LENGTH_PENALTY = 2.0

# Run full pipeline
./run_all.sh
```

**Expected timeline with new hyperparameters:**
- Stage 0 (Setup): 10 min
- Stage 1 (Data): 30 min
- Stage 2 (Baseline): **16-20 hours** (was 10-12, now 5 epochs)
- Stage 3 (Monotonic): **16-20 hours** (parallel with Stage 2)
- Stage 4 (Evaluation): 1-2 hours
- Stages 5-7 (Attacks): 3-4 hours

**Total: ~20-24 hours wall time**

---

### Step 5: Monitor Training Progress

```bash
# Check job status
squeue -u $USER

# Watch training progress (Stage 2 example)
tail -f logs/job_2_baseline_*.out

# Look for:
# - Training loss decreasing over epochs
# - Validation loss stabilizing
# - No NaN or exploding gradients
```

**Good training progress looks like:**

```
Epoch 1/5
  Train Loss: 2.4532
  Val Loss:   2.1234

Epoch 2/5
  Train Loss: 1.8765
  Val Loss:   1.9876

Epoch 3/5
  Train Loss: 1.5432
  Val Loss:   1.8234

Epoch 4/5
  Train Loss: 1.3456
  Val Loss:   1.7654

Epoch 5/5
  Train Loss: 1.2111
  Val Loss:   1.7123  ← Should be close to or better than Epoch 4
```

---

### Step 6: Interpret New Results

After Stage 4 completes, check the evaluation:

```bash
# View results
cat logs/job_4_evaluate_*.out | grep -A 10 "COMPREHENSIVE EVALUATION"
```

**Target metrics with improved hyperparameters:**

| Model | ROUGE-1 (Target) | ROUGE-2 (Target) | Length Ratio (Target) |
|-------|------------------|------------------|-----------------------|
| Standard T5 | 0.32-0.33 | 0.12 | 1.1-1.2x |
| Baseline T5 | **0.37-0.40** ⭐ | **0.15-0.18** | 1.0-1.2x |
| Monotonic T5 | **0.36-0.39** | **0.14-0.17** | 1.0-1.2x |

**Success criteria:**
1. ✅ Both fine-tuned models beat Standard T5
2. ✅ Length ratio < 1.3x (was 1.6-1.8x)
3. ✅ Monotonic within 1-2% of Baseline (proves constraints don't hurt much)

---

## Troubleshooting During Retraining

### Training Loss Not Decreasing

```python
# In experiment_config.py
LEARNING_RATE = 1e-4  # Increase further
WARMUP_RATIO = 0.15   # More warmup
```

### Out of Memory During Training

```python
# In experiment_config.py
BATCH_SIZE = 2  # Reduce from 4
GRADIENT_ACCUMULATION_STEPS = 2  # Keep effective batch size
```

### Models Still Too Verbose

```python
# In experiment_config.py
DECODE_LENGTH_PENALTY = 2.5  # Increase from 2.0
DECODE_MAX_NEW_TOKENS = 100  # Reduce from 128
```

### Training Too Slow

```bash
# Check GPU utilization
nvidia-smi

# If utilization < 80%, increase batch size:
BATCH_SIZE = 8  # If memory allows
```

---

## Expected Improvements

### Before (Initial Training)

**Problems:**
- Fine-tuned models worse than pre-trained
- Generating 1.6-1.8x reference length
- Only 3 epochs insufficient
- XSUM/SAMSum datasets failed to load

**Results:**
```
Standard T5:   ROUGE-1: 0.3265
Baseline T5:   ROUGE-1: 0.3054 ❌ (-6.5%)
Monotonic T5:  ROUGE-1: 0.2822 ❌ (-13.6%)
```

### After (With Improvements)

**Improvements:**
- 5 epochs for better convergence
- Higher learning rate (5e-5) for faster learning
- Length penalty 2.0 to control verbosity
- Dataset diagnostics to catch loading issues
- Empty dataset handling (no crashes)

**Expected Results:**
```
Standard T5:   ROUGE-1: 0.3265 (unchanged - pre-trained)
Baseline T5:   ROUGE-1: 0.3800 ✅ (+16% improvement)
Monotonic T5:  ROUGE-1: 0.3700 ✅ (+31% improvement, -2.6% vs Baseline)
```

**Key Insight:** If Monotonic is within 1-3% of Baseline, the constraints preserve performance while potentially improving robustness (measured in Stages 5-6).

---

## Quick Reference Commands

```bash
# Diagnostic
python test_dataset_loading.py

# Clean everything
./clean_all.sh --keep-cache

# Full pipeline
./run_all.sh

# Monitor
squeue -u $USER
tail -f logs/job_2_baseline_*.out

# Check results
cat logs/job_4_evaluate_*.out | grep "ROUGE-1"
```

---

## Next Steps After Retraining

1. **Analyze Results** - Compare new vs old ROUGE scores
2. **Check Adversarial Robustness** - Stages 5-6 show if constraints help
3. **Multi-Seed Runs** - Run with different seeds for statistical significance
4. **Publication** - Use `USE_FULL_TEST_SETS = True` for final results

---

**Questions?** Check logs or run diagnostics first! 🚀

