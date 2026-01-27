# Evaluation Timeout Postmortem and Fix

**Date**: January 27, 2026
**Job**: 23293358 (Stage 4: Evaluation)
**Issue**: TIMEOUT after 16 hours

## 🚨 What Happened

**Job Details**:
- Job ID: 23293358
- Stage: 4 (Comprehensive Evaluation)
- Time Limit: 16:00:00 (16 hours)
- Actual Runtime: 16:00:06 (hit limit exactly)
- Final State: TIMEOUT
- Exit Code: 0:0 (timeout, not error)

**Progress Before Timeout**:
- ✅ Standard T5 evaluation: COMPLETE (4.5 hours)
- ✅ Baseline T5 evaluation: COMPLETE (5.3 hours)
- ⏳ Monotonic T5 evaluation: IN PROGRESS (started at 04:26, killed at 10:35 = 6+ hours in)

**What Was Lost**:
- Monotonic T5 ROUGE scores (evaluation was ~80% complete)
- All result files (evaluation_results.json, final_results.json)
- Stage completion flag

**What Was Saved**:
- Training results from earlier stages ✅
- Log files showing progress ✅

---

## 📊 Why It Timed Out

**Evaluation Runtime Breakdown**:

| Model | Start | End | Duration |
|---|---|---|---|
| Standard T5 | 18:35 | 23:09 | 4.5 hours |
| Baseline T5 | 23:09 | 04:26 | 5.3 hours |
| Monotonic T5 | 04:26 | 10:35 (killed) | 6+ hours (incomplete) |
| **Total** | | | **16+ hours needed** |

**Time Allocated**: 16 hours
**Time Needed**: ~17-18 hours (estimate for complete evaluation)

**Gap**: Needed 1-2 more hours

**Why So Long**:
- Full CNN/DailyMail test set: 11,490 examples
- 3 models to evaluate
- Bootstrap confidence intervals: 1,000 resamples per metric
- Beam search decoding: 4 beams per example

**Calculation**:
- 11,490 examples × 3 models × 4 beams = ~137,000 forward passes
- Plus bootstrap resampling overhead
- ≈ 16-18 hours on single A100

---

## ✅ Fixes Applied

### Fix 1: Increase Time Limit

**Changed**:
- `hpc_version/jobs/job_4_evaluate.sh`: 16h → **20h**
- `hpc_version/configs/experiment_config.py`: TIME_EVALUATE → **20h**

**Rationale**:
- Actual need: ~17 hours
- New limit: 20 hours
- Buffer: 3 hours (adequate)

### Fix 2: Documentation

**Updated**: `COPY_RESULTS_FROM_HPC.md`
- Clarified seed-specific result paths
- Documented proper scp commands
- Prevents future confusion

---

## 🔧 How to Prevent This in Future

### Option A: Reduce Evaluation Workload (Quick Fix)

Edit `hpc_version/configs/experiment_config.py`:

```python
# Reduce test set size for faster evaluation
USE_FULL_TEST_SETS = False  # Use 200 samples instead of 11,490
# OR
QUICK_TEST_SIZE = 1000  # Use 1,000 samples (compromise)
```

**Result**: Evaluation completes in ~2-3 hours

**Trade-off**: Less robust statistics but faster iteration

### Option B: Use Full Test Set with Adequate Time (Current Fix)

**Configuration**:
```python
USE_FULL_TEST_SETS = True  # Keep full test set
TIME_EVALUATE = "20:00:00"  # Adequate time limit
```

**Result**: Robust statistics, requires patience

**This is what we implemented** ✅

### Option C: Checkpoint Evaluation Progress

**Future Enhancement**: Add checkpointing to evaluation script so it can resume if timeout

**Benefit**: Can use shorter jobs, resume if needed

**Complexity**: Moderate implementation effort

---

## 📋 What to Do Now

### Step 1: Pull Latest Code (Has Fix)

```bash
# On HPC
cd /projects/paco0228/mono-s2s
git pull origin main

# Verify you have the fix
grep "TIME_EVALUATE" hpc_version/configs/experiment_config.py
# Should show: TIME_EVALUATE = "20:00:00"

grep "time=" hpc_version/jobs/job_4_evaluate.sh
# Should show: #SBATCH --time=20:00:00
```

### Step 2: Decide on Evaluation Strategy

**Option A: Quick Mode (2-3 hours)**
```bash
# Edit hpc_version/configs/experiment_config.py
USE_FULL_TEST_SETS = False

# Then rerun
cd hpc_version
bash run_all.sh
```

**Option B: Full Test Set (17-18 hours with new limit)**
```bash
# Keep current config (USE_FULL_TEST_SETS = True)
# Just rerun with new 20-hour limit

cd hpc_version
bash run_all.sh
```

### Step 3: Resubmit Evaluation

```bash
# The fixed script will:
# - Skip completed stages 0-3 (have checkpoints)
# - Rerun stage 4 with 20-hour limit
# - Continue with stages 5-7

bash run_all.sh
```

**With 20-hour limit**: Job should complete successfully this time

---

## 🎯 Lessons Learned

1. **Always check sacct for timeouts**: `sacct -j <JOBID> --format=State,ExitCode`
2. **Time limits must account for full workload**: 11,490 samples needs 16-20 hours
3. **Test with quick mode first**: Verify pipeline works before full run
4. **Monitor long jobs**: Check progress periodically
5. **Results in seed subdirectories**: `/scratch/.../mono_s2s_results/seed_42/`

---

## ✅ Fixes Committed and Pushed

**Commits**:
- a62de30: Increase TIME_EVALUATE in config
- ba8c810: Increase time limit in job script

**Status**: Fixed ✅

**Safe to Rerun**: YES - pull latest code and resubmit

---

**The issue is now diagnosed and fixed. You can safely run the next experiment.**
