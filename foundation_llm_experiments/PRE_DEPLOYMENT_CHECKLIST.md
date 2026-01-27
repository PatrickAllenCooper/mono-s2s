# Pre-Deployment Checklist for HPC

Complete this checklist before submitting jobs to HPC.

## ✅ Phase 1: Local Verification (Required)

### Environment Setup

- [ ] Python 3.10+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] PyTorch installed with CUDA support (if available)
- [ ] Transformers library version >=4.30.0

**Verify**:
```bash
python -c "import torch, transformers; print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}')"
```

### Model and Dataset Availability

- [ ] Pythia-1.4B model is accessible
- [ ] Pile dataset is accessible
- [ ] Internet connection working
- [ ] Hugging Face Hub reachable
- [ ] No authentication issues

**Verify**:
```bash
python verify_downloads.py --quick
```

**Expected**: All download verifications pass

### Configuration Review

- [ ] `MODEL_NAME` set correctly (should be "EleutherAI/pythia-1.4b")
- [ ] `BATCH_SIZE` appropriate for A100 (recommend: 8)
- [ ] `SLURM_PARTITION` matches your cluster (aa100 for Alpine)
- [ ] Time limits reviewed and adequate
- [ ] Paths use `$SCRATCH` and `$PROJECT` environment variables
- [ ] Random seeds list contains 5 values

**Verify**:
```bash
python configs/experiment_config.py
```

### Test Suite

- [ ] Quick tests pass: `bash run_tests.sh quick`
- [ ] All unit tests pass: `bash run_tests.sh unit`
- [ ] Integration tests pass: `bash run_tests.sh integration`
- [ ] Verification script passes: `python verify_local.py`
- [ ] Coverage >70%: `bash run_tests.sh coverage`

**Verify**:
```bash
bash run_tests.sh all
python verify_local.py
```

**Expected Output**:
```
==================== 150+ passed in 2-5 min ====================
✓ ALL VERIFICATIONS PASSED
```

### Pipeline Logic

- [ ] Local pipeline test completes: `python test_pipeline_local.py`
- [ ] All 7 stages execute successfully
- [ ] Monotonicity constraints verified (weights >= 0)
- [ ] Checkpoints save correctly
- [ ] Results JSONs are valid

**Verify**:
```bash
python test_pipeline_local.py --verbose
```

**Expected Output**:
```
✓ ALL STAGES COMPLETED SUCCESSFULLY
```

## ✅ Phase 2: HPC Environment (Required)

### HPC Access

- [ ] Can SSH to HPC cluster
- [ ] Have valid allocation on partition
- [ ] Conda environment exists on HPC
- [ ] `$SCRATCH` and `$PROJECT` directories exist

**Verify**:
```bash
# On HPC login node
echo $SCRATCH
echo $PROJECT
ls -ld $SCRATCH
ls -ld $PROJECT
```

### Environment on HPC

- [ ] Conda environment `mono_s2s` exists on HPC
- [ ] Dependencies installed in HPC conda environment
- [ ] Can load CUDA module
- [ ] GPU is accessible in interactive session

**Verify**:
```bash
# On HPC
conda activate mono_s2s
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test GPU access
sinteractive --partition=aa100 --gres=gpu:1 --time=00:10:00
nvidia-smi
```

### File Transfer

- [ ] Code transferred to HPC
- [ ] File permissions set correctly
- [ ] Scripts are executable

**Transfer**:
```bash
# From local machine
rsync -avz foundation_llm_experiments/ user@hpc:/path/to/mono-s2s/foundation_llm_experiments/

# On HPC
cd /path/to/mono-s2s/foundation_llm_experiments
chmod +x run_all.sh scripts/*.py jobs/*.sh
```

### Storage Space

- [ ] Sufficient scratch space (~500GB per seed)
- [ ] Sufficient project space (~100GB)
- [ ] No quota exceeded

**Verify**:
```bash
# On HPC
du -sh $SCRATCH
quota -s
df -h $SCRATCH
```

## ✅ Phase 3: Job Configuration (Required)

### Job Scripts Review

- [ ] All 8 job scripts exist (`jobs/job_0_*.sh` through `job_7_*.sh`)
- [ ] Partition names correct for your cluster
- [ ] QOS correct for your allocation
- [ ] Time limits adequate
- [ ] Memory requests reasonable (80G for A100)
- [ ] Output/error log paths exist

**Review**:
```bash
# Check all job scripts
head -15 jobs/job_*.sh

# Verify logs directory exists
mkdir -p logs
```

### Dependency Chain

- [ ] `run_all.sh` uses correct dependency syntax
- [ ] Stage 1 depends on Stage 0
- [ ] Stage 2 depends on Stage 0
- [ ] Stage 3 depends on Stage 1
- [ ] Stage 4 depends on Stages 2,3
- [ ] Stages 5,6 depend on Stages 2,3
- [ ] Stage 7 depends on Stages 4,5,6

**Verify**:
```bash
grep -n "dependency" run_all.sh
```

### Resource Allocation

- [ ] Not requesting more GPUs than available
- [ ] Memory request matches GPU memory + overhead
- [ ] Time limits reviewed against expected runtime
- [ ] Not overloading cluster with too many simultaneous jobs

**Check Cluster Status**:
```bash
# On HPC
sinfo -p aa100
squeue -p aa100
```

## ✅ Phase 4: Data Availability (Important)

### Training Data

- [ ] Pile dataset accessible (or alternative decided)
- [ ] Enough disk space for dataset cache (~50GB)
- [ ] Network access for Hugging Face Hub
- [ ] Fallback plan if download fails

**Test Download**:
```bash
# On HPC in interactive session
python -c "from datasets import load_dataset; d = load_dataset('EleutherAI/pile', split='validation', streaming=True); print(next(iter(d)))"
```

### Evaluation Data

- [ ] Pile test split accessible
- [ ] Other benchmarks (LAMBADA, HellaSwag) available or skipped
- [ ] Evaluation data paths configured

## ✅ Phase 5: Monitoring Setup (Recommended)

### Job Monitoring

- [ ] Know how to check job status: `squeue -u $USER`
- [ ] Know how to check job details: `scontrol show job <JOBID>`
- [ ] Know how to view logs: `tail -f logs/job_0_*.out`
- [ ] Know how to cancel jobs: `scancel <JOBID>`

### Progress Tracking

- [ ] Can check completion flags: `ls $SCRATCH/foundation_llm_work/*.flag`
- [ ] Can view stage logs: `cat $SCRATCH/foundation_llm_work/stage_logs/*.log`
- [ ] Can monitor disk usage: `du -sh $SCRATCH/foundation_llm_work`

### Alerts (Optional)

- [ ] Email notifications configured in SLURM scripts
- [ ] Monitoring script set up (optional)

## ✅ Phase 6: Backup Plan (Recommended)

### Checkpointing

- [ ] Understand checkpoint resume mechanism
- [ ] Know how to manually resume from checkpoint
- [ ] Backup important checkpoints to `$PROJECT`

### Recovery Procedures

- [ ] Know how to diagnose failures
- [ ] Have procedure for OOM errors
- [ ] Have procedure for timeout errors
- [ ] Can manually resubmit individual stages

## ✅ Phase 7: Documentation Review (Recommended)

### Read Documentation

- [ ] Read `README.md`
- [ ] Read `QUICKSTART.md`
- [ ] Read `PAPER_INTEGRATION.md`
- [ ] Understand expected outputs
- [ ] Know where results will be saved

### Understand Pipeline

- [ ] Know what each stage does
- [ ] Understand stage dependencies
- [ ] Know expected runtime for each stage
- [ ] Understand success criteria

## Pre-Submission Command Sequence

### Final Checks (5 minutes)

```bash
# 1. Run tests locally
cd foundation_llm_experiments
bash run_tests.sh all

# 2. Run verification
python verify_local.py

# 3. Test local pipeline
python test_pipeline_local.py

# 4. Check configuration
python configs/experiment_config.py

# 5. Validate job scripts
bash -n jobs/*.sh  # Check syntax

# 6. Check file permissions
ls -l run_all.sh scripts/*.py jobs/*.sh
```

**All should complete without errors.**

### Transfer to HPC (2 minutes)

```bash
# From local machine
cd /path/to/mono-s2s
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    foundation_llm_experiments/ \
    user@alpine.rc.colorado.edu:/path/to/mono-s2s/foundation_llm_experiments/
```

### HPC Quick Checks (2 minutes)

```bash
# On HPC
cd /path/to/mono-s2s/foundation_llm_experiments

# Activate environment
conda activate mono_s2s

# Test imports
python -c "from utils.common_utils import make_model_monotonic; print('✓ Imports work')"

# Test config
python configs/experiment_config.py

# Check storage
du -sh $SCRATCH
quota -s
```

### Submit Jobs (1 minute)

```bash
# Review what will be submitted
cat run_all.sh

# Submit
bash run_all.sh

# Record job IDs
squeue -u $USER | tee submitted_jobs.txt
```

## Post-Submission Monitoring

### First 15 Minutes

- [ ] Check Stage 0 started: `squeue -u $USER`
- [ ] Monitor Stage 0 log: `tail -f logs/job_0_setup_*.out`
- [ ] Verify no immediate failures

### First Hour

- [ ] Stage 0 completed successfully
- [ ] Stage 1 started
- [ ] No error messages in logs
- [ ] Disk usage increasing (model downloading)

### First 12 Hours

- [ ] Stages 0-1 completed
- [ ] Stage 2 or 3 running (baseline/monotonic training)
- [ ] Training loss decreasing
- [ ] Checkpoints being saved

### Daily Checks

- [ ] Check queue status: `squeue -u $USER`
- [ ] Review logs for errors: `grep -i error logs/*.err`
- [ ] Monitor disk usage: `du -sh $SCRATCH/foundation_llm_work`
- [ ] Check completion flags: `ls $SCRATCH/foundation_llm_work/*.flag`

## Common Pre-Deployment Issues

### Issue 1: Tests Fail Locally

**Symptom**: pytest shows failures

**Action**: 
- Fix failing tests before proceeding
- Do not deploy with failing tests
- Review error messages carefully

### Issue 2: Dependencies Missing

**Symptom**: `ImportError` in verification

**Action**:
```bash
pip install -r requirements.txt
# Then rerun tests
```

### Issue 3: Config Values Unreasonable

**Symptom**: Warnings about batch size, learning rate, etc.

**Action**:
- Review `configs/experiment_config.py`
- Compare to main project values
- Adjust if necessary

### Issue 4: Storage Space Low

**Symptom**: `df -h` shows >90% usage

**Action**:
- Clean up old files
- Request more storage
- Reduce `TRAINING_SAMPLES` for initial test

## Emergency Procedures

### If You Need to Stop After Submission

```bash
# Cancel all your jobs
scancel $(squeue -u $USER -o "%i" -h)

# Or cancel specific job
scancel <JOB_ID>
```

### If Jobs Fail Immediately

```bash
# Check error logs
cat logs/job_0_setup_*.err

# Check SLURM reason
scontrol show job <JOB_ID>

# Common fixes:
# - Fix partition name
# - Fix QOS name
# - Reduce resource requests
```

### If You Need to Restart

```bash
# Checkpoint/resume should work automatically
# Just resubmit the failed job:
sbatch jobs/job_2_baseline.sh  # Example
```

## Sign-Off

Before submitting to HPC, confirm:

- [ ] ✅ All local tests pass
- [ ] ✅ Verification script passes
- [ ] ✅ Pipeline test completes successfully
- [ ] ✅ Configuration reviewed and correct
- [ ] ✅ HPC environment ready
- [ ] ✅ Storage space sufficient
- [ ] ✅ Monitoring plan in place
- [ ] ✅ Recovery procedures understood

**Signed off by**: ________________

**Date**: ________________

**Ready for HPC deployment**: YES / NO

---

## Quick Command Reference

```bash
# === LOCAL TESTING ===
bash run_tests.sh all              # Run all tests
python verify_local.py             # Verify configuration
python test_pipeline_local.py      # Test full pipeline

# === TRANSFER TO HPC ===
rsync -avz foundation_llm_experiments/ user@hpc:path/to/

# === ON HPC ===
conda activate mono_s2s            # Activate environment
python configs/experiment_config.py  # Test config
bash run_all.sh                    # Submit jobs

# === MONITORING ===
squeue -u $USER                    # Check job status
tail -f logs/job_0_setup_*.out     # Monitor log
ls $SCRATCH/foundation_llm_work/*.flag  # Check progress

# === IF ISSUES ===
scancel <JOB_ID>                   # Cancel job
sbatch jobs/job_X_*.sh             # Resubmit stage
```

---

**Do not skip this checklist.** Each item prevents potential issues that could waste hours of GPU time.
