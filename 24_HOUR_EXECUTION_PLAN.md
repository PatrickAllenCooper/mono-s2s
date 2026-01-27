# 24-Hour Execution Plan for Maximum Coverage

**Deadline**: Paper submission in 24 hours
**Strategy**: Quick mode, maximum seed coverage
**Expected Outcome**: 2-3 complete seeds with verified results

---

## ✅ Configuration: READY FOR QUICK MODE

**Latest commit**: 6a2f750 - "Enable quick mode for 24-hour deadline"

**Settings**:
- `USE_FULL_TEST_SETS = False` ✅
- Evaluation: 200 samples (~2-3 hours)
- Total per seed: ~12-14 hours
- **Can complete 2 seeds in 24 hours**

---

## 🚀 Execution Commands (Run on HPC NOW)

```bash
# 1. Pull latest code
cd /projects/paco0228/mono-s2s
git pull origin main

# 2. Verify quick mode enabled
grep "USE_FULL_TEST_SETS" hpc_version/configs/experiment_config.py
# Should show: USE_FULL_TEST_SETS = False

# 3. Start seed 42 (complete remaining stages 4-7)
cd hpc_version
bash run_all.sh

# Record job IDs for monitoring
```

**Expected Jobs**:
- Stage 4: Evaluation (~2-3 hours)
- Stage 5: UAT (~2 hours)
- Stage 6: HotFlip (~2 hours)
- Stage 7: Aggregate (~15 min)

**Total**: ~6-8 hours for seed 42

---

## ⏰ Timeline & Checkpoints

### Hour 0 (NOW): Submit Seed 42

```bash
cd /projects/paco0228/mono-s2s/hpc_version
bash run_all.sh
```

**Monitor**:
```bash
squeue -u paco0228
```

### Hour 2: Check Seed 42 Progress

```bash
# Should be in stage 5 or 6 by now
tail -20 logs/job_4_evaluate_<JOBID>.out

# Look for: "Stage 4: COMPLETED SUCCESSFULLY"
# If stuck: Cancel and investigate
```

### Hour 8: Seed 42 Complete, Start Seed 1337

```bash
# Verify seed 42 finished
ls -lh /scratch/alpine/paco0228/mono_s2s_results/seed_42/final_results.json

# If exists: SUCCESS, start next seed
EXPERIMENT_SEED=1337 bash run_all.sh
```

### Hour 10: Check Seed 1337 Progress

```bash
# Should be in stages 1-2
tail -20 logs/job_*_<LATEST>.out

# Verify data prep or training running
```

### Hour 16: Check Both Seeds

```bash
# Seed 42: Should be archived
# Seed 1337: Should be in stage 4-5

squeue -u paco0228
```

### Hour 22: Seed 1337 Should Complete

```bash
# Verify results
ls -lh /scratch/alpine/paco0228/mono_s2s_results/seed_1337/final_results.json

# Archive both seeds
bash scripts/archive_experiment.sh 42
bash scripts/archive_experiment.sh 1337
```

### Hour 24: Final Archive & Paper Update

```bash
# Copy results locally
scp -r paco0228@login.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/seed_42 ~/local/
scp -r paco0228@login.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/seed_1337 ~/local/

# Compute 2-seed statistics
# Update paper tables
```

---

## 📊 Expected Results After 24 Hours

**Completed Seeds**: 42, 1337 (maybe 2024 if lucky)

**Verified Tables**:
- ✅ Table 1: Training (seeds 42, 1337)
- ✅ Table 2: ROUGE (seeds 42, 1337)
- ✅ Table 5: HotFlip (seeds 42, 1337)
- ✅ Table 6: UAT (seeds 42, 1337)

**Partial Multi-Seed Stats**:
- Mean ± std for 2 seeds (not ideal, but shows consistency)
- Can add note: "2 of 5 seeds completed"

**Still Red**:
- Tables 3-4: Can add 2-seed stats (reduces red text)
- Table 7: Remains placeholder (no time for Pythia)

---

## 🚨 Failure Detection

### Every Stage Should Complete in Max:

| Stage | Max Time | Check At |
|---|---|---|
| 0: Setup | 30 min | 30 min |
| 1: Data | 2 hours | 2 hours |
| 2: Baseline Train | 20 hours | N/A (use checkpoints) |
| 3: Monotonic Train | 24 hours | N/A (use checkpoints) |
| 4: Evaluation | 3 hours | 3 hours |
| 5: UAT | 2 hours | 2 hours |
| 6: HotFlip | 2 hours | 2 hours |
| 7: Aggregate | 30 min | 30 min |

**In Quick Mode**: Stages 0-1 already done, stages 2-3 already done from checkpoints

**You're only running**: Stages 4-7 (~8 hours total)

### If Stage Exceeds Max Time

```bash
# 1. Check if making progress
tail -50 logs/job_*_<JOBID>.out

# 2. If stuck (no new output in 10 min):
scancel <JOBID>

# 3. Check error
cat logs/job_*_<JOBID>.err

# 4. Report to me for diagnosis
```

---

## 🎯 Success Criteria

**Minimum Success** (Seed 42 only):
- 1 complete seed with all stages
- Can update paper with single-seed values
- Better than nothing

**Good Success** (Seeds 42 + 1337):
- 2 complete seeds
- Can compute mean ± std
- Shows consistency
- Publishable

**Excellent Success** (Seeds 42 + 1337 + 2024):
- 3 complete seeds
- Robust statistics
- Strong evidence

---

## 📝 Monitoring Checklist

Print this and check off:

**Hour 0**:
- [ ] Pulled latest code (`git pull`)
- [ ] Verified quick mode (`USE_FULL_TEST_SETS = False`)
- [ ] Submitted jobs (`bash run_all.sh`)
- [ ] Recorded job IDs

**Hour 2**:
- [ ] Checked queue (`squeue -u paco0228`)
- [ ] Viewed progress logs
- [ ] Verified no errors

**Hour 4**:
- [ ] Stage 4 complete flag exists
- [ ] Stage 5 running

**Hour 6**:
- [ ] Stage 5 complete
- [ ] Stage 6 running

**Hour 8**:
- [ ] Seed 42 COMPLETE
- [ ] `final_results.json` exists
- [ ] Started seed 1337

**Hour 16**:
- [ ] Seed 1337 in stage 4-5
- [ ] No errors

**Hour 22**:
- [ ] Seed 1337 complete
- [ ] Both seeds archived

**Hour 24**:
- [ ] Results copied locally
- [ ] Paper updated
- [ ] Committed to git

---

## 🚀 START NOW

**Commands to run RIGHT NOW**:

```bash
# On HPC
cd /projects/paco0228/mono-s2s
git pull origin main
cd hpc_version
bash run_all.sh
```

**Then set a timer to check every 2 hours.**

---

**This plan maximizes your chances of getting 2-3 complete seeds in 24 hours with auditable progress at every stage.**
