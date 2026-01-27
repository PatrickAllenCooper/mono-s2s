# Recover Experimental Results

## Check Where Results Actually Are (Run on HPC)

```bash
# 1. Find ALL result files on HPC
find /scratch/alpine/paco0228 -name "evaluation_results.json" 2>/dev/null

# 2. Check different possible locations
ls -lh /scratch/alpine/paco0228/mono_s2s_work/
ls -lh /scratch/alpine/paco0228/mono_s2s_results/
ls -lh /scratch/summit/paco0228/mono_s2s_results/ 2>/dev/null

# 3. Check if job is still running or completed
sacct -j 23293358 --format=JobID,State,ExitCode,Elapsed

# 4. Check the end of the log to see where it tried to save
tail -100 /projects/paco0228/mono-s2s/hpc_version/logs/job_4_evaluate_23293358.out
```

## Possible Issues

### Issue 1: Job Still Running

If `sacct` shows "RUNNING", the job hasn't finished yet.

### Issue 2: Wrong SCRATCH Path

Results might be in:
- `/scratch/alpine/paco0228/mono_s2s_results/`
- `/scratch/summit/paco0228/mono_s2s_results/`
- Different seed directory

### Issue 3: Results Not Saved

Check the stage 4 script to see where it saves results.
