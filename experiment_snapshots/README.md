# Experiment Snapshots

**Purpose**: Daily snapshots of running experiments for progress tracking

## What's Stored

Each snapshot captures:
- Which SLURM jobs are running
- Job status (running, pending, completed)
- Runtime for each job
- Completion flags that exist
- Disk usage
- Active experiment counts

## Format

```json
{
  "snapshot_timestamp": "2026-01-27T18:00:00",
  "user": "paco0228",
  "running_jobs": [
    {
      "job_id": "23293371",
      "name": "mono_s2s",
      "state": "R",
      "time": "1:38:04",
      "node": "c3gpu-a9-u33-1"
    }
  ],
  "completion_flags": [
    "stage_0_setup_complete.flag",
    "stage_1_data_prep_complete.flag"
  ],
  "disk_usage": {
    "mono_s2s_work": "45G"
  }
}
```

## Creating Snapshots

### Manual

```bash
bash scripts/snapshot_running_experiments.sh --commit
```

### Automated (Cron)

```bash
# Add to crontab on HPC
crontab -e

# Run daily at 6 PM
0 18 * * * cd ~/mono-s2s && bash scripts/snapshot_running_experiments.sh --commit
```

## Querying Snapshots

```bash
# Find all snapshots for a date
ls experiment_snapshots/snapshot_20260127_*.json

# Check job status at specific time
cat experiment_snapshots/snapshot_20260127_180000.json | jq '.running_jobs'

# Track progress over time
for snap in experiment_snapshots/snapshot_*.json; do
    echo "$(basename $snap): $(jq -r '.running_jobs | length' $snap) jobs running"
done
```

## Use Cases

### 1. Diagnose Failures

If an experiment fails, check snapshots to see:
- When it started
- How long it ran
- What stage it was in
- What else was running

### 2. Track Progress

Monitor multi-day experiments:
- See which stages completed when
- Identify bottlenecks
- Estimate completion time

### 3. Audit Trail

Provide evidence of experimental timeline:
- When experiments ran
- How long they took
- What resources were used

## Maintenance

**Keep snapshots for**:
- Active experiments: All snapshots
- Completed experiments: Daily snapshots only
- Old experiments (>6 months): Monthly snapshots

**Prune old snapshots**:

```bash
# Keep only monthly for old experiments
bash scripts/prune_old_snapshots.sh --keep-monthly
```

---

These snapshots provide a complete historical record of your experimental work.
