#!/bin/bash
################################################################################
# Snapshot Currently Running Experiments
#
# Creates a snapshot of running experiments for tracking purposes.
# Run this periodically (e.g., daily) to maintain experiment history.
#
# Usage:
#   bash scripts/snapshot_running_experiments.sh
#   bash scripts/snapshot_running_experiments.sh --commit  # Also commit snapshot
################################################################################

COMMIT_FLAG=${1:-}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_DIR="experiment_snapshots"
SNAPSHOT_FILE="${SNAPSHOT_DIR}/snapshot_${TIMESTAMP}.json"

mkdir -p "$SNAPSHOT_DIR"

echo "======================================================================"
echo "  CREATING EXPERIMENT SNAPSHOT"
echo "======================================================================"
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Output: $SNAPSHOT_FILE"
echo ""

# Collect information
echo "Collecting experiment information..."

# Get running jobs
echo "  - SLURM job queue status"
JOBS_JSON=$(squeue -u $USER -o "%i|%j|%t|%M|%N|%R" 2>/dev/null | \
python3 -c "
import sys
import json
lines = sys.stdin.read().strip().split('\n')
if len(lines) < 2:
    print(json.dumps([]))
    sys.exit(0)
    
jobs = []
for line in lines[1:]:
    parts = line.split('|')
    if len(parts) >= 6:
        jobs.append({
            'job_id': parts[0].strip(),
            'name': parts[1].strip(),
            'state': parts[2].strip(),
            'time': parts[3].strip(),
            'node': parts[4].strip(),
            'reason': parts[5].strip(),
        })
print(json.dumps(jobs, indent=2))
" 2>/dev/null || echo "[]")

# Get completion flags
echo "  - Completion flags"
COMPLETION_FLAGS=()
if [ -d "${SCRATCH}/mono_s2s_work" ]; then
    while IFS= read -r flag; do
        COMPLETION_FLAGS+=("$(basename "$flag")")
    done < <(find "${SCRATCH}/mono_s2s_work" -name "*.flag" -type f 2>/dev/null)
fi

# Get disk usage
echo "  - Disk usage"
DISK_USAGE=$(du -sh "${SCRATCH}/mono_s2s_work" 2>/dev/null | cut -f1 || echo "unknown")

# Create snapshot JSON
echo "  - Creating snapshot JSON"
cat > "$SNAPSHOT_FILE" <<EOF
{
  "snapshot_timestamp": "$(date -Iseconds)",
  "snapshot_date": "$(date -Idate)",
  "user": "$USER",
  "hostname": "$(hostname)",
  "running_jobs": $JOBS_JSON,
  "completion_flags": $(printf '%s\n' "${COMPLETION_FLAGS[@]}" | jq -R . | jq -s . 2>/dev/null || echo "[]"),
  "disk_usage": {
    "mono_s2s_work": "$DISK_USAGE",
    "scratch_dir": "${SCRATCH:-unknown}"
  },
  "active_experiments": {
    "t5_experiments": $(ls -d experiment_results/t5_experiments/seed_* 2>/dev/null | wc -l),
    "foundation_experiments": $(ls -d experiment_results/foundation_llm_experiments/seed_* 2>/dev/null | wc -l)
  }
}
EOF

echo "✓ Snapshot created: $SNAPSHOT_FILE"
echo ""

# Display summary
echo "Snapshot Summary:"
echo "  - Running/Pending jobs: $(echo "$JOBS_JSON" | jq 'length' 2>/dev/null || echo "0")"
echo "  - Completion flags: ${#COMPLETION_FLAGS[@]}"
echo "  - Disk usage: $DISK_USAGE"
echo ""

# Optionally commit
if [ "$COMMIT_FLAG" == "--commit" ]; then
    echo "Committing snapshot to git..."
    
    git add "$SNAPSHOT_FILE"
    git commit -m "Experiment snapshot: $(date -Idate)

Captured state of running experiments at $TIMESTAMP

Running jobs: $(echo "$JOBS_JSON" | jq 'length' 2>/dev/null || echo "0")
Completion flags: ${#COMPLETION_FLAGS[@]}

This snapshot provides a record of experimental progress.
See $SNAPSHOT_FILE for details."
    
    echo "✓ Snapshot committed to git"
else
    echo "Snapshot created but not committed."
    echo "To commit: bash scripts/snapshot_running_experiments.sh --commit"
    echo "Or manually: git add $SNAPSHOT_FILE && git commit"
fi

echo ""
echo "======================================================================"
echo "  ✓ SNAPSHOT COMPLETE"
echo "======================================================================"
echo ""
echo "Snapshot file: $SNAPSHOT_FILE"
echo ""
echo "To view: cat $SNAPSHOT_FILE | jq ."
echo "To list all snapshots: ls -lh $SNAPSHOT_DIR/"
echo ""
