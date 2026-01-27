#!/bin/bash
################################################################################
# Commit Experiment Results to Git
#
# Organizes and commits experimental results with proper metadata and provenance.
# Ensures all experimental evidence is version controlled.
#
# Usage:
#   bash scripts/commit_experiment_results.sh 42 t5_summarization
#   bash scripts/commit_experiment_results.sh 42 pythia_foundation --with-analysis
################################################################################

set -euo pipefail

SEED=${1}
EXPERIMENT_TYPE=${2}
WITH_ANALYSIS=${3:-}

if [ -z "$SEED" ] || [ -z "$EXPERIMENT_TYPE" ]; then
    echo "Usage: bash scripts/commit_experiment_results.sh <seed> <experiment_type> [--with-analysis]"
    echo ""
    echo "Example:"
    echo "  bash scripts/commit_experiment_results.sh 42 t5_summarization"
    echo "  bash scripts/commit_experiment_results.sh 1337 pythia_foundation --with-analysis"
    echo ""
    echo "Experiment types:"
    echo "  - t5_summarization"
    echo "  - pythia_foundation"
    exit 1
fi

echo "======================================================================"
echo "  COMMITTING EXPERIMENT RESULTS TO GIT"
echo "======================================================================"
echo ""
echo "Seed: $SEED"
echo "Experiment Type: $EXPERIMENT_TYPE"
echo ""

# Determine source and destination
if [ "$EXPERIMENT_TYPE" == "t5_summarization" ]; then
    SOURCE="${SCRATCH:-/scratch/alpine/$USER}/mono_s2s_results"
    DEST="experiment_results/t5_experiments/seed_${SEED}"
elif [ "$EXPERIMENT_TYPE" == "pythia_foundation" ]; then
    SOURCE="${SCRATCH:-/scratch/alpine/$USER}/foundation_llm_results"
    DEST="experiment_results/foundation_llm_experiments/seed_${SEED}"
else
    echo "ERROR: Unknown experiment type: $EXPERIMENT_TYPE"
    exit 1
fi

# Check source exists
if [ ! -d "$SOURCE" ]; then
    echo "ERROR: Source directory not found: $SOURCE"
    echo "Make sure experiment has completed and results are in SCRATCH"
    exit 1
fi

# Step 1: Organize results
echo "Step 1: Organizing results..."
python scripts/organize_results.py \
    --source "$SOURCE" \
    --dest "$DEST" \
    --seed "$SEED" \
    --experiment-type "$EXPERIMENT_TYPE"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to organize results"
    exit 1
fi

echo "✓ Results organized to: $DEST"
echo ""

# Step 2: Update experiment index
echo "Step 2: Updating experiment index..."
python scripts/update_experiment_index.py \
    --experiment-dir "$DEST" \
    --seed "$SEED" \
    --type "$EXPERIMENT_TYPE"

echo "✓ Experiment index updated"
echo ""

# Step 3: Run analysis (optional)
if [ "$WITH_ANALYSIS" == "--with-analysis" ]; then
    echo "Step 3: Running result analysis..."
    python scripts/analyze_results.py --experiment-dir "$DEST"
    echo "✓ Analysis complete"
    echo ""
fi

# Step 4: Check what will be committed
echo "Step 4: Checking files to commit..."
git status --short "$DEST" experiment_index.json

echo ""
read -p "Commit these files? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled. Files organized but not committed."
    echo "To commit later: git add $DEST && git commit"
    exit 0
fi

# Step 5: Commit to git
echo ""
echo "Step 5: Committing to git..."

git add "$DEST"
git add experiment_results/experiment_index.json

# Create detailed commit message
COMMIT_MSG="$(cat <<EOF
Add experimental results: ${EXPERIMENT_TYPE}, seed ${SEED}

Experiment Details:
- Type: ${EXPERIMENT_TYPE}
- Seed: ${SEED}
- Date: $(date -Idate)
- Results: $DEST

Files Added:
$(git diff --cached --name-only | sed 's/^/  - /')

This commit adds experimental evidence for the mono-s2s project.
All results are organized following EXPERIMENT_TRACKING_SYSTEM.md.

See metadata.json in results directory for full experiment details.
EOF
)"

git commit -m "$COMMIT_MSG"

COMMIT_HASH=$(git rev-parse HEAD)

echo ""
echo "======================================================================"
echo "  ✓ RESULTS COMMITTED TO GIT"
echo "======================================================================"
echo ""
echo "Commit: $COMMIT_HASH"
echo "Files: $DEST"
echo ""
echo "Next steps:"
echo "  1. Push to remote: git push origin main"
echo "  2. Tag if significant: git tag exp-${EXPERIMENT_TYPE}-seed${SEED}"
echo "  3. Update paper if needed (see PAPER_INTEGRATION.md)"
echo ""
echo "Results are now version controlled and preserved!"
echo ""
echo "======================================================================"
