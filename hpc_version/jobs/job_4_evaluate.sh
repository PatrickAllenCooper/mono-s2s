#!/bin/bash
#SBATCH --job-name=mono_s2s_evaluate
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:50:00
#SBATCH --output=logs/job_4_evaluate_%j.out
#SBATCH --error=logs/job_4_evaluate_%j.err

# Stage 4: Comprehensive Evaluation
# Evaluates all three models on all test datasets with bootstrap CIs

echo "=========================================="
echo "SLURM Job: Stage 4 - Comprehensive Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Load modules (try to load CUDA if available)
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

# Activate conda environment (installed to /projects)
CONDA_ENV="${CONDA_ENV:-mono_s2s}"
source "${SLURM_SUBMIT_DIR}/jobs/activate_conda.sh"

# Set environment variables for determinism
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export T5_ABLATION_MODE=${T5_ABLATION_MODE:-nonneg}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Redirect HuggingFace cache to scratch
export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"

# Optional: restrict this job to a subset of datasets (comma-separated,
# e.g. "cnn_dm" or "xsum,samsum"). Used to shard USE_FULL_TEST_SETS=1 eval
# across parallel jobs so each stays under the 24h gpu-normal QoS ceiling --
# see stage_4_evaluate.py and stage_4_merge_shards.py / job_4_merge.sh.
export EVAL_DATASET_FILTER="${EVAL_DATASET_FILTER:-}"

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Navigate and run
cd $SLURM_SUBMIT_DIR/scripts

if [ -n "$EVAL_DATASET_FILTER" ]; then
    echo "Evaluating ALL THREE models on dataset shard: $EVAL_DATASET_FILTER"
else
    echo "Evaluating ALL THREE models on all test datasets..."
fi
echo "Models: Standard T5, Baseline T5, Monotonic T5"
echo "Datasets: CNN/DailyMail, XSUM, SAMSum"
echo "Includes: Bootstrap 95% CIs, length statistics, brevity penalty"
echo ""

python stage_4_evaluate.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 4: COMPLETED SUCCESSFULLY"
    echo "All models evaluated on all test datasets"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 4: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs for evaluation errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

