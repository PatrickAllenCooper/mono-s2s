#!/bin/bash
#SBATCH --job-name=mono_s2s_hotflip_transfer
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_6b_hotflip_transfer_%j.out
#SBATCH --error=logs/job_6b_hotflip_transfer_%j.err

# Stage 6b: HotFlip substitution transfer + gradient-free controls
# (re-attacks both models, cross-evaluates, adds a random-substitution
# control and an optional query-based gradient-free attack). Set
# OVERRIDE_RUN_QUERY_ATTACK=0 to skip the most expensive control if
# time-constrained. Set T5_ABLATION_MODE to run against an
# attribution-ablation arm's monotonic checkpoint instead of the main run.

echo "=========================================="
echo "SLURM Job: Stage 6b - HotFlip Transfer + Controls"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export T5_ABLATION_MODE=${T5_ABLATION_MODE:-nonneg}
export OVERRIDE_RUN_QUERY_ATTACK=${OVERRIDE_RUN_QUERY_ATTACK:-1}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

cd $SLURM_SUBMIT_DIR/scripts

echo "Running HotFlip transfer + controls (ablation_mode=$T5_ABLATION_MODE, query_attack=$OVERRIDE_RUN_QUERY_ATTACK)..."
python stage_6b_hotflip_transfer.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 6b: COMPLETED SUCCESSFULLY"
    echo "HotFlip transfer + control results saved"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 6b: FAILED (exit code: $EXIT_CODE)"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
