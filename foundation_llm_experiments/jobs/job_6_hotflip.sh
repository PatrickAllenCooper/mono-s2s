#!/bin/bash
#SBATCH --job-name=foundation_hotflip
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=77G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_6_hotflip_%j.out
#SBATCH --error=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_6_hotflip_%j.err

# Stage 6: HotFlip Gradient-Based Attacks

echo "=========================================="
echo "SLURM Job: Stage 6 - HotFlip Attacks"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s || exit 1

export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-mlp_both}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || exit 1

echo ""
echo "Running HotFlip attacks (variant=$MONOTONIC_VARIANT)..."
python stage_6_hotflip_attacks.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 6: COMPLETED SUCCESSFULLY"
    echo "HotFlip attack results saved"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 6: FAILED (exit code: $EXIT_CODE)"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
