#!/bin/bash
#SBATCH --job-name=mono_s2s_order_preservation
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/job_8_order_preservation_%j.out
#SBATCH --error=logs/job_8_order_preservation_%j.err

# Stage 8: End-to-end order preservation (inference only; T5-small forward
# passes over a few hundred short sentences plus lightweight logistic-
# regression probe fitting -- cheap relative to earlier stages). Requires
# hpc_version/data/ordered_pairs.json to exist (committed to the repo; only
# regenerate with build_ordered_pairs.py if you intend to change the corpus).

echo "=========================================="
echo "SLURM Job: Stage 8 - Order Preservation"
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
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export T5_ABLATION_MODE=${T5_ABLATION_MODE:-nonneg}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"

cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."

if [ ! -f "data/ordered_pairs.json" ]; then
    echo "data/ordered_pairs.json missing; generating it now (deterministic, committed thereafter)."
    python scripts/build_ordered_pairs.py
fi

cd scripts || exit 1

echo ""
echo "Running order-preservation measurement (ablation_mode=$T5_ABLATION_MODE)..."
python stage_8_order_preservation.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 8: COMPLETED SUCCESSFULLY"
    echo "Order-preservation results and depth plot saved"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 8: FAILED (exit code: $EXIT_CODE)"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
