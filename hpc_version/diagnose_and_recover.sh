#!/bin/bash
#
# Diagnostic and Recovery Script for Mono-S2S HPC Runs
#
# This script helps diagnose path issues and recover from failed runs
# by checking where files actually exist and creating missing completion flags
#

echo "=========================================="
echo "Mono-S2S Diagnostic and Recovery Tool"
echo "=========================================="
echo ""

# Set environment variables (critical for Alpine)
if [ -z "$SCRATCH" ]; then
    export SCRATCH="/scratch/alpine/$USER"
    echo "[FIX] SCRATCH was not set, now set to: $SCRATCH"
fi

if [ -z "$PROJECT" ]; then
    export PROJECT="/projects/$USER"
    echo "[FIX] PROJECT was not set, now set to: $PROJECT"
fi

echo ""
echo "Environment:"
echo "  USER: $USER"
echo "  SCRATCH: $SCRATCH"
echo "  PROJECT: $PROJECT"
echo ""

# Check if directories exist
echo "=========================================="
echo "Checking Directories"
echo "=========================================="

if [ ! -d "$SCRATCH" ]; then
    echo "[ERROR] SCRATCH directory does not exist: $SCRATCH"
    echo "        Cannot proceed. Check your Alpine filesystem access."
    exit 1
fi
echo "✓ SCRATCH directory exists"

if [ -d "$SCRATCH/mono_s2s_work" ]; then
    echo "✓ Work directory exists: $SCRATCH/mono_s2s_work"
else
    echo "[WARNING] Work directory does not exist: $SCRATCH/mono_s2s_work"
    echo "          Jobs may not have run yet, or ran with different paths"
fi

if [ -d "$SCRATCH/mono_s2s_results" ]; then
    echo "✓ Results directory exists: $SCRATCH/mono_s2s_results"
else
    echo "[WARNING] Results directory does not exist: $SCRATCH/mono_s2s_results"
fi

echo ""
echo "=========================================="
echo "Checking Completion Flags"
echo "=========================================="

# Check for shared flags (Stage 0, 1)
SHARED_FLAGS=(
    "stage_0_setup_complete.flag"
    "stage_1_data_prep_complete.flag"
)

for flag in "${SHARED_FLAGS[@]}"; do
    flag_path="$SCRATCH/mono_s2s_work/$flag"
    if [ -f "$flag_path" ]; then
        echo "✓ Found: $flag"
        cat "$flag_path" | sed 's/^/    /'
    else
        echo "✗ Missing: $flag"
    fi
done

echo ""
echo "Checking seed-specific flags..."

# Check all seed directories
SEEDS=(42 1337 2024 8888 12345)
for seed in "${SEEDS[@]}"; do
    seed_dir="$SCRATCH/mono_s2s_work/seed_$seed"
    if [ -d "$seed_dir" ]; then
        echo ""
        echo "Seed $seed:"
        num_flags=$(ls "$seed_dir"/*_complete.flag 2>/dev/null | wc -l)
        echo "  Found $num_flags completion flags"
        if [ $num_flags -gt 0 ]; then
            ls "$seed_dir"/*_complete.flag 2>/dev/null | sed 's/^/    /'
        fi
    fi
done

echo ""
echo "=========================================="
echo "Checking Results Files"
echo "=========================================="

# Check if any results were actually saved
for seed in "${SEEDS[@]}"; do
    results_dir="$SCRATCH/mono_s2s_results/seed_$seed"
    if [ -d "$results_dir" ]; then
        num_files=$(ls "$results_dir"/*.json 2>/dev/null | wc -l)
        if [ $num_files -gt 0 ]; then
            echo ""
            echo "Seed $seed results:"
            ls -lh "$results_dir"/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
        fi
    fi
done

echo ""
echo "=========================================="
echo "Checking Checkpoints"
echo "=========================================="

# Check for model checkpoints
for seed in "${SEEDS[@]}"; do
    checkpoint_dir="$SCRATCH/mono_s2s_work/checkpoints/seed_$seed"
    if [ -d "$checkpoint_dir" ]; then
        echo ""
        echo "Seed $seed checkpoints:"
        
        # Check baseline
        baseline_dir="$checkpoint_dir/baseline_checkpoints"
        if [ -d "$baseline_dir" ]; then
            num_epochs=$(ls "$baseline_dir"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
            has_best=$([ -f "$baseline_dir/best_model.pt" ] && echo "yes" || echo "no")
            echo "  Baseline: $num_epochs epoch checkpoints, best_model=$has_best"
        fi
        
        # Check monotonic
        monotonic_dir="$checkpoint_dir/monotonic_checkpoints"
        if [ -d "$monotonic_dir" ]; then
            num_epochs=$(ls "$monotonic_dir"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
            has_best=$([ -f "$monotonic_dir/best_model.pt" ] && echo "yes" || echo "no")
            echo "  Monotonic: $num_epochs epoch checkpoints, best_model=$has_best"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Checking Data Cache"
echo "=========================================="

data_cache="$SCRATCH/mono_s2s_work/data_cache"
if [ -d "$data_cache" ]; then
    echo "Data cache directory exists:"
    ls -lh "$data_cache"/*.pt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "[WARNING] Data cache does not exist"
fi

echo ""
echo "=========================================="
echo "Recovery Recommendations"
echo "=========================================="

# Determine what needs to be done
if [ ! -f "$SCRATCH/mono_s2s_work/stage_0_setup_complete.flag" ]; then
    if [ -f "$SCRATCH/mono_s2s_results/seed_42/setup_complete.json" ]; then
        echo ""
        echo "[RECOVERABLE] Stage 0 ran successfully but flag is missing"
        echo "  Creating flag..."
        mkdir -p "$SCRATCH/mono_s2s_work"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" > "$SCRATCH/mono_s2s_work/stage_0_setup_complete.flag"
        echo "Seed: 42" >> "$SCRATCH/mono_s2s_work/stage_0_setup_complete.flag"
        echo "Note: Recovered by diagnose_and_recover.sh" >> "$SCRATCH/mono_s2s_work/stage_0_setup_complete.flag"
        echo "  ✓ Created stage_0_setup_complete.flag"
    else
        echo ""
        echo "[ACTION NEEDED] Stage 0 (setup) needs to run"
        echo "  Run: sbatch jobs/job_0_setup.sh"
    fi
fi

if [ ! -f "$SCRATCH/mono_s2s_work/stage_1_data_prep_complete.flag" ]; then
    if [ -f "$SCRATCH/mono_s2s_work/data_cache/train_data.pt" ]; then
        echo ""
        echo "[RECOVERABLE] Stage 1 ran successfully but flag is missing"
        echo "  Creating flag..."
        mkdir -p "$SCRATCH/mono_s2s_work"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" > "$SCRATCH/mono_s2s_work/stage_1_data_prep_complete.flag"
        echo "Seed: 42" >> "$SCRATCH/mono_s2s_work/stage_1_data_prep_complete.flag"
        echo "Note: Recovered by diagnose_and_recover.sh" >> "$SCRATCH/mono_s2s_work/stage_1_data_prep_complete.flag"
        echo "  ✓ Created stage_1_data_prep_complete.flag"
    else
        echo ""
        echo "[ACTION NEEDED] Stage 1 (data preparation) needs to run"
        echo "  Run: sbatch jobs/job_1_data.sh"
    fi
fi

# Check each seed
for seed in "${SEEDS[@]}"; do
    seed_dir="$SCRATCH/mono_s2s_work/seed_$seed"
    checkpoint_dir="$SCRATCH/mono_s2s_work/checkpoints/seed_$seed"
    
    # Check baseline training
    if [ ! -f "$seed_dir/stage_2_train_baseline_complete.flag" ]; then
        baseline_dir="$checkpoint_dir/baseline_checkpoints"
        if [ -d "$baseline_dir" ]; then
            num_epochs=$(ls "$baseline_dir"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
            has_best=$([ -f "$baseline_dir/best_model.pt" ] && echo "yes" || echo "no")
            
            if [ $num_epochs -ge 5 ] && [ "$has_best" = "yes" ]; then
                echo ""
                echo "[RECOVERABLE] Seed $seed baseline training completed but flag missing"
                echo "  Creating flag..."
                mkdir -p "$seed_dir"
                echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" > "$seed_dir/stage_2_train_baseline_complete.flag"
                echo "Seed: $seed" >> "$seed_dir/stage_2_train_baseline_complete.flag"
                echo "Note: Recovered by diagnose_and_recover.sh after verifying $num_epochs checkpoints" >> "$seed_dir/stage_2_train_baseline_complete.flag"
                echo "  ✓ Created stage_2_train_baseline_complete.flag for seed $seed"
            fi
        fi
    fi
    
    # Check monotonic training
    if [ ! -f "$seed_dir/stage_3_train_monotonic_complete.flag" ]; then
        monotonic_dir="$checkpoint_dir/monotonic_checkpoints"
        if [ -d "$monotonic_dir" ]; then
            num_epochs=$(ls "$monotonic_dir"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
            has_best=$([ -f "$monotonic_dir/best_model.pt" ] && echo "yes" || echo "no")
            
            if [ $num_epochs -ge 7 ] && [ "$has_best" = "yes" ]; then
                echo ""
                echo "[RECOVERABLE] Seed $seed monotonic training completed but flag missing"
                echo "  Creating flag..."
                mkdir -p "$seed_dir"
                echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" > "$seed_dir/stage_3_train_monotonic_complete.flag"
                echo "Seed: $seed" >> "$seed_dir/stage_3_train_monotonic_complete.flag"
                echo "Note: Recovered by diagnose_and_recover.sh after verifying $num_epochs checkpoints" >> "$seed_dir/stage_3_train_monotonic_complete.flag"
                echo "  ✓ Created stage_3_train_monotonic_complete.flag for seed $seed"
            fi
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "To retry the multi-seed run with fixed paths:"
echo "  1. Make sure you're in the hpc_version directory"
echo "  2. Cancel any stuck jobs: scancel -u $USER"
echo "  3. Run: ./run_multi_seed.sh"
echo ""
echo "The updated scripts now properly set SCRATCH and PROJECT variables."
echo ""

exit 0
