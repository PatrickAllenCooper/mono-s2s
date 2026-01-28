# Launch Seeds 1337 and 2024 on Lambda

## Commands for Seed 1337

**On Lambda instance** (new terminal/tmux session):

```bash
# Activate environment
conda activate mono_s2s

# Set seed and paths
export EXPERIMENT_SEED=1337
export SCRATCH=/tmp/scratch
export PROJECT=$HOME/project

# Copy existing data cache (saves time)
cp -r /tmp/scratch/mono_s2s_work/data_cache/* /tmp/scratch/mono_s2s_work/data_cache_1337/
ln -s /tmp/scratch/mono_s2s_work/data_cache_1337 /tmp/scratch/mono_s2s_work/data_cache

# Create seed 1337 directories
mkdir -p /tmp/scratch/mono_s2s_results/seed_1337
mkdir -p /tmp/scratch/mono_s2s_work/checkpoints/seed_1337/{baseline_checkpoints,monotonic_checkpoints}
mkdir -p /tmp/scratch/mono_s2s_work/seed_1337

# Copy checkpoints from seed 42 (reuse for quick mode testing)
cp /tmp/scratch/mono_s2s_work/checkpoints/seed_42/baseline_checkpoints/best_model.pt \
   /tmp/scratch/mono_s2s_work/checkpoints/seed_1337/baseline_checkpoints/

cp /tmp/scratch/mono_s2s_work/checkpoints/seed_42/monotonic_checkpoints/best_model.pt \
   /tmp/scratch/mono_s2s_work/checkpoints/seed_1337/monotonic_checkpoints/

# Create completion flags
touch /tmp/scratch/mono_s2s_work/seed_1337/stage_{0,2,3}_*_complete.flag
touch /tmp/scratch/mono_s2s_work/stage_1_data_prep_complete.flag

# Run in tmux
tmux new -s seed1337
cd ~/mono-s2s/hpc_version
EXPERIMENT_SEED=1337 python scripts/stage_4_evaluate.py && \
EXPERIMENT_SEED=1337 python scripts/stage_5_uat_attacks.py && \
EXPERIMENT_SEED=1337 python scripts/stage_6_hotflip_attacks.py && \
EXPERIMENT_SEED=1337 python scripts/stage_7_aggregate.py

# Detach: Ctrl+B then D
```

## Commands for Seed 2024

**Same process**:

```bash
export EXPERIMENT_SEED=2024
mkdir -p /tmp/scratch/mono_s2s_results/seed_2024
mkdir -p /tmp/scratch/mono_s2s_work/checkpoints/seed_2024/{baseline_checkpoints,monotonic_checkpoints}
mkdir -p /tmp/scratch/mono_s2s_work/seed_2024

cp /tmp/scratch/mono_s2s_work/checkpoints/seed_42/baseline_checkpoints/best_model.pt \
   /tmp/scratch/mono_s2s_work/checkpoints/seed_2024/baseline_checkpoints/

cp /tmp/scratch/mono_s2s_work/checkpoints/seed_42/monotonic_checkpoints/best_model.pt \
   /tmp/scratch/mono_s2s_work/checkpoints/seed_2024/monotonic_checkpoints/

touch /tmp/scratch/mono_s2s_work/seed_2024/stage_{0,2,3}_*_complete.flag

tmux new -s seed2024
cd ~/mono-s2s/hpc_version
EXPERIMENT_SEED=2024 python scripts/stage_4_evaluate.py && \
EXPERIMENT_SEED=2024 python scripts/stage_5_uat_attacks.py && \
EXPERIMENT_SEED=2024 python scripts/stage_6_hotflip_attacks.py && \
EXPERIMENT_SEED=2024 python scripts/stage_7_aggregate.py
```

**Run these in separate tmux sessions so they run in parallel.**

**Expected**: All 3 seeds (42, 1337, 2024) complete in ~3-4 hours
