# Experimental Results Repository

**Purpose**: Centralized, version-controlled storage of all experimental evidence

## Structure

```
experiment_results/
├── t5_experiments/           # T5 summarization experiments
│   ├── seed_42/              # Results for seed 42
│   ├── seed_1337/            # Results for seed 1337
│   ├── seed_2024/            # Results for seed 2024
│   ├── seed_8888/            # Results for seed 8888
│   ├── seed_12345/           # Results for seed 12345
│   └── aggregated/           # Multi-seed aggregated results
│
├── foundation_llm_experiments/  # Pythia foundation LLM experiments
│   ├── seed_42/
│   └── (same structure as t5_experiments)
│
├── experiment_index.json     # Master index of all experiments
└── README.md                 # This file
```

## Each Seed Directory Contains

- `metadata.json` - Full experiment metadata (hyperparams, hardware, timing)
- `baseline_training.json` - Baseline training history
- `monotonic_training.json` - Monotonic training history
- `evaluation_results.json` - Evaluation metrics (ROUGE or perplexity)
- `uat_results.json` - Universal adversarial trigger results
- `hotflip_results.json` - HotFlip attack results
- `final_results.json` - Aggregated final results
- `experiment_log.txt` - Human-readable summary

## Version Control

**Tracked**: All `.json`, `.csv`, `.txt` files (small, <100MB)

**Not Tracked**: Model checkpoints (`.pt` files, GBs)
- Archived separately to `$PROJECT/mono_s2s_final_results/`
- Referenced in `metadata.json`

## Adding New Results

Use the automated workflow:

```bash
# After experiment completes on HPC
bash scripts/commit_experiment_results.sh <seed> <type>

# Example:
bash scripts/commit_experiment_results.sh 42 t5_summarization
```

This automatically:
1. Organizes results into proper structure
2. Creates metadata.json
3. Updates experiment_index.json
4. Commits to git with descriptive message

## Querying Results

```bash
# List all experiments
cat experiment_index.json | jq '.experiments[] | {id, seed, type, status}'

# Get specific experiment
cat t5_experiments/seed_42/final_results.json | jq .

# Find experiments used in paper
cat experiment_index.json | jq '.experiments[] | select(.used_in_paper == true)'
```

## Linking to Paper

All results used in the paper are linked via provenance:

```bash
# See which experiments produced which paper tables
cat ../paper_evidence/provenance.json | jq .
```

## Guidelines

1. **One directory per seed** - Never mix seeds
2. **Complete metadata** - Always include metadata.json
3. **Commit atomically** - One experiment = one commit
4. **Link to paper** - Record provenance for paper values
5. **Never delete** - Results are permanent once committed

## Reproducibility

Every experiment directory contains enough information to:
- Reproduce the exact configuration
- Verify the results
- Trace values to paper tables
- Identify hardware and software versions

See `metadata.json` for complete details.
