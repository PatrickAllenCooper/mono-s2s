# Paper Evidence - Experimental Results Used in Publication

**Purpose**: Store the exact experimental values used in the paper with full provenance

## Structure

```
paper_evidence/
├── provenance.json                    # Links experiments → paper tables
├── table_1_training_dynamics.json     # Data for Table 1
├── table_2_rouge_scores.json          # Data for Table 2
├── table_3_multiseed_training.json    # Data for Table 3
├── table_4_multiseed_rouge.json       # Data for Table 4
├── table_5_hotflip_attacks.json       # Data for Table 5
├── table_6_uat_attacks.json           # Data for Table 6
├── table_7_foundation_models.json     # Data for Table 7
└── README.md                          # This file
```

## Provenance System

`provenance.json` creates an audit trail:

```json
{
  "paper": "documentation/monotone_llms_paper.tex",
  "tables": {
    "table_1_training_dynamics": {
      "source_experiment": "t5_small_seed42_20260127",
      "source_files": [
        "experiment_results/t5_experiments/seed_42/baseline_training.json",
        "experiment_results/t5_experiments/seed_42/monotonic_training.json"
      ],
      "extracted_values": {
        "baseline_initial_loss": 2.90,
        "monotonic_initial_loss": 4.97,
        "baseline_final_loss": 2.47,
        "monotonic_final_loss": 2.69
      },
      "extraction_date": "2026-01-27",
      "verified": true
    }
  }
}
```

## Why This Matters

### For Reviewers

Reviewers can:
1. See exactly which experiments produced which claims
2. Trace any paper value back to raw experimental data
3. Verify reproducibility information is complete
4. Check for consistency across tables

### For Reproducibility

Anyone can:
1. Find the exact experimental run that produced a result
2. See the full configuration used
3. Access the complete raw data
4. Rerun the same experiment with same settings

### For Paper Revisions

If reviewers ask about a specific value:
1. Check `provenance.json` for which experiment
2. Load `final_results.json` from that experiment
3. Verify the value is correct
4. Provide complete details if needed

## Usage

### Linking Results to Paper

```bash
# Automatic linking (recommended after experiment)
python scripts/link_results_to_paper.py --auto

# Manual linking for specific table
python scripts/link_results_to_paper.py \
    --experiment seed_42 \
    --table table_2_rouge_scores \
    --values baseline_rouge_l=0.250,monotonic_rouge_l=0.242
```

### Verifying Provenance

```bash
# Check all paper tables are linked
cat provenance.json | jq '.tables | keys'

# Check specific table provenance
cat provenance.json | jq '.tables.table_2_rouge_scores'

# Verify experimental source exists
cat provenance.json | jq '.tables.table_2_rouge_scores.source_files[]' | \
    xargs -I {} test -f {} && echo "✓ Source files exist"
```

### Creating Paper-Ready Tables

```bash
# Extract data in LaTeX-ready format
python scripts/create_paper_tables.py \
    --from experiment_results/t5_experiments/aggregated/all_seeds_summary.json \
    --table table_3 \
    --output paper_evidence/table_3_multiseed_training.tex
```

## Table Files

Each `table_X_*.json` contains:

```json
{
  "table_id": "table_2_rouge_scores",
  "paper_location": "Section 4.1, Table 2",
  "caption": "Summarization quality on CNN/DailyMail",
  "data": {
    "standard_t5": {
      "rouge_1": {"mean": 0.326, "ci_lower": 0.309, "ci_upper": 0.342},
      "rouge_2": {"mean": 0.119, "ci_lower": 0.105, "ci_upper": 0.134},
      "rouge_l": {"mean": 0.266, "ci_lower": 0.250, "ci_upper": 0.281}
    },
    "baseline_t5": {
      "(...)": "similar structure"
    },
    "monotonic_t5": {
      "(...)": "similar structure"
    }
  },
  "source_experiment": "t5_small_seed42_20260127",
  "verified_date": "2026-01-27",
  "notes": "Seed 42 results, 200 samples from CNN/DM test set"
}
```

## Guidelines

1. **Never modify values manually** - Always extract from experimental results
2. **Always record provenance** - Use `link_results_to_paper.py`
3. **Verify before submission** - Cross-check paper values against source files
4. **Update on revisions** - If re-running experiments, update provenance
5. **Keep audit trail** - Provenance shows full history of paper values

## Verification Checklist

Before paper submission:

- [ ] All tables have entries in `provenance.json`
- [ ] All source files referenced exist and are in git
- [ ] All extracted values match source files
- [ ] All experiments have complete metadata
- [ ] All provenance entries marked `verified: true`

```bash
# Run verification
python scripts/verify_paper_provenance.py

# Should output:
# ✓ All tables linked to experiments
# ✓ All source files exist
# ✓ All values match sources
# ✓ All metadata complete
```

---

This directory ensures complete transparency and reproducibility of all paper claims.
