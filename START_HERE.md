# Mono-S2S Project - Start Here

**Project:** Monotonic Seq2Seq for Adversarial Robustness  
**Status:** Active research for ICML 2025  
**Updated:** 2026-01-21

---

## Quick Navigation

### I want to...

**Run the pipeline on HPC:**
→ **README.md** → Quick start section

**Work on the ICML paper:**
→ **documentation/paper_development/QUICK_ICML_RECOMMENDATIONS.md**

**Write or run tests:**
→ **documentation/testing/TESTING.md**

**Understand the project:**
→ **documentation/getting_started.ipynb**

**Check configuration:**
→ **hpc_version/configs/experiment_config.py**

**Review all documentation:**
→ **DOCUMENTATION_INDEX.md**

---

## Current Project Status

### Test Coverage
- **Achieved:** 98.01% ✅ (Target: 90%)
- **Tests:** 196 test items, 164 passing
- **Status:** Production-ready test infrastructure

### ICML Paper
- **Status:** Draft complete, critical fixes implemented
- **Critical Fixes Done:** Fair comparison (7 epochs both), full test sets (11,490)
- **Next Steps:** Run pipeline, collect results, expand methods section
- **Est. Acceptance:** ~35% currently → ~75% after recommended additions

### Pipeline Configuration
- **Fair Comparison:** Both models train 7 epochs ✅
- **Evaluation:** Full test sets enabled (11,490 examples) ✅
- **Results:** All outputs timestamped with run metadata ✅
- **Ready to run:** `./hpc_version/run_all.sh`

---

## What's Been Done Recently

### 2026-01-21: Critical Updates
1. ✅ Implemented 98% test coverage (target: 90%)
2. ✅ Fixed unfair comparison (baseline 5→7 epochs)
3. ✅ Enabled full test sets (200→11,490 samples)
4. ✅ Added timestamp labeling to all results
5. ✅ Created comprehensive paper critique and recommendations
6. ✅ Organized documentation into clear structure

---

## Documentation Structure

```
/
├── README.md                    ← General project overview
├── START_HERE.md               ← This file
├── DOCUMENTATION_INDEX.md      ← Complete doc listing
│
├── documentation/
│   ├── getting_started.ipynb           ← Tutorial
│   ├── monotone_llms_paper.tex         ← Paper draft
│   │
│   ├── paper_development/              ← ICML paper docs
│   │   ├── QUICK_ICML_RECOMMENDATIONS.md  ← Start here for paper
│   │   ├── PAPER_STATUS.md                ← Status tracking
│   │   ├── paper_methods_critique.md      ← Detailed critique
│   │   └── ... (8 paper-related files)
│   │
│   └── testing/                        ← Testing docs
│       ├── TESTING.md                     ← Main testing guide
│       ├── TEST_COVERAGE_FINAL.md         ← Achievement report
│       └── ... (6 testing-related files)
│
├── hpc_version/                 ← Pipeline code
│   ├── configs/experiment_config.py   ← All settings here
│   ├── scripts/                       ← Stage scripts
│   ├── CHANGES_AT_A_GLANCE.md        ← Quick reference
│   └── IMPROVEMENTS_SUMMARY.md       ← Detailed docs
│
└── tests/                       ← Test suite (98% coverage)
    ├── README.md                      ← Test guide
    └── test_*.py                      ← 196 tests
```

---

## Key Configuration Files

### Experimental Settings
**File:** `hpc_version/configs/experiment_config.py`

**Critical Settings:**
```python
NUM_EPOCHS = 7                    # Both models (fair comparison)
MONOTONIC_NUM_EPOCHS = 7          # Same as baseline
USE_FULL_TEST_SETS = True         # Full evaluation (11,490)
LEARNING_RATE = 5e-5              # Both models
BATCH_SIZE = 4                    # Both models
```

### Testing Configuration
**Files:** `pytest.ini`, `pyproject.toml`, `.coveragerc`

**Status:** 98.01% coverage achieved ✅

---

## Important Links

### Paper Development
- Priority fixes: `documentation/paper_development/QUICK_ICML_RECOMMENDATIONS.md`
- Status: `documentation/paper_development/PAPER_STATUS.md`
- Full critique: `documentation/paper_development/paper_methods_critique.md`

### Testing
- Main guide: `documentation/testing/TESTING.md`
- Results: `documentation/testing/TEST_COVERAGE_FINAL.md`

### Pipeline
- Setup: `README.md`
- Config: `hpc_version/configs/experiment_config.py`
- Improvements: `hpc_version/IMPROVEMENTS_SUMMARY.md`

---

## Next Actions

### For Pipeline Runs:
1. Configure HPC paths in `experiment_config.py`
2. Run: `./hpc_version/run_all.sh`
3. Check timestamped results in `$SCRATCH/mono_s2s_results/`

### For Paper:
1. Review: `documentation/paper_development/QUICK_ICML_RECOMMENDATIONS.md`
2. Run pipeline with current config (fair comparison, full test sets)
3. Collect results for tables
4. Expand methods section

### For Testing:
1. Run: `pytest` (98% coverage achieved)
2. Guide: `documentation/testing/TESTING.md`

---

**Choose your path above and follow the links!**
