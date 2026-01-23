# Documentation Index

## Quick Links by Purpose

### 🚀 Getting Started
- **README.md** - Start here: setup, pipeline overview, quick start
- **documentation/getting_started.ipynb** - Interactive tutorial for collaborators

### 📊 Running Experiments
- **hpc_version/configs/experiment_config.py** - All configuration settings
- **hpc_version/CHANGES_AT_A_GLANCE.md** - Pipeline improvements quick reference
- **hpc_version/IMPROVEMENTS_SUMMARY.md** - Detailed improvement documentation

### 📝 Paper Development (ICML 2025)
- **documentation/monotone_llms_paper.tex** - Paper draft
- **documentation/QUICK_ICML_RECOMMENDATIONS.md** - Priority fixes (START HERE)
- **documentation/paper_methods_critique.md** - Detailed methods review
- **documentation/paper_methods_fixes_brief.md** - Bullet-point fixes
- **documentation/ICML_PRIORITY_CHECKLIST.md** - Prioritized action items
- **documentation/ICML_STRENGTHENING_SUGGESTIONS.md** - Comprehensive suggestions
- **PAPER_STATUS.md** - Current paper status and roadmap
- **PAPER_METHODS_FIXES_IMPLEMENTED.md** - What's been fixed

### 🧪 Testing
- **TESTING.md** - Complete testing guide
- **tests/README.md** - Test suite reference
- **TEST_COVERAGE_FINAL.md** - Coverage achievement report (98.01%)
- **TEST_COVERAGE_SUMMARY.md** - Implementation summary
- **COVERAGE_JOURNEY.md** - How we got to 98% coverage
- **COVERAGE_STATUS.md** - Coverage analysis
- **README_TESTING.md** - Testing quick reference

### 📈 Results Tracking
All results saved with timestamp metadata:
- `_metadata.timestamp`: Full datetime (YYYY-MM-DD HH:MM:SS)
- `_metadata.run_id`: Unique run identifier (YYYYMMDD_HHMMSS_seedXX)
- `_metadata.seed`: Random seed used
- Located in: `$SCRATCH/mono_s2s_results/` and `$PROJECT/mono_s2s_final_results/`

---

## Documentation Organization

### By Audience

#### For New Users
1. README.md
2. documentation/getting_started.ipynb
3. hpc_version/CHANGES_AT_A_GLANCE.md

#### For Paper Authors
1. documentation/QUICK_ICML_RECOMMENDATIONS.md ← Start here
2. PAPER_STATUS.md
3. documentation/paper_methods_critique.md

#### For Developers
1. TESTING.md
2. tests/README.md
3. hpc_version/configs/experiment_config.py

### By Topic

#### Pipeline Setup & Usage
- README.md - Quick start
- hpc_version/configs/experiment_config.py - Configuration
- hpc_version/IMPROVEMENTS_SUMMARY.md - Pipeline details

#### ICML Paper (14 files)
- documentation/monotone_llms_paper.tex - Paper draft
- documentation/QUICK_ICML_RECOMMENDATIONS.md - Quick priorities
- documentation/ICML_PRIORITY_CHECKLIST.md - Detailed checklist
- documentation/ICML_STRENGTHENING_SUGGESTIONS.md - All suggestions
- documentation/paper_methods_critique.md - Full critique
- documentation/paper_methods_fixes_brief.md - Brief fixes
- PAPER_STATUS.md - Status summary
- PAPER_METHODS_FIXES_IMPLEMENTED.md - Implementation log

#### Testing (7 files)
- TESTING.md - Main testing guide
- tests/README.md - Test suite details
- TEST_COVERAGE_FINAL.md - Achievement report
- TEST_COVERAGE_SUMMARY.md - Implementation details
- COVERAGE_JOURNEY.md - Progress story
- COVERAGE_STATUS.md - Technical analysis
- README_TESTING.md - Quick reference

#### HPC Pipeline (2 files)
- hpc_version/IMPROVEMENTS_SUMMARY.md - Detailed improvements
- hpc_version/CHANGES_AT_A_GLANCE.md - Quick reference

---

## File Status

### Active (Keep Updated)
- README.md - Main documentation
- PAPER_STATUS.md - Paper progress
- documentation/QUICK_ICML_RECOMMENDATIONS.md - Current priorities
- hpc_version/configs/experiment_config.py - Settings

### Reference (Snapshot in Time)
- TEST_COVERAGE_FINAL.md - Coverage achievement (2026-01-21)
- PAPER_METHODS_FIXES_IMPLEMENTED.md - Fixes log (2026-01-21)
- COVERAGE_JOURNEY.md - Implementation story

### Detailed Guides (Stable)
- TESTING.md - Testing procedures
- documentation/paper_methods_critique.md - Methods analysis
- documentation/ICML_STRENGTHENING_SUGGESTIONS.md - Comprehensive suggestions

---

## Quick Navigation

### I want to...

**Run the pipeline:**
→ README.md → Quick start section

**Understand paper issues:**
→ documentation/QUICK_ICML_RECOMMENDATIONS.md

**Write tests:**
→ TESTING.md

**Check configuration:**
→ hpc_version/configs/experiment_config.py

**Review results:**
→ Check timestamped files in `$SCRATCH/mono_s2s_results/`

**Submit to ICML:**
→ PAPER_STATUS.md → Must-have checklist

---

## Maintenance

### Keep Updated
- README.md when pipeline changes
- PAPER_STATUS.md when paper progresses
- experiment_config.py when settings change

### Archive Old Docs
- Move outdated guides to `documentation/archive/`
- Keep index current

### Add New Docs
- Follow naming conventions
- Add entry to this index
- Link from README if user-facing

---

**Last Updated:** 2026-01-21  
**Total Documentation Files:** 17 markdown files + 1 notebook + 1 paper
