# Stage Scripts Directory

This directory contains Python scripts for each experimental stage.

## Current Status

### ✅ Implemented
- **stage_0_setup.py** - Environment setup and model download (fully implemented)
- **stage_1_prepare_data.py** - Data loading and caching (fully implemented)

### 📝 To Be Implemented
- **stage_2_train_baseline.py** - Extract from main code lines ~1456-1549
- **stage_3_train_monotonic.py** - Extract from main code lines ~1550-1643  
- **stage_4_evaluate.py** - Extract from main code lines ~2167-2310
- **stage_5_uat_attacks.py** - Extract from main code lines ~2824-3020
- **stage_6_hotflip_attacks.py** - Extract from main code lines ~3631-3950
- **stage_7_aggregate.py** - Combine all results, generate final analysis

## Implementation Pattern

Each stage script follows this pattern:

```python
#!/usr/bin/env python3
"""
Stage N: <Description>

Dependencies: stage_X, stage_Y
Outputs: <files created>
"""

import os
import sys

# Set env vars BEFORE torch import
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, check_dependencies, StageLogger,
    save_json, load_json
)

def main():
    logger = StageLogger("stage_N_name")
    
    # Check dependencies
    if not check_dependencies(["stage_X", "stage_Y"]):
        return 1
    
    try:
        # Set seeds
        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        
        # Load data from previous stages
        # ... load inputs ...
        
        # Do work
        logger.log("Doing work...")
        # ... main logic ...
        
        # Save outputs
        # ... save results ...
        
        # Mark complete
        return logger.complete(success=True)
        
    except Exception as e:
        logger.log(f"ERROR: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return logger.complete(success=False)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if isinstance(exit_code, int) else 1)
```

## How to Complete Implementation

1. **For training stages (2, 3):**
   - Extract `T5Trainer` class from common_utils.py or main code
   - Load train/val data from `data_cache/`
   - Create DataLoaders with worker_init_fn
   - Instantiate trainer (is_monotonic=False for baseline, True for monotonic)
   - Run training loop
   - Save checkpoints to `checkpoints/{baseline,monotonic}_checkpoints/`

2. **For evaluation stage (4):**
   - Load all three models (standard, baseline, monotonic)
   - Load test datasets from `data_cache/`
   - Call `evaluate_model_comprehensive()` for each model on each dataset
   - Save results with bootstrap CIs

3. **For attack stages (5, 6):**
   - Load all three models
   - Load attack data from `data_cache/`
   - Run UAT/HotFlip attacks on each model
   - For UAT: implement transfer matrix
   - Save attack results and trigger info

4. **For aggregation stage (7):**
   - Load all previous results (JSON files)
   - Combine into final analysis
   - Generate comparison tables
   - Create visualizations
   - Copy to project directory for persistence

## Extraction Guide

To extract code from `../mono_s2s_v1_7.py`:

```bash
# Find the relevant section
grep -n "Stage name" ../mono_s2s_v1_7.py

# Extract lines to new script
sed -n 'START,ENDp' ../mono_s2s_v1_7.py > stage_N_name.py

# Adapt:
# - Add imports at top
# - Wrap in main() function
# - Add StageLogger
# - Add dependency checks
# - Load data from cached files
# - Save outputs to standard locations
```

## Testing Individual Stages

```bash
# Test a single stage locally
cd scripts
python stage_0_setup.py  # Should complete without errors

# Check output
cat $SCRATCH/mono_s2s_results/setup_complete.json

# Check completion flag
ls $SCRATCH/mono_s2s_work/stage_0_setup_complete.flag
```

---

**Note:** The two implemented stages (0, 1) serve as templates for implementing the remaining stages. Follow the same pattern for consistency.

