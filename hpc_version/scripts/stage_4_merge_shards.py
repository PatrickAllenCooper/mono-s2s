#!/usr/bin/env python3
"""
Stage 4 (merge): Combine per-dataset evaluation shards into one file.

When USE_FULL_TEST_SETS=1, a single Stage 4 job evaluating all three
datasets (CNN/DailyMail, XSUM, SAMSum) x three models can take ~40h+,
longer than CURC's 24h gpu-normal QoS ceiling. stage_4_evaluate.py's
EVAL_DATASET_FILTER lets that work be split into independent per-dataset
SLURM jobs instead, each writing 'evaluation_results_<shard>.json' and a
'stage_4_evaluate_<shard>' completion flag. This script combines those
shard files back into the single 'evaluation_results.json' (and
'stage_4_evaluate' flag) that Stage 5/6/7 expect, so nothing downstream
needs to know evaluation ran as multiple jobs.

Inputs:
- evaluation_results_<shard>.json for each shard in EVAL_SHARDS

Outputs:
- evaluation_results.json (combined)
- stage_4_evaluate_complete.flag
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import check_dependencies, load_json, save_json, StageLogger


def main():
    logger = StageLogger("stage_4_evaluate")

    try:
        logger.log("Checking dependencies...")
        required = ['stage_0_setup', 'stage_1_data_prep',
                    'stage_2_train_baseline', 'stage_3_train_monotonic']
        if not check_dependencies(required):
            logger.complete(success=False)
            return 1

        shards_raw = os.environ.get("EVAL_SHARDS", "cnn_dm,xsum,samsum")
        shards = [s.strip() for s in shards_raw.split(",") if s.strip()]
        logger.log(f"Merging shards: {shards}")

        combined = {}
        metadata = None
        for shard in shards:
            shard_file = os.path.join(
                ExperimentConfig.RESULTS_DIR, f"evaluation_results_{shard}.json"
            )
            if not os.path.exists(shard_file):
                raise FileNotFoundError(
                    f"Missing shard result file: {shard_file}\n"
                    f"Did the '{shard}' Stage 4 shard job complete successfully?"
                )
            shard_data = load_json(shard_file)
            for key, value in shard_data.items():
                if key == 'metadata':
                    if metadata is None:
                        metadata = value
                    continue
                if key in combined:
                    raise ValueError(
                        f"Dataset key '{key}' present in more than one shard "
                        f"-- shards must partition the dataset list, not overlap."
                    )
                combined[key] = value
            logger.log(f"  ✓ Loaded {shard_file} ({[k for k in shard_data if k != 'metadata']})")

        if metadata is not None:
            combined['metadata'] = metadata

        results_file = os.path.join(ExperimentConfig.RESULTS_DIR, 'evaluation_results.json')
        save_json(combined, results_file)
        logger.log(f"\n✓ Combined results saved to: {results_file}")
        logger.log(f"  Datasets present: {[k for k in combined if k != 'metadata']}")

        logger.complete(success=True)
        return 0

    except Exception as e:
        logger.log(f"\nERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())
