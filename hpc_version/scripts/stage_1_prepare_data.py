#!/usr/bin/env python3
"""
Stage 1: Data Preparation

This stage:
1. Loads all training datasets (7 sources)
2. Loads all validation datasets
3. Loads all test datasets (CNN/DM, XSUM, SAMSum)
4. Loads attack datasets (validation for opt, test for eval)
5. Caches data to disk for faster loading

Dependencies: stage_0_setup

Outputs:
- train_data.pt (training texts + summaries)
- val_data.pt (validation texts + summaries)
- test_data.pt (all test sets)
- attack_data.pt (attack optimization + evaluation sets)
- data_statistics.json
- stage_1_complete.flag
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, load_dataset_split, check_dependencies,
    save_json, StageLogger
)

def main():
    """Run data preparation stage"""
    logger = StageLogger("stage_1_data_prep")
    
    # Check dependencies
    if not check_dependencies(["stage_0_setup"]):
        return 1
    
    try:
        # Set seeds
        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        
        # ===================================================================
        # Load Training Data
        # ===================================================================
        logger.log("\n" + "="*80)
        logger.log("LOADING TRAINING DATA (train splits)")
        logger.log("="*80)
        
        train_texts_all = []
        train_summaries_all = []
        
        for dataset_name, text_field, summary_field, label in [
            ("knkarthick/dialogsum", "dialogue", "summary", "DialogSum"),
            ("samsum", "dialogue", "summary", "SAMSum"),
            ("EdinburghNLP/xsum", "document", "summary", "XSUM"),
            ("knkarthick/AMI", "dialogue", "summary", "AMI"),
            ("knkarthick/highlightsum", "dialogue", "summary", "HighlightSum"),
            ("ccdv/arxiv-summarization", "article", "abstract", "arXiv"),
            ("knkarthick/MEETING_SUMMARY", "dialogue", "summary", "MEETING_SUMMARY"),
        ]:
            texts, summaries = load_dataset_split(
                dataset_name, "train", text_field, summary_field
            )
            train_texts_all.extend(texts)
            train_summaries_all.extend(summaries)
            logger.log(f"  {label}: {len(texts)} samples")
        
        logger.log(f"\n✓ Total training samples: {len(train_texts_all)}")
        
        # ===================================================================
        # Load Validation Data (CRITICAL: validation splits, NOT test!)
        # ===================================================================
        logger.log("\n" + "="*80)
        logger.log("LOADING VALIDATION DATA (validation splits - NO TEST!)")
        logger.log("="*80)
        
        val_texts_all = []
        val_summaries_all = []
        
        for dataset_name, text_field, summary_field, label in [
            ("knkarthick/dialogsum", "dialogue", "summary", "DialogSum"),
            ("samsum", "dialogue", "summary", "SAMSum"),
            ("EdinburghNLP/xsum", "document", "summary", "XSUM"),
            ("ccdv/arxiv-summarization", "article", "abstract", "arXiv"),
        ]:
            texts, summaries = load_dataset_split(
                dataset_name, "validation", text_field, summary_field
            )
            val_texts_all.extend(texts)
            val_summaries_all.extend(summaries)
            logger.log(f"  {label}: {len(texts)} samples")
        
        logger.log(f"\n✓ Total validation samples: {len(val_texts_all)}")
        logger.log(f"  ⚠️  CRITICAL: Using VALIDATION splits (test sets untouched)")
        
        # ===================================================================
        # Load Test Data (for final evaluation)
        # ===================================================================
        logger.log("\n" + "="*80)
        logger.log("LOADING TEST DATA (held-out evaluation)")
        logger.log("="*80)
        
        test_data = {}
        
        # CNN/DailyMail
        logger.log("Loading CNN/DailyMail test...")
        cnn_texts, cnn_sums = load_dataset_split(
            "cnn_dailymail", "test", "article", "highlights", config="3.0.0",
            max_samples=ExperimentConfig.QUICK_TEST_SIZE if not ExperimentConfig.USE_FULL_TEST_SETS else None
        )
        test_data['cnn_dm'] = {'texts': cnn_texts, 'summaries': cnn_sums}
        
        # XSUM
        logger.log("Loading XSUM test...")
        xsum_texts, xsum_sums = load_dataset_split(
            "EdinburghNLP/xsum", "test", "document", "summary",
            max_samples=ExperimentConfig.QUICK_TEST_SIZE if not ExperimentConfig.USE_FULL_TEST_SETS else None
        )
        test_data['xsum'] = {'texts': xsum_texts, 'summaries': xsum_sums}
        
        # SAMSum
        logger.log("Loading SAMSum test...")
        samsum_texts, samsum_sums = load_dataset_split(
            "samsum", "test", "dialogue", "summary",
            max_samples=ExperimentConfig.QUICK_TEST_SIZE if not ExperimentConfig.USE_FULL_TEST_SETS else None
        )
        test_data['samsum'] = {'texts': samsum_texts, 'summaries': samsum_sums}
        
        total_test = len(cnn_texts) + len(xsum_texts) + len(samsum_texts)
        logger.log(f"\n✓ Total test samples: {total_test}")
        
        # ===================================================================
        # Load Attack Data (validation for opt, test for eval)
        # ===================================================================
        logger.log("\n" + "="*80)
        logger.log("LOADING ATTACK DATA (held-out evaluation)")
        logger.log("="*80)
        
        # Trigger optimization (from validation split)
        logger.log("Loading trigger optimization data (CNN/DM validation)...")
        attack_opt_size = (ExperimentConfig.TRIGGER_OPT_SIZE_QUICK 
                          if not ExperimentConfig.USE_FULL_TEST_SETS 
                          else ExperimentConfig.TRIGGER_OPT_SIZE_FULL)
        
        attack_opt_texts, attack_opt_sums = load_dataset_split(
            "cnn_dailymail", "validation", "article", "highlights", 
            config="3.0.0", max_samples=attack_opt_size
        )
        
        # Attack evaluation (from test split - disjoint from optimization)
        logger.log("Loading attack evaluation data (CNN/DM test)...")
        attack_eval_size = (ExperimentConfig.TRIGGER_EVAL_SIZE_QUICK
                           if not ExperimentConfig.USE_FULL_TEST_SETS
                           else ExperimentConfig.TRIGGER_EVAL_SIZE_FULL)
        
        attack_eval_texts, attack_eval_sums = load_dataset_split(
            "cnn_dailymail", "test", "article", "highlights",
            config="3.0.0", max_samples=attack_eval_size
        )
        
        attack_data = {
            'optimization': {'texts': attack_opt_texts, 'summaries': attack_opt_sums},
            'evaluation': {'texts': attack_eval_texts, 'summaries': attack_eval_sums}
        }
        
        logger.log(f"  Trigger optimization: {len(attack_opt_texts)} samples (validation split)")
        logger.log(f"  Attack evaluation: {len(attack_eval_texts)} samples (test split)")
        logger.log(f"  ✓ Held-out evaluation (validation → test, disjoint)")
        
        # ===================================================================
        # Save All Data
        # ===================================================================
        logger.log("\n" + "="*80)
        logger.log("SAVING PREPARED DATA")
        logger.log("="*80)
        
        # Save training data
        train_data_path = os.path.join(ExperimentConfig.DATA_CACHE_DIR, "train_data.pt")
        torch.save({
            'texts': train_texts_all,
            'summaries': train_summaries_all
        }, train_data_path)
        logger.log(f"✓ Training data: {train_data_path}")
        
        # Save validation data
        val_data_path = os.path.join(ExperimentConfig.DATA_CACHE_DIR, "val_data.pt")
        torch.save({
            'texts': val_texts_all,
            'summaries': val_summaries_all
        }, val_data_path)
        logger.log(f"✓ Validation data: {val_data_path}")
        
        # Save test data
        test_data_path = os.path.join(ExperimentConfig.DATA_CACHE_DIR, "test_data.pt")
        torch.save(test_data, test_data_path)
        logger.log(f"✓ Test data: {test_data_path}")
        
        # Save attack data
        attack_data_path = os.path.join(ExperimentConfig.DATA_CACHE_DIR, "attack_data.pt")
        torch.save(attack_data, attack_data_path)
        logger.log(f"✓ Attack data: {attack_data_path}")
        
        # ===================================================================
        # Save Statistics
        # ===================================================================
        statistics = {
            "stage": "data_preparation",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "seed": ExperimentConfig.CURRENT_SEED,
            "training": {
                "total_samples": len(train_texts_all),
                "sources": 7,
                "splits": "train"
            },
            "validation": {
                "total_samples": len(val_texts_all),
                "sources": 4,
                "splits": "validation (NOT test - no leakage!)"
            },
            "test": {
                "cnn_dm": len(test_data['cnn_dm']['texts']),
                "xsum": len(test_data['xsum']['texts']),
                "samsum": len(test_data['samsum']['texts']),
                "total": total_test,
                "splits": "test (held-out from training)"
            },
            "attack": {
                "optimization_samples": len(attack_opt_texts),
                "optimization_source": "CNN/DM validation",
                "evaluation_samples": len(attack_eval_texts),
                "evaluation_source": "CNN/DM test (disjoint from optimization)"
            }
        }
        
        stats_path = os.path.join(ExperimentConfig.RESULTS_DIR, "data_statistics.json")
        save_json(statistics, stats_path)
        
        logger.log("\n" + "="*80)
        logger.log("DATA PREPARATION SUMMARY")
        logger.log("="*80)
        logger.log(f"Training:   {len(train_texts_all):,} samples (7 datasets, train splits)")
        logger.log(f"Validation: {len(val_texts_all):,} samples (validation splits - NO TEST!)")
        logger.log(f"Test:       {total_test:,} samples (3 datasets, test splits)")
        logger.log(f"Attack Opt: {len(attack_opt_texts):,} samples (CNN/DM validation)")
        logger.log(f"Attack Eval: {len(attack_eval_texts):,} samples (CNN/DM test)")
        logger.log("="*80)
        
        return logger.complete(success=True)
        
    except Exception as e:
        logger.log(f"\n❌ ERROR in data preparation: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return logger.complete(success=False)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if isinstance(exit_code, int) else (0 if exit_code else 1))

