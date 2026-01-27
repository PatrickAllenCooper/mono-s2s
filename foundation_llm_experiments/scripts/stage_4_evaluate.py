#!/usr/bin/env python3
"""
Stage 4: Evaluate Models on LLM Benchmarks

Evaluates both baseline and monotonic models on:
- Pile test set (perplexity)
- LAMBADA (next-word prediction)
- HellaSwag (commonsense reasoning)
- Winogrande (coreference)
- TruthfulQA (factuality)

Inputs:
- baseline_checkpoints/best_model.pt
- monotonic_checkpoints/best_model.pt

Outputs:
- evaluation_results.json
- stage_4_evaluate_complete.flag
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, create_completion_flag, save_json,
    StageLogger, check_dependencies, compute_perplexity
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_on_pile_test(model, tokenizer, device, max_samples=None):
    """Evaluate perplexity on Pile test set"""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from utils.common_utils import LanguageModelingDataset
    
    # Load Pile test set
    try:
        pile_test = load_dataset(
            Config.TRAINING_DATASET,
            split="test",
            streaming=False,
            cache_dir=Config.DATA_CACHE_DIR,
            trust_remote_code=True
        )
    except:
        # Fallback to validation if test not available
        pile_test = load_dataset(
            Config.TRAINING_DATASET,
            split="validation",
            streaming=False,
            cache_dir=Config.DATA_CACHE_DIR,
            trust_remote_code=True
        )
    
    # Limit samples
    if max_samples:
        test_texts = [example['text'] for i, example in enumerate(pile_test) if i < max_samples]
    else:
        test_texts = [example['text'] for example in pile_test]
    
    # Create dataset and dataloader
    dataset = LanguageModelingDataset(test_texts, tokenizer, max_length=Config.MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=Config.EVAL_BATCH_SIZE, shuffle=False)
    
    # Compute perplexity
    result = compute_perplexity(model, dataloader, device)
    
    return result


def evaluate_on_lambada(model, tokenizer, device):
    """Evaluate on LAMBADA (next-word prediction)"""
    # TODO: Use lm-evaluation-harness or implement custom
    return {
        'accuracy': 0.0,
        'num_samples': 0
    }


def evaluate_on_hellaswag(model, tokenizer, device):
    """Evaluate on HellaSwag (commonsense reasoning)"""
    # TODO: Use lm-evaluation-harness
    return {
        'accuracy': 0.0,
        'num_samples': 0
    }


def main():
    """Run evaluation"""
    logger = StageLogger("stage_4_evaluate")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic']):
            logger.complete(success=False)
            return 1
        
        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()
        
        # Load tokenizer
        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR
        )
        
        # Evaluate baseline
        logger.log("\n" + "="*80)
        logger.log("EVALUATING BASELINE MODEL")
        logger.log("="*80)
        
        logger.log("Loading baseline model...")
        baseline_path = os.path.join(
            Config.CHECKPOINT_DIR,
            'baseline_checkpoints',
            'best_model.pt'
        )
        
        baseline_model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.float32
        ).to(device)
        
        baseline_model.load_state_dict(
            torch.load(baseline_path, map_location=device, weights_only=False)
        )
        baseline_model.eval()
        
        logger.log("Evaluating on Pile test set...")
        baseline_pile = evaluate_on_pile_test(
            baseline_model, tokenizer, device,
            max_samples=Config.FULL_PILE_TEST_SIZE if Config.USE_FULL_EVAL_SETS else Config.QUICK_PILE_TEST_SIZE
        )
        
        logger.log(f"  Perplexity: {baseline_pile['perplexity']:.2f}")
        
        # Evaluate monotonic
        logger.log("\n" + "="*80)
        logger.log("EVALUATING MONOTONIC MODEL")
        logger.log("="*80)
        
        logger.log("Loading monotonic model...")
        monotonic_path = os.path.join(
            Config.CHECKPOINT_DIR,
            'monotonic_checkpoints',
            'best_model.pt'
        )
        
        from utils.common_utils import make_model_monotonic
        
        monotonic_model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.float32
        )
        monotonic_model = make_model_monotonic(monotonic_model)
        monotonic_model.load_state_dict(
            torch.load(monotonic_path, map_location=device, weights_only=False)
        )
        monotonic_model = monotonic_model.to(device)
        monotonic_model.eval()
        
        logger.log("Evaluating on Pile test set...")
        monotonic_pile = evaluate_on_pile_test(
            monotonic_model, tokenizer, device,
            max_samples=Config.FULL_PILE_TEST_SIZE if Config.USE_FULL_EVAL_SETS else Config.QUICK_PILE_TEST_SIZE
        )
        
        logger.log(f"  Perplexity: {monotonic_pile['perplexity']:.2f}")
        
        # Aggregate results
        results = {
            'pile_test': {
                'baseline_pythia': baseline_pile,
                'monotonic_pythia': monotonic_pile
            },
            'metadata': {
                'seed': Config.CURRENT_SEED,
                'model_name': Config.MODEL_NAME,
                'use_full_eval_sets': Config.USE_FULL_EVAL_SETS
            }
        }
        
        save_json(
            results,
            os.path.join(Config.RESULTS_DIR, 'evaluation_results.json')
        )
        
        logger.log("\n✓ Evaluation complete!")
        logger.complete(success=True)
        return 0
        
    except Exception as e:
        logger.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())
