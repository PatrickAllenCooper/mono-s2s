#!/usr/bin/env python3
"""
Stage 4: Comprehensive Model Evaluation

This stage evaluates ALL THREE models on all test datasets with:
- Fixed decoding parameters for fair comparison
- Bootstrap 95% confidence intervals
- Length statistics and brevity penalty
- Comprehensive ROUGE metrics

Models evaluated:
1. Standard T5 (pre-trained, not fine-tuned) - Reference
2. Baseline T5 (fine-tuned, unconstrained) - Fair baseline
3. Monotonic T5 (fine-tuned, W≥0 FFN constraints) - Treatment

Test datasets:
- CNN/DailyMail v3.0.0 (test split)
- XSUM (test split)
- SAMSum (test split)

Inputs:
- test_data.pt (from stage 1)
- baseline_checkpoints/best_model.pt (from stage 2)
- monotonic_checkpoints/best_model.pt (from stage 3)
- Standard T5 checkpoint (downloads if needed)

Outputs:
- evaluation_results.json (PRIMARY RESULTS with bootstrap CIs)
- stage_4_evaluate_complete.flag
"""

# Set environment variables BEFORE importing torch
import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import time
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, create_completion_flag, check_dependencies,
    save_json, StageLogger, load_model, generate_summary_fixed_params,
    compute_rouge_with_ci, compute_length_statistics, compute_brevity_penalty
)

# Import transformers AFTER environment setup
from transformers import T5Tokenizer


def evaluate_model_on_dataset(model, model_name, texts, references, 
                               tokenizer, device, logger, dataset_name="Dataset"):
    """
    Evaluate a single model on a single dataset.
    
    Returns dictionary with:
    - predictions
    - rouge_scores (with bootstrap CIs)
    - length_stats
    - brevity_penalty
    """
    logger.log(f"\nEvaluating {model_name} on {dataset_name}...")
    logger.log(f"  Samples: {len(texts)}")
    
    # Generate summaries with FIXED decoding parameters
    model.eval()
    predictions = []
    
    for text in tqdm(texts, desc=f"{model_name} - {dataset_name}"):
        summary = generate_summary_fixed_params(model, text, tokenizer, device)
        predictions.append(summary)
    
    logger.log(f"  ✓ Generated {len(predictions)} summaries")
    
    # Compute ROUGE with bootstrap CIs
    logger.log(f"  Computing ROUGE scores with bootstrap CIs...")
    rouge_scores, all_scores = compute_rouge_with_ci(
        predictions, 
        references,
        metrics=ExperimentConfig.ROUGE_METRICS,
        use_stemmer=ExperimentConfig.ROUGE_USE_STEMMER,
        n_bootstrap=ExperimentConfig.ROUGE_BOOTSTRAP_SAMPLES,
        confidence=0.95
    )
    
    logger.log(f"  ✓ ROUGE-1: {rouge_scores['rouge1']['mean']:.4f} "
               f"[{rouge_scores['rouge1']['lower']:.4f}, {rouge_scores['rouge1']['upper']:.4f}]")
    logger.log(f"  ✓ ROUGE-2: {rouge_scores['rouge2']['mean']:.4f} "
               f"[{rouge_scores['rouge2']['lower']:.4f}, {rouge_scores['rouge2']['upper']:.4f}]")
    logger.log(f"  ✓ ROUGE-L: {rouge_scores['rougeLsum']['mean']:.4f} "
               f"[{rouge_scores['rougeLsum']['lower']:.4f}, {rouge_scores['rougeLsum']['upper']:.4f}]")
    
    # Compute length statistics
    pred_length_stats = compute_length_statistics(predictions, tokenizer)
    ref_length_stats = compute_length_statistics(references, tokenizer)
    
    # Compute brevity penalty
    brevity_stats = compute_brevity_penalty(predictions, references, tokenizer)
    
    logger.log(f"  Prediction length: {pred_length_stats['mean']:.1f} ± "
               f"{pred_length_stats['std']:.1f} tokens")
    logger.log(f"  Reference length: {ref_length_stats['mean']:.1f} ± "
               f"{ref_length_stats['std']:.1f} tokens")
    logger.log(f"  Brevity penalty: {brevity_stats['brevity_penalty']:.4f}")
    logger.log(f"  Length ratio: {brevity_stats['length_ratio']:.4f}")
    
    return {
        'predictions': predictions,
        'rouge_scores': rouge_scores,
        'rouge_all_samples': all_scores,
        'prediction_length_stats': pred_length_stats,
        'reference_length_stats': ref_length_stats,
        'brevity_penalty': brevity_stats
    }


def main():
    """Run comprehensive evaluation"""
    logger = StageLogger("stage_4_evaluate")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        required = ['stage_0_setup', 'stage_1_data_prep', 
                   'stage_2_train_baseline', 'stage_3_train_monotonic']
        if not check_dependencies(required):
            logger.complete(success=False)
            return 1
        
        # Set seeds
        logger.log("Setting random seeds...")
        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        
        # Get device
        device = ExperimentConfig.get_device()
        
        # Load tokenizer
        logger.log("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(ExperimentConfig.MODEL_NAME)
        
        # Load test data
        logger.log("Loading test data...")
        data_cache_dir = ExperimentConfig.DATA_CACHE_DIR
        test_data = torch.load(os.path.join(data_cache_dir, 'test_data.pt'))
        
        logger.log(f"  CNN/DailyMail: {len(test_data['cnn_dm']['texts'])} samples")
        logger.log(f"  XSUM: {len(test_data['xsum']['texts'])} samples")
        logger.log(f"  SAMSum: {len(test_data['samsum']['texts'])} samples")
        
        # Load models
        logger.log("\n" + "="*80)
        logger.log("LOADING MODELS")
        logger.log("="*80)
        
        # 1. Standard T5 (pre-trained, reference)
        logger.log("\n1. Loading Standard T5 (pre-trained)...")
        model_standard, _ = load_model('standard', checkpoint_path=None, device=device)
        logger.log("✓ Standard T5 loaded (pre-trained, not fine-tuned)")
        
        # 2. Baseline T5 (fine-tuned, unconstrained)
        logger.log("\n2. Loading Baseline T5 (fine-tuned, unconstrained)...")
        baseline_checkpoint = os.path.join(
            ExperimentConfig.CHECKPOINT_DIR,
            'baseline_checkpoints',
            'best_model.pt'
        )
        if not os.path.exists(baseline_checkpoint):
            raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")
        
        model_baseline, _ = load_model('baseline', checkpoint_path=baseline_checkpoint, device=device)
        logger.log(f"✓ Baseline T5 loaded from: {baseline_checkpoint}")
        
        # 3. Monotonic T5 (fine-tuned, W≥0 constraints)
        logger.log("\n3. Loading Monotonic T5 (fine-tuned, W≥0 FFN constraints)...")
        monotonic_checkpoint = os.path.join(
            ExperimentConfig.CHECKPOINT_DIR,
            'monotonic_checkpoints',
            'best_model.pt'
        )
        if not os.path.exists(monotonic_checkpoint):
            raise FileNotFoundError(f"Monotonic checkpoint not found: {monotonic_checkpoint}")
        
        model_monotonic, _ = load_model('monotonic', checkpoint_path=monotonic_checkpoint, device=device)
        logger.log(f"✓ Monotonic T5 loaded from: {monotonic_checkpoint}")
        
        logger.log("\n✓ All three models loaded successfully")
        
        # Evaluate on all datasets
        logger.log("\n" + "="*80)
        logger.log("COMPREHENSIVE EVALUATION")
        logger.log("="*80)
        logger.log(f"Decoding parameters (FIXED for all models):")
        logger.log(f"  num_beams: {ExperimentConfig.DECODE_NUM_BEAMS}")
        logger.log(f"  length_penalty: {ExperimentConfig.DECODE_LENGTH_PENALTY}")
        logger.log(f"  min_new_tokens: {ExperimentConfig.DECODE_MIN_NEW_TOKENS}")
        logger.log(f"  max_new_tokens: {ExperimentConfig.DECODE_MAX_NEW_TOKENS}")
        logger.log(f"  no_repeat_ngram_size: {ExperimentConfig.DECODE_NO_REPEAT_NGRAM_SIZE}")
        logger.log(f"  early_stopping: {ExperimentConfig.DECODE_EARLY_STOPPING}")
        
        # Results structure
        results = {}
        
        # Evaluate on each dataset
        for dataset_key, dataset_name in [
            ('cnn_dm', 'CNN/DailyMail'),
            ('xsum', 'XSUM'),
            ('samsum', 'SAMSum')
        ]:
            logger.log("\n" + "="*80)
            logger.log(f"DATASET: {dataset_name}")
            logger.log("="*80)
            
            texts = test_data[dataset_key]['texts']
            references = test_data[dataset_key]['summaries']
            
            # Skip empty datasets
            if len(texts) == 0:
                logger.log(f"[SKIP] No test samples available for {dataset_name}")
                logger.log(f"  This dataset was not loaded during Stage 1")
                continue
            
            results[dataset_key] = {}
            
            # Evaluate each model
            for model, model_key, model_display_name in [
                (model_standard, 'standard_t5', 'Standard T5'),
                (model_baseline, 'baseline_t5', 'Baseline T5'),
                (model_monotonic, 'monotonic_t5', 'Monotonic T5')
            ]:
                eval_results = evaluate_model_on_dataset(
                    model, model_display_name, texts, references,
                    tokenizer, device, logger, dataset_name
                )
                
                results[dataset_key][model_key] = eval_results
        
        # Save comprehensive results
        logger.log("\n" + "="*80)
        logger.log("SAVING RESULTS")
        logger.log("="*80)
        
        # Prepare results for JSON (remove predictions to save space)
        results_for_json = {}
        for dataset_key in results:
            results_for_json[dataset_key] = {}
            for model_key in results[dataset_key]:
                model_results = results[dataset_key][model_key].copy()
                # Remove full predictions and per-sample scores (keep aggregated only)
                model_results.pop('predictions', None)
                model_results.pop('rouge_all_samples', None)
                results_for_json[dataset_key][model_key] = model_results
        
        # Add metadata
        results_for_json['metadata'] = {
            'seed': ExperimentConfig.CURRENT_SEED,
            'model_name': ExperimentConfig.MODEL_NAME,
            'use_full_test_sets': ExperimentConfig.USE_FULL_TEST_SETS,
            'decoding_params': {
                'num_beams': ExperimentConfig.DECODE_NUM_BEAMS,
                'length_penalty': ExperimentConfig.DECODE_LENGTH_PENALTY,
                'min_new_tokens': ExperimentConfig.DECODE_MIN_NEW_TOKENS,
                'max_new_tokens': ExperimentConfig.DECODE_MAX_NEW_TOKENS,
                'no_repeat_ngram_size': ExperimentConfig.DECODE_NO_REPEAT_NGRAM_SIZE,
                'early_stopping': ExperimentConfig.DECODE_EARLY_STOPPING,
            },
            'rouge_config': {
                'metrics': ExperimentConfig.ROUGE_METRICS,
                'use_stemmer': ExperimentConfig.ROUGE_USE_STEMMER,
                'bootstrap_samples': ExperimentConfig.ROUGE_BOOTSTRAP_SAMPLES,
            }
        }
        
        # Save to file
        results_file = os.path.join(
            ExperimentConfig.RESULTS_DIR,
            'evaluation_results.json'
        )
        save_json(results_for_json, results_file)
        
        logger.log(f"\n✓ Results saved to: {results_file}")
        
        # Print summary
        logger.log("\n" + "="*80)
        logger.log("EVALUATION SUMMARY")
        logger.log("="*80)
        
        for dataset_key, dataset_name in [
            ('cnn_dm', 'CNN/DailyMail'),
            ('xsum', 'XSUM'),
            ('samsum', 'SAMSum')
        ]:
            logger.log(f"\n{dataset_name}:")
            
            # Skip datasets that weren't evaluated (no test samples)
            if dataset_key not in results:
                logger.log(f"  [SKIPPED] No test samples available")
                continue
            
            for model_key, model_display_name in [
                ('standard_t5', 'Standard T5'),
                ('baseline_t5', 'Baseline T5'),
                ('monotonic_t5', 'Monotonic T5')
            ]:
                rouge1 = results[dataset_key][model_key]['rouge_scores']['rouge1']
                rouge2 = results[dataset_key][model_key]['rouge_scores']['rouge2']
                rougeL = results[dataset_key][model_key]['rouge_scores']['rougeLsum']
                
                logger.log(f"  {model_display_name}:")
                logger.log(f"    ROUGE-1: {rouge1['mean']:.4f} [{rouge1['lower']:.4f}, {rouge1['upper']:.4f}]")
                logger.log(f"    ROUGE-2: {rouge2['mean']:.4f} [{rouge2['lower']:.4f}, {rouge2['upper']:.4f}]")
                logger.log(f"    ROUGE-L: {rougeL['mean']:.4f} [{rougeL['lower']:.4f}, {rougeL['upper']:.4f}]")
        
        # Mark complete
        logger.log("\n✓ Comprehensive evaluation complete!")
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

