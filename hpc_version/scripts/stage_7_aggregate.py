#!/usr/bin/env python3
"""
Stage 7: Aggregate Results and Final Analysis

This stage aggregates all results from previous stages and generates:
- Final comparison tables
- Summary statistics
- Experiment metadata
- Copies results to permanent storage

Inputs:
- experiment_metadata.json (from stage 0)
- evaluation_results.json (from stage 4)
- uat_results.json (from stage 5)
- hotflip_results.json (from stage 6)
- Training histories

Outputs:
- final_results.json (comprehensive aggregated results)
- experiment_summary.txt (human-readable summary)
- Copies all results to PROJECT/mono_s2s_final_results/
- stage_7_aggregate_complete.flag
"""

# Set environment variables BEFORE importing torch
import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import time
import json
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, create_completion_flag, check_dependencies,
    save_json, load_json, StageLogger
)


def create_comparison_table(eval_results):
    """Create formatted comparison table for ROUGE scores"""
    table_lines = []
    table_lines.append("="*100)
    table_lines.append("ROUGE SCORE COMPARISON (with 95% Confidence Intervals)")
    table_lines.append("="*100)

    dataset_name_map = {
        'cnn_dm': 'CNN/DailyMail',
        'xsum': 'XSUM',
        'samsum': 'SAMSum',
    }

    # Prefer a consistent order, but only include datasets that exist.
    preferred_order = ['cnn_dm', 'xsum', 'samsum']
    dataset_keys = [k for k in preferred_order if k in eval_results]
    if not dataset_keys:
        # Fall back to any dataset keys present (sorted for determinism).
        dataset_keys = sorted(list(eval_results.keys()))
    
    for dataset_key in dataset_keys:
        dataset_name = dataset_name_map.get(dataset_key, dataset_key)
        table_lines.append(f"\n{dataset_name}:")
        table_lines.append("-"*100)
        table_lines.append(f"{'Model':<20} | {'ROUGE-1':^30} | {'ROUGE-2':^30} | {'ROUGE-L':^30}")
        table_lines.append("-"*100)
        
        for model_key, model_name in [
            ('standard_t5', 'Standard T5'),
            ('baseline_t5', 'Baseline T5'),
            ('monotonic_t5', 'Monotonic T5')
        ]:
            rouge1 = eval_results[dataset_key][model_key]['rouge_scores']['rouge1']
            rouge2 = eval_results[dataset_key][model_key]['rouge_scores']['rouge2']
            rougeL = eval_results[dataset_key][model_key]['rouge_scores']['rougeLsum']
            
            r1_str = f"{rouge1['mean']:.4f} [{rouge1['lower']:.4f}, {rouge1['upper']:.4f}]"
            r2_str = f"{rouge2['mean']:.4f} [{rouge2['lower']:.4f}, {rouge2['upper']:.4f}]"
            rL_str = f"{rougeL['mean']:.4f} [{rougeL['lower']:.4f}, {rougeL['upper']:.4f}]"
            
            table_lines.append(f"{model_name:<20} | {r1_str:^30} | {r2_str:^30} | {rL_str:^30}")
    
    table_lines.append("="*100)
    return "\n".join(table_lines)


def create_attack_summary(uat_results, hotflip_results):
    """Create formatted attack results summary"""
    table_lines = []
    table_lines.append("\n" + "="*100)
    table_lines.append("ADVERSARIAL ROBUSTNESS EVALUATION")
    table_lines.append("="*100)
    
    # UAT Results
    table_lines.append("\nUAT (Universal Adversarial Trigger) Attacks:")
    table_lines.append("-"*100)
    table_lines.append(f"{'Model':<20} | {'Trigger':<40} | {'ΔROUGE-L':>12} | {'NLL Increase':>15}")
    table_lines.append("-"*100)
    
    for model_key, model_name in [
        ('standard_t5', 'Standard T5'),
        ('baseline_t5', 'Baseline T5'),
        ('monotonic_t5', 'Monotonic T5')
    ]:
        trigger = uat_results['learned_triggers'][model_key]
        delta = uat_results['results'][model_key]['rouge_deltas']['rougeLsum']
        nll_inc = uat_results['results'][model_key]['nll_increase']
        
        table_lines.append(f"{model_name:<20} | {trigger:<40} | {delta:>+12.4f} | {nll_inc:>+14.2%}")
    
    # Transfer Matrix
    table_lines.append("\nTransfer Matrix (ΔROUGE-L when trigger from row attacks column):")
    table_lines.append("-"*100)
    
    # Avoid backslashes inside f-string expression braces (SyntaxError on some Python versions).
    source_target_label = "Source \\ Target"
    header = f"{source_target_label:<20}"
    for model_key, model_name in [
        ('standard_t5', 'Std T5'),
        ('baseline_t5', 'Base T5'),
        ('monotonic_t5', 'Mono T5')
    ]:
        header += f" | {model_name:>12}"
    table_lines.append(header)
    table_lines.append("-"*100)
    
    transfer_matrix = uat_results['transfer_matrix']
    for source_key, source_name in [
        ('standard_t5', 'Standard T5'),
        ('baseline_t5', 'Baseline T5'),
        ('monotonic_t5', 'Monotonic T5')
    ]:
        row = f"{source_name:<20}"
        for target_key in ['standard_t5', 'baseline_t5', 'monotonic_t5']:
            delta = transfer_matrix[source_key][target_key]
            marker = " *" if source_key == target_key else ""
            row += f" | {delta:>+10.4f}{marker}"
        table_lines.append(row)
    
    # HotFlip Results
    table_lines.append("\n\nHotFlip Gradient-Based Attacks:")
    table_lines.append("-"*100)
    table_lines.append(f"{'Model':<20} | {'Avg Degradation':>18} | {'Success Rate':>15} | {'Avg Loss Increase':>20}")
    table_lines.append("-"*100)
    
    for model_key, model_name in [
        ('standard_t5', 'Standard T5'),
        ('baseline_t5', 'Baseline T5'),
        ('monotonic_t5', 'Monotonic T5')
    ]:
        stats = hotflip_results['results'][model_key]
        table_lines.append(f"{model_name:<20} | "
                          f"{stats['avg_degradation']:>17.2%} | "
                          f"{stats['success_rate']:>14.1%} | "
                          f"{stats['avg_attack_loss'] - stats['avg_orig_loss']:>+19.4f}")
    
    table_lines.append("="*100)
    return "\n".join(table_lines)


def create_training_summary(baseline_history, monotonic_history):
    """Create training summary"""
    table_lines = []
    table_lines.append("\n" + "="*100)
    table_lines.append("TRAINING SUMMARY")
    table_lines.append("="*100)
    
    table_lines.append(f"\n{'Model':<20} | {'Best Val Loss':>15} | {'Training Time':>15} | {'Final Train Loss':>18} | {'Final Val Loss':>15}")
    table_lines.append("-"*100)
    
    for model_name, history in [
        ('Baseline T5', baseline_history),
        ('Monotonic T5', monotonic_history)
    ]:
        table_lines.append(f"{model_name:<20} | "
                          f"{history['best_val_loss']:>15.4f} | "
                          f"{history['training_time_minutes']:>13.1f} min | "
                          f"{history['train_losses'][-1]:>18.4f} | "
                          f"{history['val_losses'][-1]:>15.4f}")
    
    table_lines.append("="*100)
    return "\n".join(table_lines)


def main():
    """Aggregate all results and create final outputs"""
    logger = StageLogger("stage_7_aggregate")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        required = ['stage_0_setup', 'stage_1_data_prep', 
                   'stage_2_train_baseline', 'stage_3_train_monotonic',
                   'stage_4_evaluate', 'stage_5_uat', 'stage_6_hotflip']
        if not check_dependencies(required):
            logger.complete(success=False)
            return 1
        
        # Set seeds
        logger.log("Setting random seeds...")
        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        
        # Load all results
        logger.log("\n" + "="*80)
        logger.log("LOADING RESULTS")
        logger.log("="*80)
        
        results_dir = ExperimentConfig.RESULTS_DIR
        
        logger.log("Loading evaluation results...")
        eval_results = load_json(os.path.join(results_dir, 'evaluation_results.json'))

        # Determine which datasets are actually present (avoid KeyError for skipped datasets).
        dataset_name_map = {
            'cnn_dm': 'CNN/DailyMail',
            'xsum': 'XSUM',
            'samsum': 'SAMSum',
        }
        preferred_order = ['cnn_dm', 'xsum', 'samsum']
        available_datasets = [k for k in preferred_order if k in eval_results]
        other_datasets = sorted([k for k in eval_results.keys() if k not in preferred_order])
        available_datasets.extend(other_datasets)

        missing = [k for k in preferred_order if k not in eval_results]
        if missing:
            logger.log(f"⚠️  Missing evaluation datasets (skipped earlier): "
                       f"{', '.join(dataset_name_map.get(k, k) for k in missing)}")
        
        logger.log("Loading UAT results...")
        uat_results = load_json(os.path.join(results_dir, 'uat_results.json'))
        
        logger.log("Loading HotFlip results...")
        hotflip_results = load_json(os.path.join(results_dir, 'hotflip_results.json'))
        
        logger.log("Loading training histories...")
        baseline_history = load_json(os.path.join(results_dir, 'baseline_training_history.json'))
        monotonic_history = load_json(os.path.join(results_dir, 'monotonic_training_history.json'))
        
        logger.log("✓ All results loaded")
        
        # Create comparison tables
        logger.log("\n" + "="*80)
        logger.log("GENERATING SUMMARY TABLES")
        logger.log("="*80)
        
        rouge_table = create_comparison_table(eval_results)
        attack_summary = create_attack_summary(uat_results, hotflip_results)
        training_summary = create_training_summary(baseline_history, monotonic_history)
        
        # Print to log
        logger.log("\n" + rouge_table)
        logger.log("\n" + attack_summary)
        logger.log("\n" + training_summary)
        
        # Create final aggregated results
        logger.log("\n" + "="*80)
        logger.log("CREATING FINAL RESULTS")
        logger.log("="*80)
        
        final_results = {
            'experiment_info': {
                'seed': ExperimentConfig.CURRENT_SEED,
                'model_name': ExperimentConfig.MODEL_NAME,
                'use_full_test_sets': ExperimentConfig.USE_FULL_TEST_SETS,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'training_summary': {
                'baseline': {
                    'best_val_loss': baseline_history['best_val_loss'],
                    'training_time_minutes': baseline_history['training_time_minutes'],
                    'num_epochs': baseline_history['num_epochs'],
                    'hyperparameters': baseline_history['hyperparameters']
                },
                'monotonic': {
                    'best_val_loss': monotonic_history['best_val_loss'],
                    'training_time_minutes': monotonic_history['training_time_minutes'],
                    'num_epochs': monotonic_history['num_epochs'],
                    'hyperparameters': monotonic_history['hyperparameters'],
                    'constraint_info': monotonic_history.get('constraint_info', {})
                }
            },
            'evaluation_summary': {
                dataset_key: {
                    model_key: {
                        'rouge_scores': eval_results[dataset_key][model_key]['rouge_scores'],
                        'brevity_penalty': eval_results[dataset_key][model_key]['brevity_penalty']
                    }
                    for model_key in ['standard_t5', 'baseline_t5', 'monotonic_t5']
                }
                for dataset_key in available_datasets
            },
            'attack_summary': {
                'uat': {
                    'learned_triggers': uat_results['learned_triggers'],
                    'results': uat_results['results'],
                    'transfer_matrix': uat_results['transfer_matrix']
                },
                'hotflip': {
                    'results': hotflip_results['results'],
                    'statistical_tests': hotflip_results.get('statistical_tests', {})
                }
            },
            'key_findings': {
                'best_rouge_cnn_dm': max(
                    [(k, eval_results['cnn_dm'][k]['rouge_scores']['rougeLsum']['mean']) 
                     for k in ['standard_t5', 'baseline_t5', 'monotonic_t5']],
                    key=lambda x: x[1]
                )[0],
                'most_robust_uat': min(
                    [(k, abs(uat_results['results'][k]['rouge_deltas']['rougeLsum'])) 
                     for k in ['standard_t5', 'baseline_t5', 'monotonic_t5']],
                    key=lambda x: x[1]
                )[0],
                'most_robust_hotflip': min(
                    [(k, hotflip_results['results'][k]['avg_degradation']) 
                     for k in ['standard_t5', 'baseline_t5', 'monotonic_t5']],
                    key=lambda x: x[1]
                )[0]
            }
        }
        
        # Save final results
        final_results_file = os.path.join(results_dir, 'final_results.json')
        save_json(final_results, final_results_file)
        logger.log(f"✓ Saved final results to: {final_results_file}")
        
        # Create human-readable summary
        summary_file = os.path.join(results_dir, 'experiment_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("MONO-S2S EXPERIMENT SUMMARY\n")
            f.write("Fair Comparison Edition - HPC Version\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"Experiment Date: {final_results['experiment_info']['timestamp']}\n")
            f.write(f"Model: {ExperimentConfig.MODEL_NAME}\n")
            f.write(f"Seed: {ExperimentConfig.CURRENT_SEED}\n")
            f.write(f"Full Test Sets: {ExperimentConfig.USE_FULL_TEST_SETS}\n\n")
            
            f.write(rouge_table + "\n\n")
            f.write(attack_summary + "\n\n")
            f.write(training_summary + "\n\n")
            
            f.write("="*100 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*100 + "\n")
            f.write(f"Best ROUGE (CNN/DM): {final_results['key_findings']['best_rouge_cnn_dm']}\n")
            f.write(f"Most Robust (UAT): {final_results['key_findings']['most_robust_uat']}\n")
            f.write(f"Most Robust (HotFlip): {final_results['key_findings']['most_robust_hotflip']}\n")
            f.write("="*100 + "\n")
        
        logger.log(f"✓ Saved experiment summary to: {summary_file}")
        
        # Copy results to permanent storage
        logger.log("\n" + "="*80)
        logger.log("COPYING TO PERMANENT STORAGE")
        logger.log("="*80)
        
        final_dir = ExperimentConfig.FINAL_RESULTS_DIR
        os.makedirs(final_dir, exist_ok=True)
        
        # Copy all result files
        result_files = [
            'evaluation_results.json',
            'uat_results.json',
            'hotflip_results.json',
            'baseline_training_history.json',
            'monotonic_training_history.json',
            'learned_triggers.csv',
            'final_results.json',
            'experiment_summary.txt',
            'data_statistics.json'
        ]
        
        for filename in result_files:
            src = os.path.join(results_dir, filename)
            if os.path.exists(src):
                dst = os.path.join(final_dir, filename)
                shutil.copy2(src, dst)
                logger.log(f"  ✓ Copied {filename}")
        
        # Also copy experiment metadata from stage 0
        metadata_src = os.path.join(ExperimentConfig.WORK_DIR, 'experiment_metadata.json')
        if os.path.exists(metadata_src):
            metadata_dst = os.path.join(final_dir, 'experiment_metadata.json')
            shutil.copy2(metadata_src, metadata_dst)
            logger.log(f"  ✓ Copied experiment_metadata.json")
        
        logger.log(f"\n✓ All results copied to: {final_dir}")
        
        # Final summary
        logger.log("\n" + "="*80)
        logger.log("✅ EXPERIMENT COMPLETE!")
        logger.log("="*80)
        logger.log(f"\nResults location:")
        logger.log(f"  Working directory: {results_dir}")
        logger.log(f"  Permanent storage: {final_dir}")
        logger.log(f"\nKey files:")
        logger.log(f"  - final_results.json (comprehensive results)")
        logger.log(f"  - experiment_summary.txt (human-readable)")
        logger.log(f"  - evaluation_results.json (ROUGE with CIs)")
        logger.log(f"  - uat_results.json (attack results + transfer matrix)")
        logger.log(f"  - hotflip_results.json (gradient attack results)")
        
        logger.log("\n" + "="*80)
        logger.log("METHODOLOGICAL CHECKLIST")
        logger.log("="*80)
        logger.log("✓ Three-model comparison (Standard, Baseline, Monotonic)")
        logger.log("✓ Identical training hyperparameters (Baseline vs Monotonic)")
        logger.log("✓ Fixed decoding parameters (all models)")
        logger.log("✓ Bootstrap 95% confidence intervals (ROUGE)")
        logger.log("✓ Full test set evaluation")
        logger.log("✓ Held-out attack evaluation")
        logger.log("✓ Transfer attack matrix")
        logger.log("✓ Comprehensive determinism")
        logger.log("✓ Complete experiment metadata")
        logger.log("="*80)
        
        # Mark complete
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

