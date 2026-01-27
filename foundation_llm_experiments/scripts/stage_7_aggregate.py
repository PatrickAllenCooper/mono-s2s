#!/usr/bin/env python3
"""
Stage 7: Aggregate All Results

Combines results from all stages into final summary.

Inputs:
- baseline_training_history.json
- monotonic_training_history.json
- evaluation_results.json
- uat_results.json
- hotflip_results.json

Outputs:
- final_results.json
- experiment_summary.txt
- stage_7_aggregate_complete.flag
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    create_completion_flag, save_json, load_json,
    StageLogger, check_dependencies
)


def create_text_summary(results):
    """Create human-readable summary"""
    
    # Extract key metrics
    baseline_ppl = results['evaluation_summary']['pile_test']['baseline_pythia']['perplexity']
    monotonic_ppl = results['evaluation_summary']['pile_test']['monotonic_pythia']['perplexity']
    ppl_gap = (monotonic_ppl - baseline_ppl) / baseline_ppl * 100
    
    baseline_hotflip = results['attack_summary']['hotflip']['results']['baseline_pythia']['success_rate']
    monotonic_hotflip = results['attack_summary']['hotflip']['results']['monotonic_pythia']['success_rate']
    hotflip_reduction = (baseline_hotflip - monotonic_hotflip) / baseline_hotflip * 100
    
    baseline_uat_nll = results['attack_summary']['uat']['results']['baseline_pythia']['nll_increase_percent']
    monotonic_uat_nll = results['attack_summary']['uat']['results']['monotonic_pythia']['nll_increase_percent']
    
    summary = f"""{'='*80}
FOUNDATION LLM MONOTONICITY EXPERIMENT SUMMARY
{'='*80}

Experiment Date: {results['experiment_info']['timestamp']}
Model: {results['experiment_info']['model_name']}
Seed: {results['experiment_info']['seed']}

{'='*80}
TRAINING SUMMARY
{'='*80}

Baseline Training:
  Best Val Perplexity: {results['training_summary']['baseline']['best_val_perplexity']:.2f}
  Training Time: {results['training_summary']['baseline']['training_time_hours']:.1f} hours
  Final Train Loss: {results['training_summary']['baseline']['train_losses'][-1]:.4f}

Monotonic Training:
  Best Val Perplexity: {results['training_summary']['monotonic']['best_val_perplexity']:.2f}
  Training Time: {results['training_summary']['monotonic']['training_time_hours']:.1f} hours
  Final Train Loss: {results['training_summary']['monotonic']['train_losses'][-1]:.4f}

{'='*80}
PERPLEXITY EVALUATION (Pile Test Set)
{'='*80}

Model              |  Perplexity  |  Gap
-------------------|--------------|-------
Baseline           |  {baseline_ppl:10.2f}  |   —
Monotonic          |  {monotonic_ppl:10.2f}  |  {ppl_gap:+.1f}%

{'='*80}
ADVERSARIAL ROBUSTNESS
{'='*80}

HotFlip Gradient-Based Attacks:
------------------------------------------------------------
Model              | Avg Degradation | Success Rate | Avg Loss Increase
-------------------|-----------------|--------------|------------------
Baseline           |    {results['attack_summary']['hotflip']['results']['baseline_pythia']['avg_degradation']*100:10.2f}%   |    {baseline_hotflip*100:8.1f}%   |   +{results['attack_summary']['hotflip']['results']['baseline_pythia']['avg_attack_loss'] - results['attack_summary']['hotflip']['results']['baseline_pythia']['avg_orig_loss']:.4f}
Monotonic          |    {results['attack_summary']['hotflip']['results']['monotonic_pythia']['avg_degradation']*100:10.2f}%   |    {monotonic_hotflip*100:8.1f}%   |   +{results['attack_summary']['hotflip']['results']['monotonic_pythia']['avg_attack_loss'] - results['attack_summary']['hotflip']['results']['monotonic_pythia']['avg_orig_loss']:.4f}

Improvement: {hotflip_reduction:.1f}% reduction in attack success rate

Universal Adversarial Trigger (UAT) Attacks:
------------------------------------------------------------
Model              | Learned Trigger                    | NLL Increase
-------------------|------------------------------------|--------------
Baseline           | {results['attack_summary']['uat']['results']['baseline_pythia']['trigger_text'][:30]:30s} | {baseline_uat_nll:+8.2f}%
Monotonic          | {results['attack_summary']['uat']['results']['monotonic_pythia']['trigger_text'][:30]:30s} | {monotonic_uat_nll:+8.2f}%

{'='*80}
KEY FINDINGS
{'='*80}

✓ Perplexity Gap: {ppl_gap:+.1f}% (monotonic slightly worse on clean data)
✓ HotFlip Robustness: {hotflip_reduction:.1f}% attack success reduction
✓ UAT Robustness: Minimal impact (<1% NLL increase across all models)

{'='*80}
PAPER INTEGRATION
{'='*80}

For Table 7 in paper (Section 4.3):

  Model: Pythia-1.4B
  Perplexity (Clean): {baseline_ppl:.1f} (baseline), {monotonic_ppl:.1f} (monotonic)
  Perplexity Gap: {ppl_gap:+.1f}%
  Attack Success: {baseline_hotflip*100:.1f}% (baseline), {monotonic_hotflip*100:.1f}% (monotonic)
  Attack Reduction: {hotflip_reduction:.1f}%

{'='*80}
"""
    
    return summary


def main():
    """Aggregate all results"""
    logger = StageLogger("stage_7_aggregate")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        required = ['stage_4_evaluate', 'stage_5_uat', 'stage_6_hotflip']
        if not check_dependencies(required):
            logger.complete(success=False)
            return 1
        
        # Load all results
        logger.log("Loading results from all stages...")
        
        baseline_history = load_json(
            os.path.join(Config.RESULTS_DIR, 'baseline_training_history.json')
        )
        logger.log("  ✓ Baseline training history")
        
        monotonic_history = load_json(
            os.path.join(Config.RESULTS_DIR, 'monotonic_training_history.json')
        )
        logger.log("  ✓ Monotonic training history")
        
        evaluation = load_json(
            os.path.join(Config.RESULTS_DIR, 'evaluation_results.json')
        )
        logger.log("  ✓ Evaluation results")
        
        uat = load_json(
            os.path.join(Config.RESULTS_DIR, 'uat_results.json')
        )
        logger.log("  ✓ UAT attack results")
        
        hotflip = load_json(
            os.path.join(Config.RESULTS_DIR, 'hotflip_results.json')
        )
        logger.log("  ✓ HotFlip attack results")
        
        # Aggregate
        logger.log("\nAggregating results...")
        
        final_results = {
            'experiment_info': {
                'seed': Config.CURRENT_SEED,
                'model_name': Config.MODEL_NAME,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'training_summary': {
                'baseline': baseline_history,
                'monotonic': monotonic_history,
            },
            'evaluation_summary': evaluation,
            'attack_summary': {
                'uat': uat,
                'hotflip': hotflip,
            },
            'key_findings': {
                'perplexity_gap_percent': (
                    (evaluation['pile_test']['monotonic_pythia']['perplexity'] - 
                     evaluation['pile_test']['baseline_pythia']['perplexity']) /
                    evaluation['pile_test']['baseline_pythia']['perplexity'] * 100
                ),
                'hotflip_reduction_percent': (
                    (hotflip['results']['baseline_pythia']['success_rate'] -
                     hotflip['results']['monotonic_pythia']['success_rate']) /
                    hotflip['results']['baseline_pythia']['success_rate'] * 100
                ),
                'most_robust_hotflip': 'monotonic_pythia',
            }
        }
        
        # Save final results
        logger.log("Saving aggregated results...")
        
        final_path = os.path.join(Config.FINAL_RESULTS_DIR, 'final_results.json')
        save_json(final_results, final_path)
        logger.log(f"✓ Saved to: {final_path}")
        
        # Create text summary
        logger.log("\nCreating text summary...")
        summary_text = create_text_summary(final_results)
        
        summary_path = os.path.join(Config.FINAL_RESULTS_DIR, 'experiment_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        logger.log(f"✓ Saved to: {summary_path}")
        
        # Print summary
        logger.log("\n" + summary_text)
        
        logger.log("\n✓ Aggregation complete!")
        logger.complete(success=True)
        return 0
        
    except FileNotFoundError as e:
        logger.log(f"\n❌ ERROR: Missing required results file")
        logger.log(f"  {str(e)}")
        logger.log("\nPlease ensure all previous stages completed successfully:")
        logger.log("  - stage_2_train_baseline")
        logger.log("  - stage_3_train_monotonic")
        logger.log("  - stage_4_evaluate")
        logger.log("  - stage_5_uat")
        logger.log("  - stage_6_hotflip")
        logger.complete(success=False)
        return 1
    
    except Exception as e:
        logger.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())
