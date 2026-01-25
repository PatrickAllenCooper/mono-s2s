#!/usr/bin/env python3
"""
Aggregate results across multiple random seeds for statistical analysis.

This script:
1. Loads results from each seed's run
2. Computes mean and std across seeds for all metrics
3. Performs cross-seed statistical tests
4. Generates publication-ready summary tables
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.experiment_config import ExperimentConfig


def load_seed_results(seed: int, scratch_dir: str) -> Dict[str, Any]:
    """Load all results for a single seed."""
    # Results are stored in seed-specific subdirectories
    seed_dir = Path(scratch_dir) / "mono_s2s_results" / f"seed_{seed}"
    
    # Also check the default location (for seed 42 which may not have subdirectory)
    if not seed_dir.exists():
        seed_dir = Path(scratch_dir) / "mono_s2s_results"
    
    results = {
        'seed': seed,
        'evaluation': None,
        'uat': None,
        'hotflip': None,
    }
    
    # Load evaluation results
    eval_path = seed_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            results['evaluation'] = json.load(f)
    
    # Load UAT results
    uat_path = seed_dir / "uat_results.json"
    if uat_path.exists():
        with open(uat_path) as f:
            results['uat'] = json.load(f)
    
    # Load HotFlip results
    hotflip_path = seed_dir / "hotflip_results.json"
    if hotflip_path.exists():
        with open(hotflip_path) as f:
            results['hotflip'] = json.load(f)
    
    return results


def extract_metric(results_list: List[Dict], path: List[str]) -> List[float]:
    """Extract a specific metric from all seed results."""
    values = []
    for result in results_list:
        try:
            val = result
            for key in path:
                val = val[key]
            if isinstance(val, (int, float)):
                values.append(float(val))
        except (KeyError, TypeError):
            continue
    return values


def compute_cross_seed_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics across seeds."""
    if len(values) < 2:
        return {
            'mean': values[0] if values else np.nan,
            'std': 0.0,
            'n': len(values),
            'ci_lower': np.nan,
            'ci_upper': np.nan,
        }
    
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # Sample std
    n = len(arr)
    
    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, df=n-1)
    se = std / np.sqrt(n)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    
    return {
        'mean': mean,
        'std': std,
        'n': n,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }


def paired_t_test_across_seeds(values_a: List[float], values_b: List[float]) -> Dict[str, float]:
    """Perform paired t-test across seeds."""
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return {'t_stat': np.nan, 'p_value': np.nan, 'cohens_d': np.nan}
    
    arr_a = np.array(values_a)
    arr_b = np.array(values_b)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(arr_a, arr_b)
    
    # Cohen's d for paired samples
    diff = arr_a - arr_b
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
    }


def aggregate_results(seeds: List[int], scratch_dir: str) -> Dict[str, Any]:
    """Aggregate results across all seeds."""
    
    # Load all seed results
    all_results = []
    for seed in seeds:
        result = load_seed_results(seed, scratch_dir)
        if result['evaluation'] is not None:
            all_results.append(result)
    
    if not all_results:
        print("ERROR: No valid results found for any seed")
        return {}
    
    print(f"Loaded results for {len(all_results)} seeds")
    
    aggregated = {
        'num_seeds': len(all_results),
        'seeds': [r['seed'] for r in all_results],
        'clean_performance': {},
        'attack_results': {},
        'statistical_tests': {},
    }
    
    # Define metrics to aggregate
    models = ['standard', 'baseline', 'monotonic']
    rouge_metrics = ['rouge1', 'rouge2', 'rougeLsum']
    
    # Aggregate clean performance (ROUGE scores)
    for model in models:
        aggregated['clean_performance'][model] = {}
        for metric in rouge_metrics:
            # Extract metric from each seed
            values = []
            for result in all_results:
                try:
                    # Navigate to the metric value
                    eval_data = result['evaluation']
                    if eval_data and model in eval_data:
                        model_data = eval_data[model]
                        if 'cnn_dailymail' in model_data:
                            val = model_data['cnn_dailymail'].get(metric, {}).get('mean')
                            if val is not None:
                                values.append(val)
                except (KeyError, TypeError):
                    continue
            
            aggregated['clean_performance'][model][metric] = compute_cross_seed_stats(values)
    
    # Aggregate attack results
    for model in models:
        aggregated['attack_results'][model] = {}
        
        # HotFlip success rate
        hotflip_values = []
        for result in all_results:
            try:
                if result['hotflip'] and model in result['hotflip']:
                    val = result['hotflip'][model].get('success_rate')
                    if val is not None:
                        hotflip_values.append(val)
            except (KeyError, TypeError):
                continue
        aggregated['attack_results'][model]['hotflip_success_rate'] = compute_cross_seed_stats(hotflip_values)
        
        # UAT success rate
        uat_values = []
        for result in all_results:
            try:
                if result['uat'] and model in result['uat']:
                    val = result['uat'][model].get('success_rate')
                    if val is not None:
                        uat_values.append(val)
            except (KeyError, TypeError):
                continue
        aggregated['attack_results'][model]['uat_success_rate'] = compute_cross_seed_stats(uat_values)
    
    # Statistical tests: Baseline vs Monotonic across seeds
    print("\nPerforming cross-seed statistical tests...")
    
    # ROUGE-L comparison
    baseline_rougeL = []
    monotonic_rougeL = []
    for result in all_results:
        try:
            eval_data = result['evaluation']
            if eval_data:
                bl_val = eval_data['baseline']['cnn_dailymail']['rougeLsum']['mean']
                mono_val = eval_data['monotonic']['cnn_dailymail']['rougeLsum']['mean']
                baseline_rougeL.append(bl_val)
                monotonic_rougeL.append(mono_val)
        except (KeyError, TypeError):
            continue
    
    aggregated['statistical_tests']['rougeL_baseline_vs_monotonic'] = paired_t_test_across_seeds(
        baseline_rougeL, monotonic_rougeL
    )
    
    # HotFlip success rate comparison
    baseline_hotflip = []
    monotonic_hotflip = []
    for result in all_results:
        try:
            if result['hotflip']:
                bl_val = result['hotflip']['baseline']['success_rate']
                mono_val = result['hotflip']['monotonic']['success_rate']
                baseline_hotflip.append(bl_val)
                monotonic_hotflip.append(mono_val)
        except (KeyError, TypeError):
            continue
    
    aggregated['statistical_tests']['hotflip_baseline_vs_monotonic'] = paired_t_test_across_seeds(
        baseline_hotflip, monotonic_hotflip
    )
    
    return aggregated


def generate_summary_text(aggregated: Dict[str, Any]) -> str:
    """Generate human-readable summary."""
    lines = [
        "=" * 70,
        "MULTI-SEED AGGREGATED RESULTS",
        "=" * 70,
        f"Number of seeds: {aggregated.get('num_seeds', 0)}",
        f"Seeds: {aggregated.get('seeds', [])}",
        "",
        "CLEAN PERFORMANCE (ROUGE-L, mean +/- std across seeds)",
        "-" * 50,
    ]
    
    for model in ['standard', 'baseline', 'monotonic']:
        try:
            stats = aggregated['clean_performance'][model]['rougeLsum']
            lines.append(f"  {model:12s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
        except (KeyError, TypeError):
            lines.append(f"  {model:12s}: N/A")
    
    lines.extend([
        "",
        "ATTACK SUCCESS RATES (mean +/- std across seeds)",
        "-" * 50,
    ])
    
    for model in ['standard', 'baseline', 'monotonic']:
        try:
            hotflip = aggregated['attack_results'][model]['hotflip_success_rate']
            uat = aggregated['attack_results'][model]['uat_success_rate']
            lines.append(f"  {model:12s}:")
            lines.append(f"    HotFlip: {hotflip['mean']*100:.1f}% +/- {hotflip['std']*100:.1f}%")
            lines.append(f"    UAT:     {uat['mean']*100:.1f}% +/- {uat['std']*100:.1f}%")
        except (KeyError, TypeError):
            lines.append(f"  {model:12s}: N/A")
    
    lines.extend([
        "",
        "CROSS-SEED STATISTICAL TESTS (Baseline vs Monotonic)",
        "-" * 50,
    ])
    
    try:
        rougeL_test = aggregated['statistical_tests']['rougeL_baseline_vs_monotonic']
        lines.append(f"  ROUGE-L: t={rougeL_test['t_stat']:.3f}, p={rougeL_test['p_value']:.4f}, d={rougeL_test['cohens_d']:.3f}")
    except (KeyError, TypeError):
        lines.append("  ROUGE-L: N/A")
    
    try:
        hotflip_test = aggregated['statistical_tests']['hotflip_baseline_vs_monotonic']
        lines.append(f"  HotFlip: t={hotflip_test['t_stat']:.3f}, p={hotflip_test['p_value']:.4f}, d={hotflip_test['cohens_d']:.3f}")
    except (KeyError, TypeError):
        lines.append("  HotFlip: N/A")
    
    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 50,
    ])
    
    # Interpret results
    try:
        hotflip_test = aggregated['statistical_tests']['hotflip_baseline_vs_monotonic']
        if hotflip_test['p_value'] < 0.05:
            lines.append("  HotFlip: Significant difference (p < 0.05)")
            if hotflip_test['cohens_d'] > 0.8:
                lines.append("           Large effect size (d > 0.8)")
            elif hotflip_test['cohens_d'] > 0.5:
                lines.append("           Medium effect size (0.5 < d < 0.8)")
        else:
            lines.append("  HotFlip: No significant difference (p >= 0.05)")
    except (KeyError, TypeError):
        pass
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                        help='Seeds to aggregate')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for aggregated results')
    args = parser.parse_args()
    
    scratch_dir = os.environ.get('SCRATCH', f"/scratch/alpine/{os.environ.get('USER', 'user')}")
    
    print(f"Aggregating results for seeds: {args.seeds}")
    print(f"Looking in: {scratch_dir}")
    
    # Aggregate
    aggregated = aggregate_results(args.seeds, scratch_dir)
    
    if not aggregated:
        print("ERROR: No results to aggregate")
        sys.exit(1)
    
    # Save aggregated results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "multi_seed_results.json"
    with open(json_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Saved: {json_path}")
    
    # Save text summary
    summary = generate_summary_text(aggregated)
    summary_path = output_dir / "multi_seed_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Saved: {summary_path}")
    
    # Print summary
    print("\n" + summary)


if __name__ == "__main__":
    main()
