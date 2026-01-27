#!/usr/bin/env python3
"""
Link Experimental Results to Paper Tables

Creates a provenance file linking experimental results to specific paper tables.
This ensures we can always trace which experiments produced which paper claims.

Usage:
    python scripts/link_results_to_paper.py --experiment seed_42 \
                                            --table table_1 \
                                            --values baseline_initial_loss=2.90,monotonic_initial_loss=4.97

    python scripts/link_results_to_paper.py --auto  # Auto-link from current results
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def create_or_update_provenance(table_id, experiment_id, source_files, extracted_values):
    """Create or update provenance tracking"""
    
    provenance_file = Path("paper_evidence/provenance.json")
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing provenance
    if provenance_file.exists():
        with open(provenance_file) as f:
            provenance = json.load(f)
    else:
        provenance = {
            "paper": "documentation/monotone_llms_paper.tex",
            "created": datetime.now().isoformat(),
            "tables": {}
        }
    
    provenance["last_updated"] = datetime.now().isoformat()
    
    # Add/update table provenance
    provenance["tables"][table_id] = {
        "source_experiment": experiment_id,
        "source_files": source_files,
        "extracted_values": extracted_values,
        "extraction_date": datetime.now().isoformat(),
        "verified": False,  # Manual verification required
    }
    
    # Save
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"✓ Provenance updated: {table_id} → {experiment_id}")
    print(f"  Saved to: {provenance_file}")
    
    return provenance


def auto_link_current_results():
    """Automatically link current results to paper tables"""
    
    print("Auto-linking current experimental results to paper...")
    
    # Find most recent T5 experiment
    t5_experiments = list(Path("experiment_results/t5_experiments").glob("seed_*/final_results.json"))
    
    if not t5_experiments:
        print("No T5 experiments found")
        return
    
    # Use most recent
    most_recent = max(t5_experiments, key=lambda p: p.stat().st_mtime)
    experiment_dir = most_recent.parent
    seed = experiment_dir.name.replace("seed_", "")
    
    print(f"Using experiment: {experiment_dir}")
    print(f"Seed: {seed}")
    
    # Load results
    with open(most_recent) as f:
        results = json.load(f)
    
    # Extract values for each table
    tables_to_link = {
        "table_1_training_dynamics": {
            "baseline_initial_loss": results['training_summary']['baseline']['train_losses'][0],
            "baseline_final_loss": results['training_summary']['baseline']['train_losses'][-1],
            "monotonic_initial_loss": results['training_summary']['monotonic']['train_losses'][0],
            "monotonic_final_loss": results['training_summary']['monotonic']['train_losses'][-1],
        },
        "table_2_rouge_scores": {
            "baseline_rouge_l": results['evaluation_summary']['cnn_dm']['baseline_t5']['rouge_scores']['rougeLsum']['mean'],
            "monotonic_rouge_l": results['evaluation_summary']['cnn_dm']['monotonic_t5']['rouge_scores']['rougeLsum']['mean'],
        },
        "table_5_hotflip": {
            "baseline_success_rate": results['attack_summary']['hotflip']['results']['baseline_t5']['success_rate'],
            "monotonic_success_rate": results['attack_summary']['hotflip']['results']['monotonic_t5']['success_rate'],
        },
    }
    
    # Create provenance for each table
    experiment_id = f"t5_small_seed{seed}"
    
    for table_id, values in tables_to_link.items():
        source_files = [str(most_recent)]
        create_or_update_provenance(table_id, experiment_id, source_files, values)
    
    print(f"\n✓ Auto-linked {len(tables_to_link)} tables to experiment")


def main():
    parser = argparse.ArgumentParser(description="Link results to paper tables")
    parser.add_argument('--experiment', type=str,
                       help="Experiment ID or directory")
    parser.add_argument('--table', type=str,
                       help="Table ID (e.g., table_1, table_2)")
    parser.add_argument('--values', type=str,
                       help="Comma-separated key=value pairs")
    parser.add_argument('--auto', action='store_true',
                       help="Auto-link from current results")
    
    args = parser.parse_args()
    
    if args.auto:
        auto_link_current_results()
    elif args.experiment and args.table and args.values:
        # Manual linking
        values = {}
        for pair in args.values.split(','):
            k, v = pair.split('=')
            try:
                values[k.strip()] = float(v.strip())
            except:
                values[k.strip()] = v.strip()
        
        source_files = [f"experiment_results/*/seed_*/{args.experiment}"]
        create_or_update_provenance(args.table, args.experiment, source_files, values)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
