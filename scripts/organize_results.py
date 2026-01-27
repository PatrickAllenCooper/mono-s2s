#!/usr/bin/env python3
"""
Organize Experimental Results for Version Control

This script organizes experimental results into a structured format
suitable for version control and paper integration.

Usage:
    python scripts/organize_results.py --source $SCRATCH/mono_s2s_results \
                                       --dest experiment_results/t5_experiments/seed_42 \
                                       --seed 42 \
                                       --experiment-type t5_summarization

    python scripts/organize_results.py --auto  # Auto-detect from current results
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def create_metadata(experiment_type, seed, source_dir, job_ids=None):
    """Create experiment metadata"""
    
    metadata = {
        "experiment_id": f"{experiment_type}_seed{seed}_{datetime.now().strftime('%Y%m%d')}",
        "experiment_type": experiment_type,
        "timestamp_organized": datetime.now().isoformat(),
        "seed": seed,
        "source_directory": str(source_dir),
        "organized_by": "organize_results.py",
    }
    
    # Try to extract info from results
    if os.path.exists(os.path.join(source_dir, 'setup_complete.json')):
        with open(os.path.join(source_dir, 'setup_complete.json')) as f:
            setup = json.load(f)
            metadata["model"] = {
                "name": setup.get("model_name", "unknown"),
                "parameters": setup.get("num_parameters", 0),
            }
            metadata["hardware"] = {
                "gpu": setup.get("gpu_name", "unknown"),
                "cuda_version": setup.get("cuda_version", "unknown"),
            }
    
    if job_ids:
        metadata["slurm_job_ids"] = job_ids
    
    # Extract hyperparameters from training history
    if os.path.exists(os.path.join(source_dir, 'baseline_training_history.json')):
        with open(os.path.join(source_dir, 'baseline_training_history.json')) as f:
            baseline = json.load(f)
            metadata["hyperparameters"] = baseline.get("hyperparameters", {})
    
    metadata["status"] = "organized"
    
    return metadata


def organize_experiment_results(source_dir, dest_dir, seed, experiment_type, job_ids=None):
    """
    Organize experimental results from scratch/source into version-controlled structure
    
    Args:
        source_dir: Source directory (e.g., $SCRATCH/mono_s2s_results)
        dest_dir: Destination directory (e.g., experiment_results/t5_experiments/seed_42)
        seed: Random seed used
        experiment_type: Type of experiment (t5_summarization, foundation_llm, etc.)
        job_ids: Optional list of SLURM job IDs
    """
    
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    print(f"Organizing experiment results...")
    print(f"  Source: {source_dir}")
    print(f"  Dest: {dest_dir}")
    print(f"  Seed: {seed}")
    print(f"  Type: {experiment_type}")
    print()
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to copy (small result files only, not checkpoints)
    result_files = [
        'setup_complete.json',
        'data_statistics.json',
        'baseline_training_history.json',
        'monotonic_training_history.json',
        'evaluation_results.json',
        'uat_results.json',
        'hotflip_results.json',
        'final_results.json',
        'experiment_summary.txt',
        'learned_triggers.csv',
    ]
    
    copied_files = []
    
    for filename in result_files:
        source_file = source_dir / filename
        if source_file.exists():
            dest_file = dest_dir / filename
            shutil.copy2(source_file, dest_file)
            print(f"  ✓ Copied: {filename}")
            copied_files.append(filename)
        else:
            print(f"  ⚠️  Missing: {filename}")
    
    # Create metadata
    print("\nCreating metadata...")
    metadata = create_metadata(experiment_type, seed, source_dir, job_ids)
    metadata["copied_files"] = copied_files
    
    with open(dest_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Created: metadata.json")
    
    # Create experiment log
    print("\nCreating experiment log...")
    log_lines = [
        f"Experiment: {metadata['experiment_id']}",
        f"Type: {experiment_type}",
        f"Seed: {seed}",
        f"Organized: {metadata['timestamp_organized']}",
        f"",
        f"Files copied: {len(copied_files)}",
        f"Source: {source_dir}",
        f"",
        f"Results:",
    ]
    
    # Add key results if available
    if 'final_results.json' in copied_files:
        with open(dest_dir / 'final_results.json') as f:
            final = json.load(f)
            if 'key_findings' in final:
                log_lines.append(f"  - Best ROUGE: {final['key_findings'].get('best_rouge_cnn_dm', 'N/A')}")
                log_lines.append(f"  - Most robust (HotFlip): {final['key_findings'].get('most_robust_hotflip', 'N/A')}")
    
    with open(dest_dir / "experiment_log.txt", 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"  ✓ Created: experiment_log.txt")
    
    print(f"\n✓ Experiment results organized successfully!")
    print(f"  Location: {dest_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review results in {dest_dir}")
    print(f"  2. Add to git: git add {dest_dir}")
    print(f"  3. Commit: git commit -m 'Add experiment results for seed {seed}'")
    
    return dest_dir


def create_experiment_index(results_dir):
    """Create master index of all experiments"""
    
    results_dir = Path(results_dir)
    
    index = {
        "created": datetime.now().isoformat(),
        "experiments": {},
        "total_experiments": 0,
    }
    
    # Scan for all experiments
    for experiment_type_dir in results_dir.iterdir():
        if not experiment_type_dir.is_dir():
            continue
        
        experiment_type = experiment_type_dir.name
        index["experiments"][experiment_type] = {}
        
        for seed_dir in experiment_type_dir.iterdir():
            if not seed_dir.is_dir() or seed_dir.name == "aggregated":
                continue
            
            seed = seed_dir.name
            metadata_file = seed_dir / "metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                index["experiments"][experiment_type][seed] = {
                    "experiment_id": metadata.get("experiment_id"),
                    "status": metadata.get("status"),
                    "timestamp": metadata.get("timestamp_organized"),
                    "path": str(seed_dir.relative_to(results_dir)),
                }
                index["total_experiments"] += 1
    
    # Save index
    with open(results_dir / "experiment_index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✓ Created experiment index: {results_dir / 'experiment_index.json'}")
    print(f"  Total experiments: {index['total_experiments']}")
    
    return index


def main():
    parser = argparse.ArgumentParser(description="Organize experimental results")
    parser.add_argument('--source', type=str, help="Source directory with results")
    parser.add_argument('--dest', type=str, help="Destination directory")
    parser.add_argument('--seed', type=int, help="Random seed used")
    parser.add_argument('--experiment-type', type=str, 
                       choices=['t5_summarization', 'foundation_llm', 'other'],
                       help="Type of experiment")
    parser.add_argument('--job-ids', type=str, help="Comma-separated SLURM job IDs")
    parser.add_argument('--auto', action='store_true',
                       help="Auto-detect from mono_s2s_results/")
    parser.add_argument('--create-index', action='store_true',
                       help="Create experiment index")
    
    args = parser.parse_args()
    
    if args.auto:
        # Auto-detect from current results
        print("Auto-detecting from mono_s2s_results/...")
        
        source_dir = "mono_s2s_results"
        if not os.path.exists(source_dir):
            print(f"ERROR: {source_dir} not found")
            return 1
        
        # Try to detect seed from final_results.json
        if os.path.exists(f"{source_dir}/final_results.json"):
            with open(f"{source_dir}/final_results.json") as f:
                final = json.load(f)
                seed = final.get('experiment_info', {}).get('seed', 42)
        else:
            print("⚠️  Could not detect seed, using default: 42")
            seed = 42
        
        dest_dir = f"experiment_results/t5_experiments/seed_{seed}"
        experiment_type = "t5_summarization"
        job_ids = None
        
        print(f"  Detected seed: {seed}")
        print(f"  Destination: {dest_dir}")
        print()
        
        organize_experiment_results(source_dir, dest_dir, seed, experiment_type, job_ids)
        
        # Create index
        create_experiment_index("experiment_results")
        
    elif args.create_index:
        # Just create/update index
        create_experiment_index(args.dest or "experiment_results")
        
    else:
        # Manual specification
        if not all([args.source, args.dest, args.seed, args.experiment_type]):
            print("ERROR: Must specify --source, --dest, --seed, --experiment-type")
            print("Or use --auto for automatic organization")
            return 1
        
        job_ids = [int(x.strip()) for x in args.job_ids.split(',')] if args.job_ids else None
        
        organize_experiment_results(
            args.source,
            args.dest,
            args.seed,
            args.experiment_type,
            job_ids
        )
        
        # Update index
        if os.path.exists("experiment_results"):
            create_experiment_index("experiment_results")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
