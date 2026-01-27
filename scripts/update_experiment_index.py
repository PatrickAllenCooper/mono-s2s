#!/usr/bin/env python3
"""
Update Experiment Index

Maintains a central index of all experimental runs.
Called automatically by commit_experiment_results.sh.

Usage:
    python scripts/update_experiment_index.py --experiment-dir experiment_results/t5_experiments/seed_42 \
                                               --seed 42 \
                                               --type t5_summarization
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def update_index(experiment_dir, seed, experiment_type):
    """Update the experiment index with new experiment"""
    
    index_file = Path("experiment_results/experiment_index.json")
    
    # Load existing index
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
    else:
        index = {
            "created": datetime.now().isoformat(),
            "experiments": [],
            "summary": {
                "total_experiments": 0,
                "by_type": {},
                "by_seed": {},
            }
        }
    
    index["last_updated"] = datetime.now().isoformat()
    
    # Load metadata from experiment
    metadata_file = Path(experiment_dir) / "metadata.json"
    if not metadata_file.exists():
        print(f"Warning: No metadata.json found in {experiment_dir}")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Create index entry
    entry = {
        "experiment_id": metadata.get("experiment_id"),
        "type": experiment_type,
        "seed": seed,
        "directory": str(experiment_dir),
        "date": metadata.get("timestamp_organized", datetime.now().isoformat()),
        "model": metadata.get("model", {}),
        "status": metadata.get("status", "unknown"),
        "used_in_paper": metadata.get("verification", {}).get("used_in_paper", False),
        "paper_tables": metadata.get("verification", {}).get("paper_tables", []),
    }
    
    # Check if experiment already indexed
    existing_idx = None
    for i, exp in enumerate(index["experiments"]):
        if exp.get("experiment_id") == entry["experiment_id"]:
            existing_idx = i
            break
    
    if existing_idx is not None:
        # Update existing
        index["experiments"][existing_idx] = entry
        print(f"✓ Updated existing experiment: {entry['experiment_id']}")
    else:
        # Add new
        index["experiments"].append(entry)
        print(f"✓ Added new experiment: {entry['experiment_id']}")
    
    # Update summary
    index["summary"]["total_experiments"] = len(index["experiments"])
    
    # Count by type
    index["summary"]["by_type"] = {}
    for exp in index["experiments"]:
        exp_type = exp.get("type", "unknown")
        index["summary"]["by_type"][exp_type] = index["summary"]["by_type"].get(exp_type, 0) + 1
    
    # Count by seed
    index["summary"]["by_seed"] = {}
    for exp in index["experiments"]:
        exp_seed = str(exp.get("seed", "unknown"))
        index["summary"]["by_seed"][exp_seed] = index["summary"]["by_seed"].get(exp_seed, 0) + 1
    
    # Save index
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✓ Index updated: {index_file}")
    
    # Print summary
    print(f"\nExperiment Index Summary:")
    print(f"  Total experiments: {index['summary']['total_experiments']}")
    print(f"  By type: {index['summary']['by_type']}")
    print(f"  By seed: {index['summary']['by_seed']}")


def main():
    parser = argparse.ArgumentParser(description="Update experiment index")
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help="Path to organized experiment directory")
    parser.add_argument('--seed', type=int, required=True,
                       help="Random seed used")
    parser.add_argument('--type', type=str, required=True,
                       choices=['t5_summarization', 'pythia_foundation'],
                       help="Type of experiment")
    
    args = parser.parse_args()
    
    update_index(args.experiment_dir, args.seed, args.type)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
