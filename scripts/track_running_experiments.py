#!/usr/bin/env python3
"""
Track Currently Running Experiments

Monitors running SLURM jobs and creates a tracking log.
Run periodically to maintain a record of all experimental runs.

Usage:
    python scripts/track_running_experiments.py
    python scripts/track_running_experiments.py --update  # Update existing log
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def get_running_jobs(user=None):
    """Get list of running jobs from squeue"""
    
    if user is None:
        user = os.environ.get('USER', 'unknown')
    
    try:
        # Run squeue command
        cmd = ['squeue', '-u', user, '-o', '%i|%j|%t|%M|%N|%R']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Warning: squeue command failed")
            return []
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return []
        
        jobs = []
        for line in lines[1:]:  # Skip header
            parts = line.split('|')
            if len(parts) >= 6:
                jobs.append({
                    'job_id': parts[0].strip(),
                    'name': parts[1].strip(),
                    'state': parts[2].strip(),
                    'time': parts[3].strip(),
                    'node': parts[4].strip(),
                    'reason': parts[5].strip(),
                })
        
        return jobs
        
    except Exception as e:
        print(f"Warning: Could not get running jobs: {e}")
        return []


def update_experiment_tracking_log(jobs, log_file="experiment_tracking_log.json"):
    """Update experiment tracking log with current jobs"""
    
    # Load existing log
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {
            "created": datetime.now().isoformat(),
            "experiments": {},
        }
    
    log["last_updated"] = datetime.now().isoformat()
    
    # Update with current jobs
    for job in jobs:
        job_id = job['job_id']
        
        if job_id not in log["experiments"]:
            # New job
            log["experiments"][job_id] = {
                "job_id": job_id,
                "name": job['name'],
                "first_seen": datetime.now().isoformat(),
                "status_history": [],
            }
        
        # Add status update
        log["experiments"][job_id]["status_history"].append({
            "timestamp": datetime.now().isoformat(),
            "state": job['state'],
            "time": job['time'],
            "node": job['node'],
            "reason": job['reason'],
        })
        
        log["experiments"][job_id]["last_state"] = job['state']
        log["experiments"][job_id]["last_seen"] = datetime.now().isoformat()
    
    # Save log
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    
    return log


def print_experiment_summary(log):
    """Print summary of tracked experiments"""
    
    print("\n" + "="*80)
    print("  EXPERIMENT TRACKING SUMMARY")
    print("="*80)
    print()
    
    # Count by status
    running = sum(1 for exp in log["experiments"].values() if exp.get("last_state") == "R")
    pending = sum(1 for exp in log["experiments"].values() if exp.get("last_state") == "PD")
    total = len(log["experiments"])
    
    print(f"Total tracked experiments: {total}")
    print(f"  Currently running: {running}")
    print(f"  Currently pending: {pending}")
    print(f"  Completed/other: {total - running - pending}")
    print()
    
    # Recent experiments
    print("Recent experiments:")
    sorted_exps = sorted(
        log["experiments"].items(),
        key=lambda x: x[1].get("first_seen", ""),
        reverse=True
    )
    
    for job_id, exp in sorted_exps[:10]:
        name = exp.get("name", "unknown")
        state = exp.get("last_state", "unknown")
        first_seen = exp.get("first_seen", "unknown")[:19]
        
        print(f"  {job_id}: {name:20s} {state:5s} (first seen: {first_seen})")


def main():
    parser = argparse.ArgumentParser(description="Track running experiments")
    parser.add_argument('--update', action='store_true',
                       help="Update experiment tracking log")
    parser.add_argument('--log-file', type=str,
                       default="experiment_tracking_log.json",
                       help="Path to tracking log file")
    parser.add_argument('--user', type=str,
                       help="Username (defaults to $USER)")
    
    args = parser.parse_args()
    
    print("Tracking running experiments...")
    print()
    
    # Get running jobs
    jobs = get_running_jobs(args.user)
    
    if not jobs:
        print("No jobs currently running or squeue not available")
        print("(This is OK if running locally or jobs completed)")
        return 0
    
    print(f"Found {len(jobs)} jobs:")
    for job in jobs:
        print(f"  {job['job_id']}: {job['name']} ({job['state']}) - {job['time']}")
    print()
    
    # Update log
    if args.update or not os.path.exists(args.log_file):
        log = update_experiment_tracking_log(jobs, args.log_file)
        print(f"✓ Updated tracking log: {args.log_file}")
        
        # Print summary
        print_experiment_summary(log)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
