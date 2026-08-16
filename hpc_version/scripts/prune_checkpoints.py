#!/usr/bin/env python3
"""
Keep best_model.pt plus the newest N epoch checkpoints.

Usage (on CURC, after training or between resubmits):

    python prune_checkpoints.py --dir $SCRATCH/mono_s2s_work/checkpoints/seed_42/baseline_checkpoints
    python prune_checkpoints.py --dir $SCRATCH/foundation_llm_work_seed42/checkpoints/monotonic_checkpoints --keep 1

Run curc-quota first. Archive only small JSON results to /projects.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.common_utils import prune_epoch_checkpoints


def main():
    parser = argparse.ArgumentParser(description="Prune old epoch checkpoints")
    parser.add_argument("--dir", required=True, help="Checkpoint directory")
    parser.add_argument("--keep", type=int, default=1, help="Newest epoch files to keep")
    args = parser.parse_args()

    removed = prune_epoch_checkpoints(args.dir, keep_last_n=args.keep)
    print(f"Removed {len(removed)} checkpoint(s) from {args.dir}")
    for path in removed:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
