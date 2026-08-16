#!/usr/bin/env python3
"""Keep best_model.pt plus the newest N epoch checkpoints. See hpc_version copy."""

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
