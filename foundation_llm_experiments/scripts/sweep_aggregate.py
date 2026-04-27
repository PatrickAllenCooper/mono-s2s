#!/usr/bin/env python3
"""
Sweep Aggregator

Reads per-cell evaluation_results.json and hotflip_results.json from
/persist/sweep_results_seed<S>_<V>/ directories, prints a comparison
table, selects the winning variant per the decision rule, and writes
paper_evidence/sweep_summary.json.

Decision rule
-------------
  Winner = variant with highest (baseline_hotflip_SR - monotonic_hotflip_SR)
           subject to monotonic_ppl <= ppl_ceiling * baseline_ppl.

  If no variant satisfies the ceiling at the specified multiplier the caller
  can raise it and retry.

Usage
-----
  # Print table for Phase A on seed 42
  python sweep_aggregate.py \\
      --seeds 42 \\
      --variants mlp_both mlp_in_attn_out \\
      --results-root /persist/sweep_results

  # Auto-select winner and write it to a file
  python sweep_aggregate.py \\
      --seeds 42 \\
      --variants mlp_both mlp_in_attn_out \\
      --ppl-ceiling 2.0 --pick-winner \\
      --winner-file /persist/sweep_winner.txt \\
      --results-root /persist/sweep_results

  # Full 3-seed aggregate after Phase B
  python sweep_aggregate.py \\
      --seeds 42 1337 2024 \\
      --variants mlp_in_attn_out \\
      --results-root /persist/sweep_results \\
      --output paper_evidence/sweep_summary.json
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_cell(results_root, seed, variant):
    """Load evaluation and hotflip data for one (seed, variant) cell."""
    cell_tag = f"seed{seed}_{variant}"
    cell_dir = f"{results_root}_{cell_tag}"

    eval_path = os.path.join(cell_dir, "evaluation_results.json")
    hf_path   = os.path.join(cell_dir, "hotflip_results.json")

    eval_data = _load_json(eval_path)
    hf_data   = _load_json(hf_path)

    row = {"seed": seed, "variant": variant, "cell_dir": cell_dir}

    if eval_data:
        pt = eval_data.get("pile_test", {})
        bp = pt.get("baseline_pythia", {})
        mp = pt.get("monotonic_pythia", {})
        row["ppl_baseline"]  = bp.get("perplexity")
        row["ppl_monotonic"] = mp.get("perplexity")
        if row["ppl_baseline"] and row["ppl_monotonic"]:
            row["ppl_ratio"] = row["ppl_monotonic"] / row["ppl_baseline"]
        else:
            row["ppl_ratio"] = None
    else:
        row["ppl_baseline"] = row["ppl_monotonic"] = row["ppl_ratio"] = None

    if hf_data:
        res = hf_data.get("results", {})
        bp_hf = res.get("baseline_pythia", {})
        mp_hf = res.get("monotonic_pythia", {})
        row["hf_sr_baseline"]   = bp_hf.get("success_rate")
        row["hf_sr_monotonic"]  = mp_hf.get("success_rate")
        row["hf_deg_baseline"]  = bp_hf.get("avg_degradation")
        row["hf_deg_monotonic"] = mp_hf.get("avg_degradation")
        if row["hf_sr_baseline"] is not None and row["hf_sr_monotonic"] is not None:
            row["hf_sr_drop"] = row["hf_sr_baseline"] - row["hf_sr_monotonic"]
        else:
            row["hf_sr_drop"] = None
    else:
        row["hf_sr_baseline"] = row["hf_sr_monotonic"] = None
        row["hf_deg_baseline"] = row["hf_deg_monotonic"] = None
        row["hf_sr_drop"] = None

    return row


def print_table(rows):
    cols = [
        ("seed",          "Seed",       6),
        ("variant",       "Variant",    18),
        ("ppl_baseline",  "PPL base",   9),
        ("ppl_monotonic", "PPL mono",   9),
        ("ppl_ratio",     "PPL ratio",  9),
        ("hf_sr_baseline","HF SR base", 10),
        ("hf_sr_monotonic","HF SR mono",10),
        ("hf_sr_drop",    "SR drop",    8),
        ("hf_deg_baseline","Deg base",  9),
        ("hf_deg_monotonic","Deg mono", 9),
    ]
    header = "  ".join(f"{label:<{w}}" for _, label, w in cols)
    sep    = "  ".join("-"*w for _, _, w in cols)
    print(header)
    print(sep)
    for row in rows:
        parts = []
        for key, _, w in cols:
            val = row.get(key)
            if val is None:
                parts.append(f"{'N/A':<{w}}")
            elif isinstance(val, float):
                parts.append(f"{val:<{w}.3f}")
            else:
                parts.append(f"{str(val):<{w}}")
        print("  ".join(parts))
    print()


def compute_stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    var  = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def pick_winner(rows, ppl_ceiling, variants):
    """Return winning variant name or None."""
    # Group by variant; a variant only qualifies if ALL seed rows satisfy
    # the ppl ceiling and have a positive SR drop.
    by_variant = {}
    for row in rows:
        v = row["variant"]
        by_variant.setdefault(v, []).append(row)

    best_v = None
    best_mean_drop = float('-inf')

    for v in variants:
        vrows = by_variant.get(v, [])
        if not vrows:
            continue
        # Disqualify if any seed violates ppl ceiling
        qualified = True
        for r in vrows:
            ratio = r.get("ppl_ratio")
            if ratio is None or ratio > ppl_ceiling:
                qualified = False
                break
        if not qualified:
            continue
        # Disqualify if any seed has no SR drop measured or SR drop is not positive
        drops = [r.get("hf_sr_drop") for r in vrows]
        if any(d is None or d <= 0 for d in drops):
            continue
        mean_drop = sum(drops) / len(drops)
        if mean_drop > best_mean_drop:
            best_mean_drop = mean_drop
            best_v = v

    return best_v


def main():
    parser = argparse.ArgumentParser(description="Sweep aggregator")
    parser.add_argument("--seeds",        nargs="+", type=int, required=True)
    parser.add_argument("--variants",     nargs="+", required=True)
    parser.add_argument("--results-root", default="/persist/sweep_results",
                        help="Path prefix; cell dirs are <root>_seed<S>_<V>/")
    parser.add_argument("--ppl-ceiling",  type=float, default=2.0,
                        help="Max allowed ppl_monotonic/ppl_baseline ratio for winner")
    parser.add_argument("--pick-winner",  action="store_true")
    parser.add_argument("--winner-file",  default=None)
    parser.add_argument("--output",       default=None,
                        help="Path to write sweep_summary.json")
    args = parser.parse_args()

    print(f"\nSweep aggregate: seeds={args.seeds} variants={args.variants}")
    print(f"Results root: {args.results_root}_seed<S>_<V>/\n")

    rows = []
    for seed in args.seeds:
        for variant in args.variants:
            row = load_cell(args.results_root, seed, variant)
            rows.append(row)
            if row["ppl_baseline"] is None:
                print(f"  WARNING: {row['cell_dir']} - evaluation_results.json missing or incomplete")
            if row["hf_sr_baseline"] is None:
                print(f"  WARNING: {row['cell_dir']} - hotflip_results.json missing or incomplete")

    print_table(rows)

    # Per-variant summary over seeds
    if len(args.seeds) > 1:
        print("Per-variant summary (mean ± std across seeds):")
        header_line = (f"  {'Variant':<18}  {'PPL ratio':>9}  "
                       f"{'SR drop':>8}  {'HF SR mono':>10}")
        print(header_line)
        print("  " + "-"*60)
        for v in args.variants:
            vrows = [r for r in rows if r["variant"] == v]
            m_ratio, s_ratio   = compute_stats([r.get("ppl_ratio")  for r in vrows])
            m_drop,  s_drop    = compute_stats([r.get("hf_sr_drop") for r in vrows])
            m_sr,    s_sr      = compute_stats([r.get("hf_sr_monotonic") for r in vrows])
            ratio_s = f"{m_ratio:.3f}±{s_ratio:.3f}" if s_ratio is not None else (f"{m_ratio:.3f}" if m_ratio else "N/A")
            drop_s  = f"{m_drop:.3f}±{s_drop:.3f}"   if s_drop  is not None else (f"{m_drop:.3f}"  if m_drop  is not None else "N/A")
            sr_s    = f"{m_sr:.3f}±{s_sr:.3f}"       if s_sr    is not None else (f"{m_sr:.3f}"    if m_sr    is not None else "N/A")
            print(f"  {v:<18}  {ratio_s:>9}  {drop_s:>8}  {sr_s:>10}")
        print()

    # Winner selection
    winner = None
    if args.pick_winner or args.winner_file:
        winner = pick_winner(rows, args.ppl_ceiling, args.variants)
        if winner:
            print(f"Winner (ppl_ceiling={args.ppl_ceiling}x): {winner}")
            if args.winner_file:
                os.makedirs(os.path.dirname(args.winner_file) or ".", exist_ok=True)
                with open(args.winner_file, "w") as f:
                    f.write(winner)
                print(f"  Written to {args.winner_file}")
        else:
            print(f"No winner satisfies ppl_ceiling={args.ppl_ceiling}x. "
                  "Retry with higher ceiling or --phase-b <variant> to force.")

    # Write summary JSON
    if args.output:
        summary = {
            "seeds": args.seeds,
            "variants": args.variants,
            "ppl_ceiling_used": args.ppl_ceiling,
            "winner": winner,
            "cells": rows,
        }
        # Per-variant aggregated stats
        summary["by_variant"] = {}
        for v in args.variants:
            vrows = [r for r in rows if r["variant"] == v]
            m_ratio, s_ratio = compute_stats([r.get("ppl_ratio")          for r in vrows])
            m_drop,  s_drop  = compute_stats([r.get("hf_sr_drop")         for r in vrows])
            m_sr,    s_sr    = compute_stats([r.get("hf_sr_monotonic")     for r in vrows])
            m_srd,   s_srd   = compute_stats([r.get("hf_deg_monotonic")    for r in vrows])
            summary["by_variant"][v] = {
                "ppl_ratio_mean":       m_ratio,
                "ppl_ratio_std":        s_ratio,
                "hf_sr_drop_mean":      m_drop,
                "hf_sr_drop_std":       s_drop,
                "hf_sr_monotonic_mean": m_sr,
                "hf_sr_monotonic_std":  s_sr,
                "hf_deg_monotonic_mean":m_srd,
                "hf_deg_monotonic_std": s_srd,
            }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
