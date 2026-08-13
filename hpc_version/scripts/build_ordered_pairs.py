#!/usr/bin/env python3
"""
Build a templated ordered-pair dataset for the end-to-end order-preservation
experiment (Section "Order Preservation" of the paper / stage_8_order_preservation.py).

Generates chains of prompts that progressively strengthen along some
semantic axis -- e.g. escalating medication severity, or a process
description that accumulates safety constraints -- of exactly the kind
motivated in the paper's introduction:

    "The medication has side effects"
    (preceq) "The medication has severe side effects"
    (preceq) "The medication has severe side effects and requires medical
              supervision"

Each chain instantiates one of several axis *templates* (severity, safety
constraints, quantity, certainty, urgency, specificity, formality,
obligation strength, risk disclosure, permission level, evidence strength,
confidentiality, temporal scope) across many different topics, so the
corpus spans a genuinely diverse set of semantic directions rather than a
single hand-picked one.

The output is deterministic given --seed and is committed to the repository
(hpc_version/data/ordered_pairs.json) so the measurement in
stage_8_order_preservation.py is fully reproducible without needing to
regenerate the dataset. Re-running this script with the same seed is a
no-op (byte-identical output).

This script has no model or GPU dependency -- it is pure data generation
and can be run anywhere, including outside the HPC environment.

Usage:
    python build_ordered_pairs.py [--seed 42] [--fit-fraction 0.7] [--out PATH]
"""

import argparse
import hashlib
import json
import os
import random

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'ordered_pairs.json')

# ----------------------------------------------------------------------
# Topics: generic nouns/situations reused across axis templates so each
# axis is tested on a broad, overlapping set of subjects.
# ----------------------------------------------------------------------
TOPICS = [
    "the medication", "the procedure", "the chemical", "the software update",
    "the weather warning", "the financial transaction", "the construction site",
    "the vaccine", "the industrial process", "the road closure",
    "the data breach", "the recall notice", "the travel advisory",
    "the allergy reaction", "the equipment malfunction", "the contract clause",
    "the security vulnerability", "the diagnosis",
]

# ----------------------------------------------------------------------
# Axis templates. Each is a function topic -> ordered list of strings
# (weakest/least-strong first), representing one semantic direction along
# which strength monotonically increases.
# ----------------------------------------------------------------------


def _severity(topic):
    return [
        f"{topic} has side effects.".capitalize(),
        f"{topic} has severe side effects.".capitalize(),
        f"{topic} has severe side effects and requires medical supervision.".capitalize(),
        f"{topic} has severe, potentially fatal side effects and requires immediate medical supervision.".capitalize(),
    ]


def _safety_constraint(topic):
    return [
        f"Explain {topic}.".capitalize(),
        f"Explain {topic} safely.".capitalize(),
        f"Explain {topic} safely without providing actionable steps.".capitalize(),
        f"Explain {topic} safely, without providing actionable steps, and include a prominent warning.".capitalize(),
    ]


def _quantity(topic):
    return [
        f"A small amount of {topic} was affected.".capitalize(),
        f"A moderate amount of {topic} was affected.".capitalize(),
        f"A large amount of {topic} was affected.".capitalize(),
        f"Nearly all of {topic} was affected.".capitalize(),
    ]


def _certainty(topic):
    return [
        f"{topic} might be a problem.".capitalize(),
        f"{topic} is likely a problem.".capitalize(),
        f"{topic} is almost certainly a problem.".capitalize(),
        f"{topic} is confirmed to be a problem.".capitalize(),
    ]


def _urgency(topic):
    return [
        f"Please review {topic} when convenient.".capitalize(),
        f"Please review {topic} soon.".capitalize(),
        f"Please review {topic} as soon as possible.".capitalize(),
        f"Please review {topic} immediately; this cannot wait.".capitalize(),
    ]


def _specificity(topic):
    return [
        f"There is an issue with {topic}.".capitalize(),
        f"There is a known issue with {topic} affecting one component.".capitalize(),
        f"There is a known issue with {topic} affecting one component, identified in the audit log.".capitalize(),
        f"There is a known issue with {topic} affecting one component, identified in the audit log at line 42.".capitalize(),
    ]


def _formality(topic):
    return [
        f"Heads up about {topic}.".capitalize(),
        f"Please note the following regarding {topic}.".capitalize(),
        f"This notice serves to formally inform you regarding {topic}.".capitalize(),
        f"This constitutes formal, binding notification pertaining to {topic}, pursuant to applicable policy.".capitalize(),
    ]


def _obligation(topic):
    return [
        f"You may want to check {topic}.".capitalize(),
        f"You should check {topic}.".capitalize(),
        f"You must check {topic}.".capitalize(),
        f"You are required to check {topic} before proceeding, without exception.".capitalize(),
    ]


def _risk_disclosure(topic):
    return [
        f"{topic} carries some risk.".capitalize(),
        f"{topic} carries significant risk.".capitalize(),
        f"{topic} carries significant, well-documented risk.".capitalize(),
        f"{topic} carries significant, well-documented risk that has caused harm in the past.".capitalize(),
    ]


def _permission(topic):
    return [
        f"Access to {topic} is generally available.".capitalize(),
        f"Access to {topic} is restricted to authorized staff.".capitalize(),
        f"Access to {topic} is restricted to authorized staff with prior approval.".capitalize(),
        f"Access to {topic} is restricted to authorized staff with prior written approval and audit logging.".capitalize(),
    ]


def _evidence(topic):
    return [
        f"Some reports mention {topic}.".capitalize(),
        f"Multiple independent reports mention {topic}.".capitalize(),
        f"Multiple independent, peer-reviewed reports document {topic}.".capitalize(),
        f"Multiple independent, peer-reviewed reports, corroborated by internal audits, document {topic}.".capitalize(),
    ]


def _confidentiality(topic):
    return [
        f"Details about {topic} are shared internally.".capitalize(),
        f"Details about {topic} are shared only with the project team.".capitalize(),
        f"Details about {topic} are confidential and shared only with the project team.".capitalize(),
        f"Details about {topic} are strictly confidential, shared only with the project team under NDA.".capitalize(),
    ]


def _temporal_scope(topic):
    return [
        f"{topic} was reviewed this week.".capitalize(),
        f"{topic} has been reviewed every week this quarter.".capitalize(),
        f"{topic} has been reviewed every week for the past two years.".capitalize(),
        f"{topic} has been reviewed every week without exception for the past two years.".capitalize(),
    ]


AXES = {
    "severity": _severity,
    "safety_constraint": _safety_constraint,
    "quantity": _quantity,
    "certainty": _certainty,
    "urgency": _urgency,
    "specificity": _specificity,
    "formality": _formality,
    "obligation_strength": _obligation,
    "risk_disclosure": _risk_disclosure,
    "permission_level": _permission,
    "evidence_strength": _evidence,
    "confidentiality": _confidentiality,
    "temporal_scope": _temporal_scope,
}


def _topic_slug(topic):
    return topic.replace("the ", "").replace(" ", "_")


def build_chains():
    chains = []
    chain_id = 0
    for axis_name, template_fn in AXES.items():
        for topic in TOPICS:
            sequence = template_fn(topic)
            chains.append({
                "chain_id": chain_id,
                "axis": axis_name,
                "topic": _topic_slug(topic),
                "sequence": sequence,
            })
            chain_id += 1
    return chains


def build_pairs(chains, include_skip_pairs=False):
    """Adjacent pairs within each chain are the unit of 'ordered pair'
    evaluated in stage_8_order_preservation.py -- this keeps the evaluation
    set at the "few hundred held-out pairs" scale the plan calls for and
    avoids flooding it with many highly-correlated, easier non-adjacent
    (skip) comparisons. Set include_skip_pairs=True to additionally emit
    non-adjacent pairs (kept off by default; useful only for a probe-fitting
    ablation, not for the main measurement)."""
    pairs = []
    pair_id = 0
    for chain in chains:
        seq = chain["sequence"]
        gaps = [1] if not include_skip_pairs else range(1, len(seq))
        for i in range(len(seq)):
            for gap in gaps:
                j = i + gap
                if j >= len(seq):
                    continue
                pairs.append({
                    "pair_id": pair_id,
                    "chain_id": chain["chain_id"],
                    "axis": chain["axis"],
                    "topic": chain["topic"],
                    "weaker": seq[i],
                    "stronger": seq[j],
                    "level_gap": j - i,
                })
                pair_id += 1
    return pairs


def assign_splits(chains, fit_fraction, seed):
    """Split at the chain (not pair) level, stratified per axis, so no
    topic/chain appears in both the probe-fit and evaluation sets."""
    rng = random.Random(seed)
    by_axis = {}
    for chain in chains:
        by_axis.setdefault(chain["axis"], []).append(chain)

    splits = {}
    for axis_name, axis_chains in by_axis.items():
        indices = list(range(len(axis_chains)))
        rng.shuffle(indices)
        n_fit = max(1, round(len(indices) * fit_fraction))
        fit_idx = set(indices[:n_fit])
        for local_i, chain in enumerate(axis_chains):
            splits[chain["chain_id"]] = "fit" if local_i in fit_idx else "eval"
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Split-assignment seed (data content is deterministic).")
    parser.add_argument("--fit-fraction", type=float, default=0.7, help="Fraction of chains per axis used for probe fitting.")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output JSON path.")
    args = parser.parse_args()

    chains = build_chains()
    splits = assign_splits(chains, args.fit_fraction, args.seed)
    for chain in chains:
        chain["split"] = splits[chain["chain_id"]]

    pairs = build_pairs(chains)
    for pair in pairs:
        pair["split"] = splits[pair["chain_id"]]

    fit_pairs = sum(1 for p in pairs if p["split"] == "fit")
    eval_pairs = sum(1 for p in pairs if p["split"] == "eval")
    fit_chains = sum(1 for c in chains if c["split"] == "fit")
    eval_chains = sum(1 for c in chains if c["split"] == "eval")

    content_hash = hashlib.sha256(
        json.dumps([c["sequence"] for c in chains], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    dataset = {
        "metadata": {
            "description": (
                "Templated ordered-pair dataset for end-to-end order-preservation "
                "measurement (see paper Section 2 / stage_8_order_preservation.py)."
            ),
            "num_axes": len(AXES),
            "axes": sorted(AXES.keys()),
            "num_topics": len(TOPICS),
            "num_chains": len(chains),
            "num_pairs": len(pairs),
            "fit_fraction": args.fit_fraction,
            "split_seed": args.seed,
            "num_fit_chains": fit_chains,
            "num_eval_chains": eval_chains,
            "num_fit_pairs": fit_pairs,
            "num_eval_pairs": eval_pairs,
            "content_hash": content_hash,
        },
        "chains": chains,
        "pairs": pairs,
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Wrote {len(chains)} chains ({len(pairs)} ordered pairs) to {out_path}")
    print(f"  Axes: {len(AXES)}, topics: {len(TOPICS)}")
    print(f"  Fit split:  {fit_chains} chains / {fit_pairs} pairs")
    print(f"  Eval split: {eval_chains} chains / {eval_pairs} pairs")
    print(f"  Content hash: {content_hash}")


if __name__ == "__main__":
    main()
