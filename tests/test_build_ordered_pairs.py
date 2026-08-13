"""
Tests for hpc_version/scripts/build_ordered_pairs.py: the templated
ordered-pair dataset generator that feeds stage_8_order_preservation.py.

This script has no model or GPU dependency, so it is exercised directly and
exhaustively (it is also the one new stage-adjacent script not excluded
from the coverage gate, since it does not match the hpc_version/scripts/
stage_*.py coverage-omit pattern).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version" / "scripts"))

import build_ordered_pairs
from build_ordered_pairs import (
    AXES,
    TOPICS,
    build_chains,
    build_pairs,
    assign_splits,
    _topic_slug,
)


class TestTopicSlug:
    def test_strips_leading_article_and_spaces(self):
        assert _topic_slug("the medication") == "medication"
        assert _topic_slug("the data breach") == "data_breach"

    def test_leaves_topics_without_article_alone(self):
        assert _topic_slug("something else") == "something_else"


class TestBuildChains:
    def test_one_chain_per_axis_topic_combination(self):
        chains = build_chains()
        assert len(chains) == len(AXES) * len(TOPICS)

    def test_chain_ids_are_unique_and_sequential(self):
        chains = build_chains()
        ids = [c["chain_id"] for c in chains]
        assert ids == list(range(len(chains)))

    def test_every_chain_has_a_strictly_increasing_sequence_length(self):
        chains = build_chains()
        for chain in chains:
            assert len(chain["sequence"]) == 4, (
                "Every axis template must emit exactly 4 escalating levels"
            )
            # Levels should be textually distinct (monotonically escalating
            # phrasing), not accidental duplicates.
            assert len(set(chain["sequence"])) == 4

    def test_every_chain_axis_is_a_known_axis(self):
        chains = build_chains()
        for chain in chains:
            assert chain["axis"] in AXES

    def test_chain_sequences_are_capitalized_sentences(self):
        chains = build_chains()
        for chain in chains:
            for sentence in chain["sequence"]:
                assert sentence[0].isupper()
                assert sentence.endswith(".")


class TestBuildPairs:
    def test_default_emits_only_adjacent_pairs(self):
        chains = build_chains()
        pairs = build_pairs(chains, include_skip_pairs=False)
        # 4-level chains -> 3 adjacent pairs each.
        assert len(pairs) == len(chains) * 3
        for pair in pairs:
            assert pair["level_gap"] == 1

    def test_skip_pairs_include_non_adjacent_comparisons(self):
        chains = build_chains()
        adjacent_only = build_pairs(chains, include_skip_pairs=False)
        with_skips = build_pairs(chains, include_skip_pairs=True)
        assert len(with_skips) > len(adjacent_only)
        gaps = {p["level_gap"] for p in with_skips}
        assert gaps == {1, 2, 3}

    def test_pair_ids_are_unique_and_sequential(self):
        chains = build_chains()
        pairs = build_pairs(chains)
        ids = [p["pair_id"] for p in pairs]
        assert ids == list(range(len(pairs)))

    def test_weaker_always_precedes_stronger_in_source_sequence(self):
        chains = build_chains()
        pairs = build_pairs(chains, include_skip_pairs=True)
        chains_by_id = {c["chain_id"]: c["sequence"] for c in chains}
        for pair in pairs:
            seq = chains_by_id[pair["chain_id"]]
            assert seq.index(pair["weaker"]) < seq.index(pair["stronger"])

    def test_pairs_reference_valid_chain_metadata(self):
        chains = build_chains()
        pairs = build_pairs(chains)
        chains_by_id = {c["chain_id"]: c for c in chains}
        for pair in pairs:
            chain = chains_by_id[pair["chain_id"]]
            assert pair["axis"] == chain["axis"]
            assert pair["topic"] == chain["topic"]


class TestAssignSplits:
    def test_every_chain_gets_a_split(self):
        chains = build_chains()
        splits = assign_splits(chains, fit_fraction=0.7, seed=42)
        assert set(splits.keys()) == {c["chain_id"] for c in chains}
        assert set(splits.values()) <= {"fit", "eval"}

    def test_fit_fraction_approximately_respected_per_axis(self):
        chains = build_chains()
        splits = assign_splits(chains, fit_fraction=0.7, seed=42)
        by_axis = {}
        for chain in chains:
            by_axis.setdefault(chain["axis"], []).append(chain["chain_id"])
        for axis, chain_ids in by_axis.items():
            fit_count = sum(1 for cid in chain_ids if splits[cid] == "fit")
            fraction = fit_count / len(chain_ids)
            # Rounding on small per-axis counts (18 topics) means this can't
            # be exact, but it should be close to the requested 0.7.
            assert 0.5 <= fraction <= 0.9

    def test_every_axis_has_at_least_one_fit_chain(self):
        chains = build_chains()
        splits = assign_splits(chains, fit_fraction=0.7, seed=42)
        by_axis = {}
        for chain in chains:
            by_axis.setdefault(chain["axis"], []).append(chain["chain_id"])
        for axis, chain_ids in by_axis.items():
            assert any(splits[cid] == "fit" for cid in chain_ids)

    def test_deterministic_given_same_seed(self):
        chains = build_chains()
        splits_a = assign_splits(chains, fit_fraction=0.7, seed=123)
        splits_b = assign_splits(chains, fit_fraction=0.7, seed=123)
        assert splits_a == splits_b

    def test_different_seeds_can_change_the_split(self):
        chains = build_chains()
        splits_a = assign_splits(chains, fit_fraction=0.7, seed=1)
        splits_b = assign_splits(chains, fit_fraction=0.7, seed=2)
        assert splits_a != splits_b


class TestMainCLI:
    """The script's CLI entry point: writes a deterministic, self-describing
    JSON dataset to --out, matching the fields stage_8_order_preservation.py
    expects (metadata, chains, pairs)."""

    def test_main_writes_expected_json_structure(self, tmp_path, monkeypatch):
        out_path = tmp_path / "ordered_pairs.json"
        monkeypatch.setattr(
            sys, "argv",
            ["build_ordered_pairs.py", "--seed", "1", "--fit-fraction", "0.7", "--out", str(out_path)],
        )
        build_ordered_pairs.main()

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)

        assert "metadata" in data and "chains" in data and "pairs" in data
        meta = data["metadata"]
        assert meta["num_chains"] == len(AXES) * len(TOPICS)
        assert meta["num_pairs"] == meta["num_fit_pairs"] + meta["num_eval_pairs"]
        assert meta["num_chains"] == meta["num_fit_chains"] + meta["num_eval_chains"]
        assert len(meta["content_hash"]) == 16

    def test_main_is_deterministic_given_same_seed(self, tmp_path, monkeypatch):
        out_a = tmp_path / "a.json"
        out_b = tmp_path / "b.json"

        monkeypatch.setattr(
            sys, "argv",
            ["build_ordered_pairs.py", "--seed", "7", "--out", str(out_a)],
        )
        build_ordered_pairs.main()
        monkeypatch.setattr(
            sys, "argv",
            ["build_ordered_pairs.py", "--seed", "7", "--out", str(out_b)],
        )
        build_ordered_pairs.main()

        with open(out_a) as f:
            data_a = json.load(f)
        with open(out_b) as f:
            data_b = json.load(f)

        assert data_a["metadata"]["content_hash"] == data_b["metadata"]["content_hash"]
        assert data_a["pairs"] == data_b["pairs"]
        assert data_a["chains"] == data_b["chains"]

    def test_main_every_pair_has_a_split(self, tmp_path, monkeypatch):
        out_path = tmp_path / "ordered_pairs.json"
        monkeypatch.setattr(
            sys, "argv",
            ["build_ordered_pairs.py", "--seed", "3", "--out", str(out_path)],
        )
        build_ordered_pairs.main()
        with open(out_path) as f:
            data = json.load(f)
        for pair in data["pairs"]:
            assert pair["split"] in ("fit", "eval")
        for chain in data["chains"]:
            assert chain["split"] in ("fit", "eval")


class TestNoLeakageBetweenFitAndEval:
    """The critical correctness property for this dataset: no topic/chain
    may appear in both the probe-fit and evaluation splits, since that
    would let the probe measurement leak information about the eval set."""

    def test_no_chain_in_both_splits(self):
        chains = build_chains()
        splits = assign_splits(chains, fit_fraction=0.7, seed=42)
        for chain in chains:
            chain["split"] = splits[chain["chain_id"]]
        pairs = build_pairs(chains)
        for pair in pairs:
            pair["split"] = splits[pair["chain_id"]]

        fit_chain_ids = {c["chain_id"] for c in chains if c["split"] == "fit"}
        eval_chain_ids = {c["chain_id"] for c in chains if c["split"] == "eval"}
        assert fit_chain_ids.isdisjoint(eval_chain_ids)

        fit_pair_chain_ids = {p["chain_id"] for p in pairs if p["split"] == "fit"}
        eval_pair_chain_ids = {p["chain_id"] for p in pairs if p["split"] == "eval"}
        assert fit_pair_chain_ids.isdisjoint(eval_pair_chain_ids)
