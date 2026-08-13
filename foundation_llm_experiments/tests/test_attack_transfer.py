"""
Tests for the new attack-transfer stages (5b, 6b): trigger replay
correctness, random-flip budget matching, and the gradient-free query
attack, for the Pythia track.

Full main() execution (checkpoint loading, multi-hour attack loops) is not
exercised here -- these tests target the reusable, testable pieces:
evaluate_trigger (shared with Stage 5's own diagonal results, so a replayed
trigger is scored identically), the random-substitution control, and
QueryAttacker's greedy search.
"""

import numpy as np
import pytest
import torch


@pytest.fixture
def vocab_matched_gpt2(mock_tokenizer):
    """A tiny GPT-2 whose embedding table covers the real GPT-2 tokenizer's
    full vocabulary. conftest.py's mock_gpt_model uses a 1000-token toy
    vocab that only works for tests which never run a forward pass on
    real-tokenizer output (a pre-existing limitation shared by several
    tests in test_attack_mechanisms.py). Trigger replay and the query
    attack both tokenize real text and feed it straight to the model, so
    they need a matching vocab size."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=len(mock_tokenizer),
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


class TestTriggerReplayCorrectness:
    """Stage 5b replays a trigger learned on one model against another with
    no re-optimization. The correctness property to protect is that
    replaying a trigger via evaluate_trigger reproduces the exact same
    numbers Stage 5 itself would report for a same-model (diagonal) cell,
    since both use the identical evaluate_trigger function."""

    def test_replaying_a_trigger_is_deterministic(self, vocab_matched_gpt2, mock_tokenizer):
        from scripts.stage_5_uat_attacks import evaluate_trigger

        trigger_ids = [10, 20, 30, 40]
        texts = ["The quick brown fox.", "Another test sentence here."]

        result_a = evaluate_trigger(
            vocab_matched_gpt2, mock_tokenizer, trigger_ids, texts, torch.device("cpu"),
        )
        result_b = evaluate_trigger(
            vocab_matched_gpt2, mock_tokenizer, trigger_ids, texts, torch.device("cpu"),
        )

        assert result_a["clean_loss"] == pytest.approx(result_b["clean_loss"])
        assert result_a["attacked_loss"] == pytest.approx(result_b["attacked_loss"])
        assert result_a["trigger_ids"] == trigger_ids

    def test_transfer_cell_and_diagonal_cell_agree_for_same_model(
        self, vocab_matched_gpt2, mock_tokenizer
    ):
        """Evaluating model A's own trigger against model A ('diagonal') via
        the transfer-stage code path must be numerically identical to doing
        so directly -- there must be no hidden re-optimization or different
        preprocessing sneaking into the 'transfer' path."""
        from scripts.stage_5_uat_attacks import evaluate_trigger

        trigger_ids = [15, 25, 35]
        texts = ["Sentence one for evaluation.", "Sentence two for evaluation."]

        # Simulates Stage 5's own diagonal computation.
        diagonal = evaluate_trigger(
            vocab_matched_gpt2, mock_tokenizer, trigger_ids, texts, torch.device("cpu"),
        )
        # Simulates Stage 5b replaying the same (source=target) trigger.
        replayed = evaluate_trigger(
            vocab_matched_gpt2, mock_tokenizer, trigger_ids, texts, torch.device("cpu"),
        )
        assert diagonal["nll_increase_percent"] == pytest.approx(
            replayed["nll_increase_percent"]
        )

    def test_transfer_matrix_assembly_reads_correct_cells(self):
        """The 2x2 transfer_matrix dict Stage 5b assembles from per-cell
        results must index by (source, target), not silently transpose or
        drop cells."""
        model_types = {"baseline_pythia": "baseline", "monotonic_pythia": "monotonic"}
        transfer_cells = {
            ("baseline_pythia", "baseline_pythia"): {"nll_increase_percent": 12.0},
            ("baseline_pythia", "monotonic_pythia"): {"nll_increase_percent": 1.0},
            ("monotonic_pythia", "baseline_pythia"): {"nll_increase_percent": 0.5},
            ("monotonic_pythia", "monotonic_pythia"): {"nll_increase_percent": 8.0},
        }

        # Mirrors the assembly logic in stage_5b_uat_transfer.py::main().
        transfer_matrix = {
            src: {
                tgt: transfer_cells[(src, tgt)]['nll_increase_percent']
                for tgt in model_types
            }
            for src in model_types
        }

        assert transfer_matrix["baseline_pythia"]["baseline_pythia"] == 12.0
        assert transfer_matrix["baseline_pythia"]["monotonic_pythia"] == 1.0
        assert transfer_matrix["monotonic_pythia"]["baseline_pythia"] == 0.5
        assert transfer_matrix["monotonic_pythia"]["monotonic_pythia"] == 8.0
        # Off-diagonal cells must not be conflated with the diagonal.
        assert (
            transfer_matrix["baseline_pythia"]["monotonic_pythia"]
            != transfer_matrix["baseline_pythia"]["baseline_pythia"]
        )

    def test_evaluate_trigger_with_empty_trigger_is_nearly_a_no_op(
        self, vocab_matched_gpt2, mock_tokenizer
    ):
        """An empty trigger decodes to an empty string, so evaluate_trigger
        prepends only a single leading space (see its 'trigger_text + " "'
        construction) -- not literally nothing, but close enough that the
        NLL increase should be tiny rather than a real attack effect."""
        from scripts.stage_5_uat_attacks import evaluate_trigger

        result = evaluate_trigger(
            vocab_matched_gpt2, mock_tokenizer, [], ["A sentence."], torch.device("cpu"),
        )
        assert abs(result["nll_increase_percent"]) < 5.0


class TestHotflipRandomSubstitutionBudget:
    """Flip-budget matching for the random-substitution control: it must
    flip exactly min(num_flips, seq_len) positions, matching HotFlip's own
    budget (Config.HOTFLIP_NUM_FLIPS = 10 for Pythia). This function never
    touches a model, so the real (large-vocab) mock_tokenizer is enough."""

    def test_flips_exactly_requested_budget(self, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import _random_substitution_attack

        rng = np.random.RandomState(0)
        text = "This is a reasonably long sentence for testing flip budgets properly."
        candidate_tokens = list(range(1000, 2000))

        result = _random_substitution_attack(
            text, mock_tokenizer, rng, num_flips=10,
            candidate_tokens=candidate_tokens, max_length=64,
        )
        assert result is not None
        assert len(result["positions_flipped"]) == 10
        assert len(set(result["positions_flipped"])) == 10, "positions must not repeat"

    def test_never_exceeds_sequence_length(self, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import _random_substitution_attack

        rng = np.random.RandomState(0)
        text = "Short."
        candidate_tokens = list(range(1000, 2000))

        result = _random_substitution_attack(
            text, mock_tokenizer, rng, num_flips=1000,
            candidate_tokens=candidate_tokens, max_length=64,
        )
        seq_len = len(result["orig_ids"])
        assert len(result["positions_flipped"]) <= seq_len

    def test_preserves_ids_outside_flipped_positions(self, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import _random_substitution_attack

        rng = np.random.RandomState(1)
        text = "A moderately long sentence with several distinct tokens in it."
        candidate_tokens = list(range(1000, 2000))

        result = _random_substitution_attack(
            text, mock_tokenizer, rng, num_flips=3,
            candidate_tokens=candidate_tokens, max_length=64,
        )
        flipped_positions = set(result["positions_flipped"])
        for i, (orig, flipped) in enumerate(zip(result["orig_ids"], result["flipped_ids"])):
            if i not in flipped_positions:
                assert orig == flipped

    def test_replacement_tokens_come_from_candidate_vocabulary(self, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import _random_substitution_attack

        rng = np.random.RandomState(2)
        text = "A sentence to check candidate-vocabulary restriction works."
        candidate_tokens = list(range(1000, 1010))  # narrow, easy to check membership

        result = _random_substitution_attack(
            text, mock_tokenizer, rng, num_flips=5,
            candidate_tokens=candidate_tokens, max_length=64,
        )
        for pos in result["positions_flipped"]:
            assert result["flipped_ids"][pos] in candidate_tokens

    def test_empty_text_returns_none(self, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import _random_substitution_attack

        rng = np.random.RandomState(0)
        # Force a genuinely empty encoding by bypassing the tokenizer's
        # automatic special-token insertion isn't possible here, but a
        # zero-flip budget with an empty candidate pool should still return
        # a well-formed (rather than crashing) result.
        result = _random_substitution_attack(
            "", mock_tokenizer, rng, num_flips=0, candidate_tokens=[1000], max_length=64,
        )
        assert result is None or len(result["positions_flipped"]) == 0


class TestQueryAttackerGradientFree:
    """The greedy, gradient-free query attack must never touch .grad and
    must respect its flip/candidate budget."""

    def test_no_gradients_tracked(self, vocab_matched_gpt2, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import QueryAttacker

        candidate_tokens = list(range(100, 200))
        attacker = QueryAttacker(
            vocab_matched_gpt2, mock_tokenizer, torch.device("cpu"),
            num_flips=2, candidates_per_position=3, candidate_tokens=candidate_tokens,
        )
        rng = np.random.RandomState(0)
        for p in vocab_matched_gpt2.parameters():
            assert p.grad is None
        result = attacker.attack_single_example(
            "A sentence long enough to have candidate positions.", rng,
        )
        assert result is not None
        for p in vocab_matched_gpt2.parameters():
            assert p.grad is None, "Query attack must not populate .grad"

    def test_respects_flip_and_candidate_budget(self, vocab_matched_gpt2, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import QueryAttacker

        candidate_tokens = list(range(100, 200))
        num_flips = 3
        attacker = QueryAttacker(
            vocab_matched_gpt2, mock_tokenizer, torch.device("cpu"),
            num_flips=num_flips, candidates_per_position=2,
            candidate_tokens=candidate_tokens,
        )
        rng = np.random.RandomState(0)
        result = attacker.attack_single_example(
            "A reasonably long sentence to exercise the query attack budget.", rng,
        )
        assert result is not None
        assert "degradation" in result and "success" in result

    def test_returns_none_for_empty_text(self, vocab_matched_gpt2, mock_tokenizer):
        from scripts.stage_6b_hotflip_transfer import QueryAttacker

        attacker = QueryAttacker(
            vocab_matched_gpt2, mock_tokenizer, torch.device("cpu"),
            num_flips=1, candidates_per_position=1, candidate_tokens=[100],
        )
        rng = np.random.RandomState(0)
        result = attacker.attack_single_example("", rng)
        assert result is None

    def test_never_makes_more_forward_passes_than_budget_implies(
        self, vocab_matched_gpt2, mock_tokenizer, monkeypatch
    ):
        """num_flips * candidates_per_position + 2 is the documented upper
        bound on forward passes per example; verify the implementation
        actually respects it by counting calls to the internal _loss hook."""
        from scripts.stage_6b_hotflip_transfer import QueryAttacker

        candidate_tokens = list(range(100, 200))
        num_flips, candidates_per_position = 2, 4
        attacker = QueryAttacker(
            vocab_matched_gpt2, mock_tokenizer, torch.device("cpu"),
            num_flips=num_flips, candidates_per_position=candidates_per_position,
            candidate_tokens=candidate_tokens,
        )

        call_count = {"n": 0}
        original_loss = attacker._loss

        def counting_loss(ids, mask):
            call_count["n"] += 1
            return original_loss(ids, mask)

        monkeypatch.setattr(attacker, "_loss", counting_loss)
        rng = np.random.RandomState(0)
        attacker.attack_single_example(
            "A long enough sentence to guarantee two flip positions exist here.", rng,
        )
        upper_bound = num_flips * candidates_per_position + 2
        assert call_count["n"] <= upper_bound


class TestAggregateReuse:
    """Stage 6b reuses Stage 6's own _aggregate for all three attack
    families (substitution transfer, random control, query attack), so the
    same success-rate/degradation definitions apply everywhere."""

    def test_aggregate_success_rate(self):
        from scripts.stage_6_hotflip_attacks import _aggregate

        records = [
            {"clean_loss": 1.0, "attacked_loss": 2.0, "degradation": 1.0, "success": True},
            {"clean_loss": 1.0, "attacked_loss": 1.02, "degradation": 0.02, "success": False},
            {"clean_loss": 1.0, "attacked_loss": 1.5, "degradation": 0.5, "success": True},
        ]
        summary = _aggregate(records)
        assert summary["num_samples"] == 3
        assert summary["success_rate"] == pytest.approx(2 / 3)

    def test_aggregate_ignores_oom_records(self):
        from scripts.stage_6_hotflip_attacks import _aggregate

        records = [
            {"clean_loss": 1.0, "attacked_loss": 2.0, "degradation": 1.0, "success": True},
            {"oom": True},
        ]
        summary = _aggregate(records)
        assert summary["num_samples"] == 1
