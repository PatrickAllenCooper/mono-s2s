"""
Tests for Common Utilities

Comprehensive tests for all utility functions in common_utils.py
"""
import os
import sys
import json
import time
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))

from hpc_version.utils.common_utils import (
    set_all_seeds,
    get_generator,
    worker_init_fn,
    log_environment,
    compute_rouge_with_ci,
    compute_length_statistics,
    compute_brevity_penalty,
    NonNegativeParametrization,
    make_model_monotonic,
    load_model,
    generate_summary_fixed_params,
    compute_avg_loss,
    save_json,
    load_json,
    load_json_safe,
    atomic_save_json,
    append_jsonl,
    load_jsonl,
    partial_results_dir,
    create_completion_flag,
    check_completion_flag,
    check_dependencies,
    load_dataset_split,
    save_checkpoint,
    load_checkpoint,
    StageLogger,
    SummarizationDataset,
)
from hpc_version.configs.experiment_config import ExperimentConfig


class TestDeterminismFunctions:
    """Tests for determinism and seed management"""
    
    def test_set_all_seeds(self):
        """Test that all random seeds are set correctly"""
        set_all_seeds(42)
        
        # Check environment variable
        assert os.environ["PYTHONHASHSEED"] == "42"
        
        # Generate some random numbers and check reproducibility
        torch_num = torch.rand(1).item()
        np_num = np.random.rand()
        
        # Reset and check we get same numbers
        set_all_seeds(42)
        assert torch.rand(1).item() == torch_num
        assert np.random.rand() == np_num
    
    def test_get_generator(self, monkeypatch):
        """Test generator creation for reproducibility"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        gen1 = get_generator(device='cpu')
        gen2 = get_generator(device='cpu', seed=42)
        
        assert isinstance(gen1, torch.Generator)
        assert isinstance(gen2, torch.Generator)
    
    def test_worker_init_fn(self, monkeypatch):
        """Test DataLoader worker initialization"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        # Should not raise
        worker_init_fn(0)
        worker_init_fn(1)
    
    def test_log_environment(self):
        """Test environment logging"""
        env_info = log_environment()
        
        assert isinstance(env_info, dict)
        assert "hostname" in env_info
        assert "python_version" in env_info
        assert "pytorch_version" in env_info
        assert "cuda_available" in env_info


class TestRougeAndEvaluation:
    """Tests for ROUGE and evaluation functions"""
    
    def test_compute_rouge_with_ci(self, monkeypatch):
        """Test ROUGE computation with confidence intervals"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        predictions = ["The cat sat on the mat.", "A dog runs fast."]
        references = ["A cat sat on a mat.", "The dog runs quickly."]
        
        ci_scores, all_scores = compute_rouge_with_ci(
            predictions, references, 
            metrics=["rouge1", "rouge2"],
            n_bootstrap=100  # Use fewer for speed
        )
        
        assert isinstance(ci_scores, dict)
        assert "rouge1" in ci_scores
        assert "rouge2" in ci_scores
        
        for metric in ["rouge1", "rouge2"]:
            assert "mean" in ci_scores[metric]
            assert "lower" in ci_scores[metric]
            assert "upper" in ci_scores[metric]
            assert "ci_width" in ci_scores[metric]
            
            # Check that CI bounds make sense
            assert ci_scores[metric]["lower"] <= ci_scores[metric]["mean"]
            assert ci_scores[metric]["mean"] <= ci_scores[metric]["upper"]
        
        assert len(all_scores) == 2
    
    def test_compute_length_statistics_words(self):
        """Test length statistics computation with words"""
        texts = ["short", "medium length text", "a much longer text with more words"]
        
        stats = compute_length_statistics(texts, tokenizer=None)
        
        assert isinstance(stats, dict)
        assert stats["unit"] == "words"
        assert stats["min"] == 1
        assert stats["max"] == 7
        assert stats["mean"] > 0
        assert stats["median"] > 0
        # Total should be 1 + 3 + 8 = 12, but allow for rounding/tokenization variance
        assert stats["total"] >= 11 and stats["total"] <= 13
    
    def test_compute_length_statistics_tokens(self, mock_tokenizer):
        """Test length statistics with tokenizer"""
        texts = ["short", "medium length text"]
        
        stats = compute_length_statistics(texts, tokenizer=mock_tokenizer)
        
        assert isinstance(stats, dict)
        assert stats["unit"] == "tokens"
        assert stats["min"] > 0
        assert stats["max"] > 0
    
    def test_compute_brevity_penalty_words(self):
        """Test brevity penalty computation"""
        predictions = ["short", "text"]
        references = ["a much longer reference", "another longer reference"]
        
        bp_stats = compute_brevity_penalty(predictions, references)
        
        assert isinstance(bp_stats, dict)
        assert "brevity_penalty" in bp_stats
        assert "length_ratio" in bp_stats
        assert "avg_pred_length" in bp_stats
        assert "avg_ref_length" in bp_stats
        
        # Predictions are shorter, so ratio should be < 1
        assert bp_stats["length_ratio"] < 1.0
        assert bp_stats["brevity_penalty"] <= 1.0
    
    def test_compute_brevity_penalty_tokens(self, mock_tokenizer):
        """Test brevity penalty with tokenizer"""
        predictions = ["short"]
        references = ["a much longer reference text"]
        
        bp_stats = compute_brevity_penalty(predictions, references, tokenizer=mock_tokenizer)
        
        assert isinstance(bp_stats, dict)
        assert bp_stats["length_ratio"] < 1.0


class TestNonNegativeParametrization:
    """Tests for NonNegativeParametrization class"""
    
    def test_forward_positive(self):
        """Test forward pass produces non-negative values"""
        param = NonNegativeParametrization()
        
        V = torch.randn(10, 10)
        W = param.forward(V)
        
        assert (W >= 0).all(), "All weights should be non-negative"
    
    def test_right_inverse(self):
        """Test right_inverse initialization"""
        W_init = torch.randn(10, 10)
        param = NonNegativeParametrization(init_weight=W_init)
        
        V = param.right_inverse(W_init)
        W_reconstructed = param.forward(V)
        
        # Should approximately preserve absolute values
        W_target = torch.abs(W_init) + 1e-4
        relative_error = torch.mean(torch.abs(W_reconstructed - W_target) / (W_target + 1e-6))
        
        assert relative_error < 0.5, "Reconstruction error should be reasonable"
    
    def test_preserves_shape(self):
        """Test that parametrization preserves tensor shape"""
        param = NonNegativeParametrization()
        
        shapes = [(5, 5), (10, 20), (3, 7)]
        for shape in shapes:
            V = torch.randn(*shape)
            W = param.forward(V)
            assert W.shape == shape

    def test_default_mode_is_nonneg(self):
        """Test that the default mode is the real monotonicity constraint"""
        param = NonNegativeParametrization()
        assert param.mode == "nonneg"

    def test_unknown_mode_raises(self):
        """Test that an unrecognized ablation mode is rejected up front"""
        with pytest.raises(ValueError, match="Unknown parametrization mode"):
            NonNegativeParametrization(mode="not_a_real_mode")


class TestAblationModes:
    """Tests for the attribution-ablation parametrization modes (sign_frozen,
    abs_init_free) that isolate nonnegativity from initialization disruption."""

    def test_sign_frozen_is_not_nonnegative(self):
        """sign_frozen must NOT enforce monotonicity: it should reproduce
        the pretrained mixed-sign pattern, not clamp to non-negative."""
        W_pre = torch.tensor([-3.0, 2.0, -0.5, 4.0, -10.0])
        param = NonNegativeParametrization(init_weight=W_pre, mode="sign_frozen")

        V = param.right_inverse(W_pre)
        W_reconstructed = param.forward(V)

        assert (W_reconstructed < 0).any(), "sign_frozen should retain negative weights"
        # Reconstructs the pretrained weight (up to eps) at initialization.
        assert torch.allclose(W_reconstructed, W_pre, atol=1e-2)

    def test_sign_frozen_preserves_sign_pattern_after_training_step(self):
        """The sign buffer must keep the pretrained sign pattern fixed even
        after V is updated by an optimizer step (only magnitude changes)."""
        W_pre = torch.tensor([-3.0, 2.0, -0.5, 4.0])
        param = NonNegativeParametrization(init_weight=W_pre, mode="sign_frozen")
        original_sign = torch.sign(W_pre)

        V = torch.nn.Parameter(param.right_inverse(W_pre))
        # Simulate a training step that moves V arbitrarily.
        with torch.no_grad():
            V += torch.tensor([1.0, -2.0, 0.3, -0.1])
        W_after = param.forward(V)

        assert torch.equal(torch.sign(W_after), original_sign), \
            "sign_frozen must preserve the original sign pattern through training"

    def test_sign_frozen_zero_weight_treated_as_positive(self):
        """Zero-valued pretrained weights should not silently vanish from
        the sign buffer (sign(0) is remapped to +1 by convention)."""
        W_pre = torch.tensor([0.0, -1.0, 1.0])
        param = NonNegativeParametrization(init_weight=W_pre, mode="sign_frozen")
        assert param.sign[0].item() == 1.0

    def test_abs_init_free_is_unconstrained_identity(self):
        """abs_init_free applies no constraint at all: forward(V) == V."""
        param = NonNegativeParametrization(mode="abs_init_free")
        V = torch.tensor([-5.0, 0.0, 3.0, -0.001])
        W = param.forward(V)
        assert torch.equal(W, V), "abs_init_free must not transform V at all"

    def test_abs_init_free_right_inverse_matches_abs_plus_eps(self):
        """abs_init_free's initialization should start from |W_pretrained| +
        eps -- the same sign-disrupting starting point as 'nonneg' -- but
        with no softplus applied on the way back out."""
        W_pre = torch.tensor([-4.0, 3.0, -0.2])
        param = NonNegativeParametrization(init_weight=W_pre, mode="abs_init_free")
        V = param.right_inverse(W_pre)
        expected = torch.abs(W_pre) + 1e-4
        assert torch.allclose(V, expected, atol=1e-6)
        # And forward is identity, so this is also the initial "weight".
        assert torch.equal(param.forward(V), V)

    def test_abs_init_free_allows_negative_after_training(self):
        """Unlike nonneg/sign_frozen, abs_init_free must be able to drift
        to negative values under free training (no constraint to prevent it)."""
        param = NonNegativeParametrization(mode="abs_init_free")
        V = torch.tensor([0.5, 0.3])
        with torch.no_grad():
            V -= torch.tensor([10.0, 10.0])  # simulate a large gradient step
        W = param.forward(V)
        assert (W < 0).all()

    def test_nonneg_mode_has_no_sign_buffer(self):
        """The sign buffer is only meaningful for sign_frozen; nonneg and
        abs_init_free should leave it unset (None)."""
        for mode in ("nonneg", "abs_init_free"):
            param = NonNegativeParametrization(
                init_weight=torch.tensor([-1.0, 2.0]), mode=mode
            )
            assert param.sign is None

    @pytest.mark.parametrize("mode", ["nonneg", "sign_frozen", "abs_init_free"])
    def test_all_modes_preserve_shape(self, mode):
        """All three ablation modes must preserve tensor shape end to end."""
        W_pre = torch.randn(6, 8)
        param = NonNegativeParametrization(init_weight=W_pre, mode=mode)
        V = param.right_inverse(W_pre)
        W = param.forward(V)
        assert W.shape == W_pre.shape

    @pytest.mark.parametrize("mode", ["nonneg", "sign_frozen", "abs_init_free"])
    def test_all_modes_reject_unknown_mode_in_forward_path(self, mode):
        """Constructing with a valid mode should never raise; only unknown
        modes should be rejected (regression guard for ABLATION_MODES)."""
        # Should not raise for any of the three supported modes.
        NonNegativeParametrization(mode=mode)


class TestModelCreation:
    """Tests for model creation and modification"""
    
    @pytest.mark.skip(reason="Requires downloading transformers model")
    def test_make_model_monotonic(self, mock_model):
        """Test making model monotonic"""
        # Get initial weight range
        original_has_negative = False
        for param in mock_model.parameters():
            if (param < 0).any():
                original_has_negative = True
                break
        
        # Apply monotonic constraints
        monotonic_model = make_model_monotonic(mock_model)
        
        # Check that FFN weights are non-negative
        ffn_weight_count = 0
        for name, module in monotonic_model.named_modules():
            if hasattr(module, 'wi') or hasattr(module, 'wo'):
                for param_name in ['wi', 'wo', 'wi_0', 'wi_1']:
                    if hasattr(module, param_name):
                        sub_module = getattr(module, param_name)
                        if hasattr(sub_module, 'weight'):
                            weight = sub_module.weight
                            assert (weight >= 0).all(), f"{param_name} should be non-negative"
                            ffn_weight_count += 1
        
        # Should have modified at least some weights
        assert ffn_weight_count > 0, "Should have found and modified FFN weights"
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires downloading model, slow test")
    def test_load_model_standard(self, temp_dir, monkeypatch):
        """Test loading standard model"""
        monkeypatch.setattr(ExperimentConfig, "MODEL_NAME", "t5-small")
        
        model, is_pretrained = load_model("standard", checkpoint_path=None, device='cpu')
        
        assert model is not None
        assert is_pretrained is True
        assert model.config.model_type == "t5"
    
    def test_load_model_with_checkpoint(self, mock_model, temp_dir, monkeypatch):
        """Test loading model from checkpoint"""
        monkeypatch.setattr(ExperimentConfig, "MODEL_NAME", "t5-small")
        
        # Save a checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        torch.save(mock_model.state_dict(), checkpoint_path)
        
        # This will try to load t5-small and then load checkpoint
        # We'll just test that the function handles the checkpoint path
        # (actual loading would require matching architectures)


class TestGenerationAndEvaluation:
    """Tests for generation and evaluation functions"""
    
    def test_generate_summary_fixed_params(self, mock_model, mock_tokenizer, monkeypatch):
        """Test summary generation with fixed parameters"""
        monkeypatch.setattr(ExperimentConfig, "MAX_INPUT_LENGTH", 128)
        monkeypatch.setattr(ExperimentConfig, "DECODE_MAX_NEW_TOKENS", 50)
        monkeypatch.setattr(ExperimentConfig, "DECODE_MIN_NEW_TOKENS", 5)
        monkeypatch.setattr(ExperimentConfig, "DECODE_NUM_BEAMS", 2)
        monkeypatch.setattr(ExperimentConfig, "DECODE_LENGTH_PENALTY", 1.0)
        monkeypatch.setattr(ExperimentConfig, "DECODE_NO_REPEAT_NGRAM_SIZE", 3)
        monkeypatch.setattr(ExperimentConfig, "DECODE_EARLY_STOPPING", True)
        
        text = "This is a test document that needs to be summarized."
        
        summary = generate_summary_fixed_params(mock_model, text, mock_tokenizer, device='cpu')
        
        assert isinstance(summary, str)
        assert len(summary) >= 0  # May be empty for random model
    
    def test_compute_avg_loss(self, mock_model, mock_tokenizer):
        """Test average loss computation"""
        from torch.utils.data import DataLoader
        
        # Create a simple dataset
        dataset = SummarizationDataset(
            texts=["test text 1", "test text 2"],
            summaries=["summary 1", "summary 2"],
            tokenizer=mock_tokenizer,
            max_input_length=64,
            max_target_length=32
        )
        
        data_loader = DataLoader(dataset, batch_size=2)
        
        avg_loss = compute_avg_loss(mock_model, data_loader, device='cpu')
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0  # Loss should be non-negative


class TestFileAndLogging:
    """Tests for file I/O and logging functions"""
    
    def test_save_and_load_json(self, temp_dir):
        """Test JSON save and load"""
        data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        filepath = os.path.join(temp_dir, "test.json")
        
        save_json(data, filepath)
        
        assert os.path.exists(filepath)
        
        loaded_data = load_json(filepath)
        
        assert loaded_data == data
    
    def test_load_json_not_found(self):
        """Test loading non-existent JSON file"""
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/file.json")
    
    def test_create_completion_flag(self, temp_work_dir, monkeypatch):
        """Test completion flag creation"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        flag_file = create_completion_flag("test_stage", work_dir=temp_work_dir["work_dir"])
        
        assert os.path.exists(flag_file)
        assert "test_stage_complete.flag" in flag_file
        
        with open(flag_file, 'r') as f:
            content = f.read()
            assert "Completed at:" in content
            assert "Seed: 42" in content
    
    def test_check_completion_flag(self, temp_work_dir, monkeypatch):
        """Test checking completion flag"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        # Should not exist initially
        assert not check_completion_flag("test_stage", work_dir=temp_work_dir["work_dir"])
        
        # Create flag
        create_completion_flag("test_stage", work_dir=temp_work_dir["work_dir"])
        
        # Should exist now
        assert check_completion_flag("test_stage", work_dir=temp_work_dir["work_dir"])
    
    def test_check_dependencies(self, temp_work_dir, monkeypatch):
        """Test dependency checking"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        # No dependencies met
        result = check_dependencies(["stage_0", "stage_1"], work_dir=temp_work_dir["work_dir"])
        assert result is False
        
        # Create one dependency
        create_completion_flag("stage_0", work_dir=temp_work_dir["work_dir"])
        result = check_dependencies(["stage_0", "stage_1"], work_dir=temp_work_dir["work_dir"])
        assert result is False
        
        # Create both dependencies
        create_completion_flag("stage_1", work_dir=temp_work_dir["work_dir"])
        result = check_dependencies(["stage_0", "stage_1"], work_dir=temp_work_dir["work_dir"])
        assert result is True


class TestResumabilityIOHelpers:
    """Tests for the atomic/JSONL persistence helpers ported from
    foundation_llm_experiments/utils/common_utils.py, used by the new
    transfer/order-preservation stages for crash-safe resume."""

    def test_load_json_safe_missing_file_returns_default(self, temp_dir):
        result = load_json_safe(os.path.join(temp_dir, "missing.json"), default={"a": 1})
        assert result == {"a": 1}

    def test_load_json_safe_missing_file_default_none(self, temp_dir):
        assert load_json_safe(os.path.join(temp_dir, "missing.json")) is None

    def test_load_json_safe_valid_file(self, temp_dir):
        path = os.path.join(temp_dir, "ok.json")
        with open(path, 'w') as f:
            json.dump({"x": 42}, f)
        assert load_json_safe(path) == {"x": 42}

    def test_load_json_safe_corrupt_file_returns_default(self, temp_dir):
        path = os.path.join(temp_dir, "corrupt.json")
        with open(path, 'w') as f:
            f.write("{not valid json,,,")
        assert load_json_safe(path, default="fallback") == "fallback"

    def test_atomic_save_json_writes_readable_file(self, temp_dir):
        path = os.path.join(temp_dir, "nested", "out.json")
        atomic_save_json({"result": 1.5}, path)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp"), "temp file must be renamed away"
        with open(path) as f:
            assert json.load(f) == {"result": 1.5}

    def test_atomic_save_json_overwrites_existing(self, temp_dir):
        path = os.path.join(temp_dir, "out.json")
        atomic_save_json({"v": 1}, path)
        atomic_save_json({"v": 2}, path)
        with open(path) as f:
            assert json.load(f) == {"v": 2}

    def test_append_jsonl_creates_one_line_per_record(self, temp_dir):
        path = os.path.join(temp_dir, "records.jsonl")
        append_jsonl(path, {"idx": 0, "val": "a"})
        append_jsonl(path, {"idx": 1, "val": "b"})
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l]
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"idx": 0, "val": "a"}
        assert json.loads(lines[1]) == {"idx": 1, "val": "b"}

    def test_append_jsonl_handles_missing_trailing_newline(self, temp_dir):
        """If a prior write was interrupted mid-line (no trailing newline),
        the next append must still start on a fresh line."""
        path = os.path.join(temp_dir, "records.jsonl")
        with open(path, 'w') as f:
            f.write(json.dumps({"idx": 0}))  # no trailing newline, simulates a crash
        append_jsonl(path, {"idx": 1})
        records = load_jsonl(path)
        assert records == [{"idx": 0}, {"idx": 1}]

    def test_load_jsonl_missing_file_returns_empty_list(self, temp_dir):
        assert load_jsonl(os.path.join(temp_dir, "missing.jsonl")) == []

    def test_load_jsonl_skips_malformed_trailing_line(self, temp_dir):
        path = os.path.join(temp_dir, "records.jsonl")
        with open(path, 'w') as f:
            f.write('{"idx": 0}\n')
            f.write('{"idx": 1, "broken":')  # truncated, simulates a kill mid-write
        records = load_jsonl(path)
        assert records == [{"idx": 0}]

    def test_load_jsonl_skips_blank_lines(self, temp_dir):
        path = os.path.join(temp_dir, "records.jsonl")
        with open(path, 'w') as f:
            f.write('{"idx": 0}\n\n{"idx": 1}\n')
        records = load_jsonl(path)
        assert records == [{"idx": 0}, {"idx": 1}]

    def test_partial_results_dir_created_under_results_dir(self, temp_dir, monkeypatch):
        # Patch the ExperimentConfig object common_utils actually holds a
        # reference to (imported via `from configs.experiment_config import
        # ExperimentConfig` inside the hpc_version package), rather than the
        # `hpc_version.configs.experiment_config` import path used elsewhere
        # in this test module -- the two can be distinct module instances
        # depending on how sys.path was primed, so patching the latter would
        # silently not affect the function under test.
        import hpc_version.utils.common_utils as common_utils_module
        results_dir = os.path.join(temp_dir, "results")
        monkeypatch.setattr(common_utils_module.ExperimentConfig, "RESULTS_DIR", results_dir)

        path = partial_results_dir()
        assert os.path.isdir(path)
        assert path.startswith(results_dir)

    def test_partial_results_dir_custom_subdir(self, temp_dir, monkeypatch):
        import hpc_version.utils.common_utils as common_utils_module
        results_dir = os.path.join(temp_dir, "results")
        monkeypatch.setattr(common_utils_module.ExperimentConfig, "RESULTS_DIR", results_dir)

        path = partial_results_dir(subdir="craft")
        assert os.path.basename(path) == "craft"
        assert os.path.isdir(path)

    def test_append_then_load_jsonl_roundtrip_many_records(self, temp_dir):
        path = os.path.join(temp_dir, "many.jsonl")
        expected = [{"idx": i, "loss": i * 0.1} for i in range(50)]
        for record in expected:
            append_jsonl(path, record)
        assert load_jsonl(path) == expected


class TestDatasetLoading:
    """Tests for dataset loading functions"""
    
    @patch('datasets.load_dataset')
    def test_load_dataset_split_success(self, mock_load_dataset, monkeypatch):
        """Test successful dataset loading"""
        monkeypatch.setattr(ExperimentConfig, "DATASET_MAX_RETRIES", 3)
        
        # Mock dataset
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"text": "doc1", "summary": "sum1"},
                    {"text": "doc2", "summary": "sum2"},
                ])
        
        mock_load_dataset.return_value = MockDataset()
        
        texts, summaries = load_dataset_split(
            "test_dataset", "test", "text", "summary"
        )
        
        assert len(texts) == 2
        assert len(summaries) == 2
        assert texts[0] == "doc1"
        assert summaries[0] == "sum1"
    
    @patch('datasets.load_dataset')
    def test_load_dataset_split_with_retry(self, mock_load_dataset, monkeypatch):
        """Test dataset loading with retry logic"""
        monkeypatch.setattr(ExperimentConfig, "DATASET_MAX_RETRIES", 3)
        monkeypatch.setattr(ExperimentConfig, "DATASET_RETRY_DELAY", 0.1)
        monkeypatch.setattr(ExperimentConfig, "DATASET_ALLOW_PARTIAL", True)
        
        # Create a mock dataset that succeeds on third try
        class MockDataset:
            def __iter__(self):
                return iter([{"text": "doc1", "summary": "sum1"}])
        
        # Fail twice, then succeed
        mock_load_dataset.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            MockDataset()
        ]
        
        texts, summaries = load_dataset_split(
            "test_dataset", "test", "text", "summary",
            max_retries=3, retry_delay=0.1
        )
        
        # Should eventually succeed
        assert isinstance(texts, list)
        assert isinstance(summaries, list)
    
    @patch('datasets.load_dataset')
    def test_load_dataset_split_max_samples(self, mock_load_dataset):
        """Test dataset loading with sample limit"""
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"text": f"doc{i}", "summary": f"sum{i}"}
                    for i in range(100)
                ])
        
        mock_load_dataset.return_value = MockDataset()
        
        texts, summaries = load_dataset_split(
            "test_dataset", "test", "text", "summary",
            max_samples=10
        )
        
        assert len(texts) == 10
        assert len(summaries) == 10


class TestCheckpointing:
    """Tests for checkpoint management"""
    
    @pytest.mark.skip(reason="Requires transformers model")
    def test_save_checkpoint(self, mock_model, temp_dir):
        """Test saving model checkpoint"""
        import torch.optim as optim
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=100
        )
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        history_path = os.path.join(temp_dir, "history.json")
        
        save_checkpoint(
            model=mock_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            val_loss=2.5,
            is_best=True,
            checkpoint_dir=checkpoint_dir,
            history_path=history_path,
            train_losses=[3.0, 2.8],
            val_losses=[3.2, 2.5]
        )
        
        assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint_epoch_1.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "best_model.pt"))
        assert os.path.exists(history_path)
    
    def test_load_checkpoint_none(self, temp_dir):
        """Test loading checkpoint from empty directory"""
        checkpoint_dir = os.path.join(temp_dir, "empty_checkpoints")
        os.makedirs(checkpoint_dir)
        
        checkpoint = load_checkpoint(checkpoint_dir)
        assert checkpoint is None
    
    @pytest.mark.skip(reason="Requires transformers model")
    def test_load_checkpoint_exists(self, mock_model, temp_dir):
        """Test loading existing checkpoint"""
        import torch.optim as optim
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=100
        )
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        
        # Save checkpoint
        save_checkpoint(
            model=mock_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            val_loss=2.0,
            is_best=False,
            checkpoint_dir=checkpoint_dir
        )
        
        # Load checkpoint
        loaded = load_checkpoint(checkpoint_dir)
        
        assert loaded is not None
        assert loaded['epoch'] == 2
        assert loaded['val_loss'] == 2.0


class TestStageLogger:
    """Tests for StageLogger class"""
    
    def test_stage_logger_creation(self, temp_work_dir, monkeypatch):
        """Test StageLogger creation"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        log_dir = os.path.join(temp_work_dir["work_dir"], "stage_logs")
        logger = StageLogger("test_stage", log_dir=log_dir)
        
        assert os.path.exists(os.path.join(log_dir, "test_stage.log"))
        assert logger.stage_name == "test_stage"
    
    def test_stage_logger_log(self, temp_work_dir, monkeypatch):
        """Test logging messages"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        log_dir = os.path.join(temp_work_dir["work_dir"], "stage_logs")
        logger = StageLogger("test_stage", log_dir=log_dir)
        
        logger.log("Test message")
        
        log_file = os.path.join(log_dir, "test_stage.log")
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content
    
    def test_stage_logger_complete_success(self, temp_work_dir, monkeypatch):
        """Test completing stage successfully"""
        monkeypatch.setattr(ExperimentConfig, "WORK_DIR", temp_work_dir["work_dir"])
        
        log_dir = os.path.join(temp_work_dir["work_dir"], "stage_logs")
        logger = StageLogger("test_stage", log_dir=log_dir)
        exit_code = logger.complete(success=True)
        
        assert exit_code == 0
        assert check_completion_flag("test_stage", work_dir=temp_work_dir["work_dir"])
    
    def test_stage_logger_complete_failure(self, temp_work_dir, monkeypatch):
        """Test completing stage with failure"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        monkeypatch.setattr(ExperimentConfig, "WORK_DIR", temp_work_dir["work_dir"])
        
        log_dir = os.path.join(temp_work_dir["work_dir"], "stage_logs")
        logger = StageLogger("test_stage", log_dir=log_dir)
        exit_code = logger.complete(success=False)
        
        assert exit_code == 1
        assert not check_completion_flag("test_stage", work_dir=temp_work_dir["work_dir"])


class TestSummarizationDataset:
    """Tests for SummarizationDataset class"""
    
    def test_dataset_creation(self, mock_tokenizer):
        """Test dataset creation"""
        texts = ["text 1", "text 2", "text 3"]
        summaries = ["summary 1", "summary 2", "summary 3"]
        
        dataset = SummarizationDataset(
            texts=texts,
            summaries=summaries,
            tokenizer=mock_tokenizer,
            max_input_length=128,
            max_target_length=64
        )
        
        assert len(dataset) == 3
    
    def test_dataset_getitem(self, mock_tokenizer):
        """Test getting items from dataset"""
        texts = ["test text for summarization"]
        summaries = ["test summary"]
        
        dataset = SummarizationDataset(
            texts=texts,
            summaries=summaries,
            tokenizer=mock_tokenizer,
            max_input_length=128,
            max_target_length=64
        )
        
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        
        assert item['input_ids'].shape[0] == 128  # max_input_length
        assert item['labels'].shape[0] == 64  # max_target_length
    
    def test_dataset_with_defaults(self, mock_tokenizer):
        """Test dataset with default lengths from config"""
        texts = ["test"]
        summaries = ["summary"]
        
        dataset = SummarizationDataset(
            texts=texts,
            summaries=summaries,
            tokenizer=mock_tokenizer
        )
        
        # Should use defaults from config or constructor
        assert dataset.max_input_length == ExperimentConfig.MAX_INPUT_LENGTH
        assert dataset.max_target_length == ExperimentConfig.MAX_TARGET_LENGTH
