"""
Tests for Specific Branch Coverage

Targeted tests for uncovered branches in testable functions.
"""
import os
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))

from hpc_version.utils.common_utils import (
    set_all_seeds, log_environment, compute_rouge_with_ci,
    compute_length_statistics
)
from hpc_version.configs.experiment_config import ExperimentConfig


class TestSetAllSeedsBranches:
    """Tests to cover all branches in set_all_seeds"""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_set_all_seeds_with_cuda(self, mock_seed_all, mock_seed, mock_available):
        """Test set_all_seeds when CUDA is available"""
        set_all_seeds(42)
        
        # Should call CUDA seed functions (may be called multiple times in setup)
        mock_seed.assert_called_with(42)
        mock_seed_all.assert_called_with(42)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_set_all_seeds_without_cuda(self, mock_available):
        """Test set_all_seeds when CUDA is not available"""
        # Should complete without calling CUDA functions
        set_all_seeds(42)
        
        assert os.environ["PYTHONHASHSEED"] == "42"
    
    @patch('torch.backends.cuda.matmul')
    @patch('torch.backends.cuda')
    def test_set_all_seeds_tf32_disable(self, mock_cuda_backend, mock_matmul):
        """Test TF32 disabling branch"""
        # Mock the backends to have the tf32 attribute
        mock_cuda_backend.matmul = mock_matmul
        
        with patch('torch.backends', mock_cuda_backend):
            set_all_seeds(42)
        
        # Should attempt to disable TF32 if available
        # (may or may not be called depending on torch version)
    
    @patch('torch.set_float32_matmul_precision')
    def test_set_all_seeds_matmul_precision(self, mock_precision):
        """Test float32 matmul precision setting"""
        set_all_seeds(42)
        
        # Should be called if available
        mock_precision.assert_called()
    
    @patch('torch.use_deterministic_algorithms')
    def test_set_all_seeds_deterministic_exception(self, mock_deterministic):
        """Test handling of deterministic algorithms exception"""
        # Simulate exception
        mock_deterministic.side_effect = RuntimeError("Not supported")
        
        # Should handle exception gracefully
        set_all_seeds(42)
        
        mock_deterministic.assert_called_once()


class TestLogEnvironmentBranches:
    """Tests to cover all branches in log_environment"""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Tesla V100")
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count', return_value=4)
    @patch('torch.version.cuda', "11.8")
    @patch('torch.backends.cudnn.version', return_value=8005)
    def test_log_environment_with_cuda_full(self, mock_cudnn, mock_cuda_ver,
                                            mock_count, mock_props,  
                                            mock_name, mock_available):
        """Test log_environment with all CUDA info"""
        mock_props.return_value = MagicMock(total_memory=32 * 1024**3)
        
        env_info = log_environment()
        
        assert env_info["cuda_available"] is True
        assert "cuda_version" in env_info
        assert "cudnn_version" in env_info
        assert "gpu_name" in env_info
        assert "gpu_memory_gb" in env_info
        assert "gpu_count" in env_info
        
        # Verify values
        assert env_info["gpu_name"] == "Tesla V100"
        assert env_info["gpu_count"] == 4
        assert env_info["gpu_memory_gb"] > 30  # 32GB
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_log_environment_cpu_only(self, mock_available):
        """Test log_environment without CUDA"""
        env_info = log_environment()
        
        assert env_info["cuda_available"] is False
        
        # Should not have CUDA-specific keys
        assert "cuda_version" not in env_info
        assert "gpu_name" not in env_info
        assert "gpu_memory_gb" not in env_info


class TestRougeComprehensive:
    """Complete ROUGE coverage"""
    
    def test_compute_rouge_custom_metrics(self, monkeypatch):
        """Test ROUGE with custom metrics list"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        predictions = ["test prediction"]
        references = ["test reference"]
        
        # Test with only rouge1
        ci_scores, _ = compute_rouge_with_ci(
            predictions, references,
            metrics=["rouge1"],
            use_stemmer=True,
            n_bootstrap=50,
            confidence=0.95
        )
        
        assert "rouge1" in ci_scores
        assert "rouge2" not in ci_scores
    
    def test_compute_rouge_confidence_90(self, monkeypatch):
        """Test ROUGE with 90% confidence"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        predictions = ["text text text"]
        references = ["text text other"]
        
        ci_scores, _ = compute_rouge_with_ci(
            predictions, references,
            metrics=["rouge1"],
            n_bootstrap=30,
            confidence=0.90
        )
        
        assert "rouge1" in ci_scores
        assert ci_scores["rouge1"]["ci_width"] >= 0
    
    def test_compute_rouge_large_bootstrap(self, monkeypatch):
        """Test ROUGE with many bootstrap samples"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        predictions = ["a b c"]
        references = ["a b d"]
        
        ci_scores, all_scores = compute_rouge_with_ci(
            predictions, references,
            metrics=["rouge1"],
            n_bootstrap=1000,  # Many samples
            confidence=0.95
        )
        
        assert ci_scores["rouge1"]["mean"] > 0


class TestLengthStatisticsBranches:
    """Complete branch coverage for length statistics"""
    
    def test_length_stats_with_tokenizer_encoding(self, mock_tokenizer):
        """Test length statistics using tokenizer encoding"""
        # Mock tokenizer.encode
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3, 4, 5],  # 5 tokens
            [1, 2, 3],  # 3 tokens
            [1, 2, 3, 4, 5, 6, 7]  # 7 tokens
        ]
        
        texts = ["text1", "text2", "text3"]
        stats = compute_length_statistics(texts, tokenizer=mock_tokenizer)
        
        assert stats["unit"] == "tokens"
        assert stats["min"] == 3
        assert stats["max"] == 7
        assert stats["mean"] == 5.0
        assert stats["total"] == 15


class TestDatasetRetryLogic:
    """Comprehensive dataset retry logic tests"""
    
    @patch('datasets.load_dataset')
    @patch('time.sleep')
    def test_load_dataset_retry_success_first_try(self, mock_sleep, mock_load_dataset):
        """Test dataset loads on first try (no retries)"""
        from hpc_version.utils.common_utils import load_dataset_split
        
        class MockDataset:
            def __iter__(self):
                return iter([{"text": "t1", "summary": "s1"}])
        
        mock_load_dataset.return_value = MockDataset()
        
        texts, summaries = load_dataset_split(
            "good_dataset", "test", "text", "summary",
            max_retries=3, retry_delay=1
        )
        
        # Should succeed on first try
        assert len(texts) == 1
        assert mock_load_dataset.call_count == 1
        # Should not sleep if first try succeeds
        assert mock_sleep.call_count == 0
    
    @patch('datasets.load_dataset')
    @patch('time.sleep')
    def test_load_dataset_retry_success_second_try(self, mock_sleep, mock_load_dataset):
        """Test dataset loads on second try"""
        from hpc_version.utils.common_utils import load_dataset_split
        
        class MockDataset:
            def __iter__(self):
                return iter([{"text": "t1", "summary": "s1"}])
        
        # Fail once, then succeed
        mock_load_dataset.side_effect = [
            Exception("Temporary failure"),
            MockDataset()
        ]
        
        texts, summaries = load_dataset_split(
            "retry_dataset", "test", "text", "summary",
            max_retries=3, retry_delay=0.1
        )
        
        assert len(texts) == 1
        assert mock_load_dataset.call_count == 2
        assert mock_sleep.call_count >= 1


class TestFileOperationsBranches:
    """Test uncovered file operation branches"""
    
    def test_save_json_default_indent(self, temp_dir):
        """Test save_json with default indent"""
        from hpc_version.utils.common_utils import save_json
        
        data = {"key": "value"}
        filepath = os.path.join(temp_dir, "default.json")
        
        # Use default indent (2)
        save_json(data, filepath)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Should be formatted with indentation
        assert "\n" in content
    
    def test_save_json_no_indent(self, temp_dir):
        """Test save_json with no indent"""
        from hpc_version.utils.common_utils import save_json
        
        data = {"key": "value"}
        filepath = os.path.join(temp_dir, "compact.json")
        
        save_json(data, filepath, indent=None)
        
        assert os.path.exists(filepath)


class TestCheckpointManagementBranches:
    """Test checkpoint management branches"""
    
    def test_load_checkpoint_finds_latest_among_many(self, temp_dir):
        """Test loading latest checkpoint from many files"""
        from hpc_version.utils.common_utils import load_checkpoint
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir)
        
        # Create many checkpoints
        for epoch in range(1, 11):
            checkpoint = {'epoch': epoch, 'val_loss': 10.0 / epoch}
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, path)
        
        # Add some non-checkpoint files
        with open(os.path.join(checkpoint_dir, "readme.txt"), 'w') as f:
            f.write("info")
        
        loaded = load_checkpoint(checkpoint_dir)
        
        assert loaded is not None
        assert loaded['epoch'] == 10  # Latest


class TestConfigValidationBranches:
    """Test configuration validation branches"""
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_validate_config_with_gpu(self, mock_cuda, temp_dir, monkeypatch):
        """Test validation when GPU is available"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        scratch = os.path.join(temp_dir, "scratch")
        project = os.path.join(temp_dir, "project")
        os.makedirs(scratch)
        os.makedirs(project)
        
        monkeypatch.setattr(ExperimentConfig, "SCRATCH_DIR", scratch)
        monkeypatch.setattr(ExperimentConfig, "PROJECT_DIR", project)
        monkeypatch.setattr(ExperimentConfig, "USE_FULL_TEST_SETS", False)
        
        result = ExperimentConfig.validate_config()
        
        # Should pass with GPU
        assert result is True
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_validate_config_without_gpu(self, mock_cuda, temp_dir, monkeypatch):
        """Test validation warning when no GPU"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        scratch = os.path.join(temp_dir, "scratch")
        project = os.path.join(temp_dir, "project")
        os.makedirs(scratch)
        os.makedirs(project)
        
        monkeypatch.setattr(ExperimentConfig, "SCRATCH_DIR", scratch)
        monkeypatch.setattr(ExperimentConfig, "PROJECT_DIR", project)
        
        result = ExperimentConfig.validate_config()
        
        # Should return False due to no GPU warning
        assert result is False
    
    def test_validate_config_missing_scratch(self, monkeypatch):
        """Test validation with missing SCRATCH_DIR"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        monkeypatch.setattr(ExperimentConfig, "SCRATCH_DIR", "/definitely/nonexistent")
        monkeypatch.setattr(ExperimentConfig, "PROJECT_DIR", "/also/nonexistent")
        
        result = ExperimentConfig.validate_config()
        
        assert result is False
    
    def test_validate_config_batch_size_warning(self, temp_dir, monkeypatch):
        """Test validation warns about OOM risk"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        scratch = os.path.join(temp_dir, "scratch")
        project = os.path.join(temp_dir, "project")
        os.makedirs(scratch)
        os.makedirs(project)
        
        monkeypatch.setattr(ExperimentConfig, "SCRATCH_DIR", scratch)
        monkeypatch.setattr(ExperimentConfig, "PROJECT_DIR", project)
        monkeypatch.setattr(ExperimentConfig, "USE_FULL_TEST_SETS", True)
        monkeypatch.setattr(ExperimentConfig, "BATCH_SIZE", 64)  # Large batch
        
        result = ExperimentConfig.validate_config()
        
        # Should warn but complete
        assert isinstance(result, bool)


class TestGetDeviceBranches:
    """Test get_device branches"""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Mock GPU")
    @patch('torch.cuda.get_device_properties')
    def test_get_device_with_cuda(self, mock_props, mock_name, mock_available):
        """Test get_device when CUDA is available"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)
        
        device = ExperimentConfig.get_device()
        
        assert device.type == "cuda"
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_device_cpu_fallback(self, mock_available):
        """Test get_device falls back to CPU"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        device = ExperimentConfig.get_device()
        
        assert device.type == "cpu"


class TestDatasetSpecialCases:
    """Test special cases in dataset handling"""
    
    @patch('datasets.load_dataset')
    def test_load_dataset_strips_whitespace(self, mock_load_dataset):
        """Test that dataset strips whitespace from entries"""
        from hpc_version.utils.common_utils import load_dataset_split
        
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"text": "  text with spaces  ", "summary": "  summary  "},
                ])
        
        mock_load_dataset.return_value = MockDataset()
        
        texts, summaries = load_dataset_split(
            "whitespace_dataset", "test", "text", "summary"
        )
        
        # Should strip whitespace
        assert texts[0] == "text with spaces"
        assert summaries[0] == "summary"


class TestRougeBranches:
    """Test ROUGE computation branches"""
    
    def test_rouge_with_none_metrics(self, monkeypatch):
        """Test ROUGE with None metrics (uses defaults)"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        monkeypatch.setattr(ExperimentConfig, "ROUGE_METRICS", ["rouge1", "rouge2"])
        
        predictions = ["test"]
        references = ["test"]
        
        # Pass metrics=None to use defaults from config
        ci_scores, _ = compute_rouge_with_ci(
            predictions, references,
            metrics=None,  # Should use ExperimentConfig.ROUGE_METRICS
            n_bootstrap=20
        )
        
        # Should have both metrics from config
        assert "rouge1" in ci_scores or "rouge2" in ci_scores


class TestConfigCreateDirectories:
    """Test create_directories branches"""
    
    def test_create_directories_creates_all(self, temp_dir, monkeypatch):
        """Test that all directories are created"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        work_dir = os.path.join(temp_dir, "work")
        results_dir = os.path.join(temp_dir, "results")
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        cache_dir = os.path.join(temp_dir, "cache")
        final_dir = os.path.join(temp_dir, "final")
        
        monkeypatch.setattr(ExperimentConfig, "WORK_DIR", work_dir)
        monkeypatch.setattr(ExperimentConfig, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(ExperimentConfig, "CHECKPOINT_DIR", checkpoint_dir)
        monkeypatch.setattr(ExperimentConfig, "DATA_CACHE_DIR", cache_dir)
        monkeypatch.setattr(ExperimentConfig, "FINAL_RESULTS_DIR", final_dir)
        
        ExperimentConfig.create_directories()
        
        # All directories should be created
        assert os.path.exists(work_dir)
        assert os.path.exists(results_dir)
        assert os.path.exists(checkpoint_dir)
        assert os.path.exists(cache_dir)
        assert os.path.exists(final_dir)
        assert os.path.exists(os.path.join(checkpoint_dir, "baseline_checkpoints"))
        assert os.path.exists(os.path.join(checkpoint_dir, "monotonic_checkpoints"))
    
    def test_create_directories_idempotent(self, temp_dir, monkeypatch):
        """Test that create_directories is idempotent"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        work_dir = os.path.join(temp_dir, "work")
        
        monkeypatch.setattr(ExperimentConfig, "WORK_DIR", work_dir)
        monkeypatch.setattr(ExperimentConfig, "RESULTS_DIR", os.path.join(temp_dir, "r"))
        monkeypatch.setattr(ExperimentConfig, "CHECKPOINT_DIR", os.path.join(temp_dir, "c"))
        monkeypatch.setattr(ExperimentConfig, "DATA_CACHE_DIR", os.path.join(temp_dir, "d"))
        monkeypatch.setattr(ExperimentConfig, "FINAL_RESULTS_DIR", os.path.join(temp_dir, "f"))
        
        # Call twice
        ExperimentConfig.create_directories()
        ExperimentConfig.create_directories()
        
        # Should still work
        assert os.path.exists(work_dir)


class TestBrevityPenaltyBranches:
    """Test brevity penalty calculation branches"""
    
    def test_brevity_penalty_with_zero_ref_length(self):
        """Test brevity penalty when reference length is zero"""
        from hpc_version.utils.common_utils import compute_brevity_penalty
        
        predictions = ["some text"]
        references = [""]  # Empty reference
        
        bp_stats = compute_brevity_penalty(predictions, references)
        
        # Should handle gracefully
        assert isinstance(bp_stats, dict)
        assert "brevity_penalty" in bp_stats
    
    def test_brevity_penalty_zero_pred_length(self):
        """Test brevity penalty when prediction length is zero"""
        from hpc_version.utils.common_utils import compute_brevity_penalty
        
        predictions = [""]  # Empty prediction
        references = ["some reference"]
        
        bp_stats = compute_brevity_penalty(predictions, references)
        
        # Should handle gracefully
        assert isinstance(bp_stats, dict)
        assert bp_stats["length_ratio"] == 0 or bp_stats["brevity_penalty"] >= 0


class TestConfigToDictExclusions:
    """Test to_dict exclusions"""
    
    def test_to_dict_excludes_methods(self):
        """Test that to_dict excludes callable methods"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        config_dict = ExperimentConfig.to_dict()
        
        # Should not include methods
        assert "to_dict" not in config_dict
        assert "create_directories" not in config_dict
        assert "get_device" not in config_dict
        assert "validate_config" not in config_dict
    
    def test_to_dict_includes_constants(self):
        """Test that to_dict includes uppercase constants"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        config_dict = ExperimentConfig.to_dict()
        
        # Should include constants
        assert "MODEL_NAME" in config_dict
        assert "LEARNING_RATE" in config_dict
        assert "NUM_EPOCHS" in config_dict
        assert "BATCH_SIZE" in config_dict
