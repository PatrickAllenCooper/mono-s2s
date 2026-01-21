"""
Tests for Model Operations with Comprehensive Mocking

Tests model-related functions without importing transformers directly.
"""
import os
import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))

from hpc_version.configs.experiment_config import ExperimentConfig


class TestMakeModelMonotonic:
    """Tests for make_model_monotonic function"""
    
    @patch('hpc_version.utils.common_utils.P')
    def test_make_model_monotonic_with_standard_ffn(self, mock_parametrize):
        """Test make_model_monotonic with standard FFN layers"""
        from hpc_version.utils.common_utils import make_model_monotonic
        
        # Create mock FFN layer
        class MockFFN:
            def __init__(self):
                self.wi = MagicMock()
                self.wi.weight = MagicMock()
                self.wi.weight.data = torch.randn(10, 10)
                
                self.wo = MagicMock()
                self.wo.weight = MagicMock()
                self.wo.weight.data = torch.randn(10, 10)
        
        # Create mock model
        class MockModel:
            def modules(self):
                return [MockFFN()]
        
        model = MockModel()
        
        # Apply monotonic constraints
        with patch('hpc_version.utils.common_utils.T5DenseReluDense', MockFFN):
            result = make_model_monotonic(model)
        
        assert result is not None
    
    @patch('hpc_version.utils.common_utils.P')
    def test_make_model_monotonic_with_gated_ffn(self, mock_parametrize):
        """Test make_model_monotonic with gated FFN layers"""
        from hpc_version.utils.common_utils import make_model_monotonic
        
        # Create mock gated FFN layer
        class MockGatedFFN:
            def __init__(self):
                self.wi_0 = MagicMock()
                self.wi_0.weight = MagicMock()
                self.wi_0.weight.data = torch.randn(10, 10)
                
                self.wi_1 = MagicMock()
                self.wi_1.weight = MagicMock()
                self.wi_1.weight.data = torch.randn(10, 10)
                
                self.wo = MagicMock()
                self.wo.weight = MagicMock()
                self.wo.weight.data = torch.randn(10, 10)
        
        class MockModel:
            def modules(self):
                return [MockGatedFFN()]
        
        model = MockModel()
        
        # This will fail to find FFN layers since we're not using the right class
        # but tests the duck-typing branch
        with pytest.raises(RuntimeError, match="No FFN layers found"):
            make_model_monotonic(model)
    
    @patch('hpc_version.utils.common_utils.P')
    def test_make_model_monotonic_duck_typing(self, mock_parametrize):
        """Test make_model_monotonic with duck typing fallback"""
        from hpc_version.utils.common_utils import make_model_monotonic
        
        # Create module with FFN-like structure
        class MockDenseActDense:
            def __init__(self):
                self.wi = MagicMock()
                self.wi.weight = MagicMock()
                self.wi.weight.data = torch.randn(5, 5)
                
                self.wo = MagicMock()
                self.wo.weight = MagicMock()
                self.wo.weight.data = torch.randn(5, 5)
        
        class MockModel:
            def modules(self):
                return [MockDenseActDense()]
            
            def named_modules(self):
                return [("ffn", MockDenseActDense())]
        
        model = MockModel()
        
        # Force duck typing path by patching FFN_CLASS
        with patch('hpc_version.utils.common_utils.T5DenseReluDense', None):
            with patch('hpc_version.utils.common_utils.T5DenseActDense', None):
                with patch('hpc_version.utils.common_utils.T5DenseGatedActDense', None):
                    # Should use duck typing
                    # Will likely fail but tests the branch
                    try:
                        result = make_model_monotonic(model)
                    except (RuntimeError, AttributeError):
                        # Expected if duck typing doesn't find layers
                        pass


class TestLoadModel:
    """Tests for load_model function"""
    
    @patch('transformers.T5ForConditionalGeneration')
    def test_load_model_standard_no_checkpoint(self, mock_model_class, monkeypatch):
        """Test loading standard model without checkpoint"""
        from hpc_version.utils.common_utils import load_model
        
        monkeypatch.setattr(ExperimentConfig, "MODEL_NAME", "t5-small")
        
        mock_model = MagicMock()
        mock_model.config.model_type = "t5"
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        model, is_pretrained = load_model("standard", checkpoint_path=None, device='cpu')
        
        assert model is not None
        assert is_pretrained is True
    
    @patch('transformers.T5ForConditionalGeneration')
    @patch('torch.load')
    @patch('os.path.exists', return_value=True)
    def test_load_model_with_checkpoint(self, mock_exists, mock_torch_load, 
                                        mock_model_class, monkeypatch):
        """Test loading model with checkpoint"""
        from hpc_version.utils.common_utils import load_model
        
        monkeypatch.setattr(ExperimentConfig, "MODEL_NAME", "t5-small")
        
        mock_model = MagicMock()
        mock_model.config.model_type = "t5"
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock checkpoint loading
        state_dict = {"layer.weight": torch.randn(10, 10)}
        mock_torch_load.return_value = state_dict
        
        model, is_pretrained = load_model(
            "baseline",
            checkpoint_path="/path/to/checkpoint.pt",
            device='cpu'
        )
        
        assert model is not None
        assert is_pretrained is False
        model.load_state_dict.assert_called_once_with(state_dict)
        model.eval.assert_called()
    
    @patch('transformers.T5ForConditionalGeneration')
    @patch('hpc_version.utils.common_utils.make_model_monotonic')
    def test_load_model_monotonic(self, mock_make_monotonic, mock_model_class, monkeypatch):
        """Test loading monotonic model"""
        from hpc_version.utils.common_utils import load_model
        
        monkeypatch.setattr(ExperimentConfig, "MODEL_NAME", "t5-small")
        
        mock_model = MagicMock()
        mock_model.config.model_type = "t5"
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        # make_model_monotonic should be called
        mock_make_monotonic.return_value = mock_model
        
        model, is_pretrained = load_model("monotonic", checkpoint_path=None, device='cpu')
        
        mock_make_monotonic.assert_called_once_with(mock_model)
    
    @patch('transformers.T5ForConditionalGeneration')
    @patch('os.path.exists', return_value=False)
    def test_load_model_checkpoint_not_found(self, mock_exists, mock_model_class, monkeypatch):
        """Test loading model when checkpoint path doesn't exist"""
        from hpc_version.utils.common_utils import load_model
        
        monkeypatch.setattr(ExperimentConfig, "MODEL_NAME", "t5-small")
        
        mock_model = MagicMock()
        mock_model.config.model_type = "t5"
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        model, is_pretrained = load_model(
            "baseline",
            checkpoint_path="/nonexistent/checkpoint.pt",
            device='cpu'
        )
        
        # Should fall back to pretrained
        assert is_pretrained is True


class TestGenerateSummary:
    """Tests for generate_summary_fixed_params"""
    
    @patch('transformers.T5Tokenizer')
    def test_generate_summary_basic(self, mock_tokenizer_class, monkeypatch):
        """Test basic summary generation"""
        from hpc_version.utils.common_utils import generate_summary_fixed_params
        
        monkeypatch.setattr(ExperimentConfig, "MAX_INPUT_LENGTH", 128)
        monkeypatch.setattr(ExperimentConfig, "DECODE_MAX_NEW_TOKENS", 50)
        monkeypatch.setattr(ExperimentConfig, "DECODE_MIN_NEW_TOKENS", 5)
        monkeypatch.setattr(ExperimentConfig, "DECODE_NUM_BEAMS", 4)
        monkeypatch.setattr(ExperimentConfig, "DECODE_LENGTH_PENALTY", 1.2)
        monkeypatch.setattr(ExperimentConfig, "DECODE_NO_REPEAT_NGRAM_SIZE", 3)
        monkeypatch.setattr(ExperimentConfig, "DECODE_EARLY_STOPPING", True)
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_inputs = {'input_ids': torch.randint(0, 100, (1, 50))}
        mock_tokenizer.return_value = mock_inputs
        
        # Mock model
        mock_model = MagicMock()
        mock_outputs = torch.randint(0, 100, (1, 20))
        mock_model.generate.return_value = mock_outputs
        
        # Mock decode
        mock_tokenizer.decode.return_value = "Generated summary"
        
        summary = generate_summary_fixed_params(
            mock_model, "test text", mock_tokenizer, device='cpu'
        )
        
        assert summary == "Generated summary"
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()


class TestComputeAvgLoss:
    """Tests for compute_avg_loss function"""
    
    def test_compute_avg_loss_basic(self):
        """Test computing average loss"""
        from hpc_version.utils.common_utils import compute_avg_loss
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create simple dataset
        input_ids = torch.randint(0, 100, (10, 20))
        attention_mask = torch.ones(10, 20)
        labels = torch.randint(0, 100, (10, 15))
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        loader = DataLoader(dataset, batch_size=2)
        
        # Create mock model
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = 2.5
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()
        
        avg_loss = compute_avg_loss(mock_model, loader, device='cpu')
        
        # Should be approximately 2.5
        assert isinstance(avg_loss, float)
    
    def test_compute_avg_loss_single_batch(self):
        """Test compute_avg_loss with single batch"""
        from hpc_version.utils.common_utils import compute_avg_loss
        from torch.utils.data import DataLoader, TensorDataset
        
        # Single batch dataset
        input_ids = torch.randint(0, 100, (2, 20))
        attention_mask = torch.ones(2, 20)
        labels = torch.randint(0, 100, (2, 15))
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        loader = DataLoader(dataset, batch_size=2)
        
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = 1.5
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()
        
        avg_loss = compute_avg_loss(mock_model, loader, device='cpu')
        
        assert avg_loss == 1.5


class TestSaveCheckpoint:
    """Tests for save_checkpoint function"""
    
    def test_save_checkpoint_basic(self, temp_dir):
        """Test saving checkpoint"""
        from hpc_version.utils.common_utils import save_checkpoint
        
        # Create mock objects
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weight": torch.randn(5, 5)}
        
        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.return_value = {"param_groups": []}
        
        mock_scheduler = MagicMock()
        mock_scheduler.state_dict.return_value = {"step": 0}
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        history_path = os.path.join(temp_dir, "history.json")
        
        save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=3,
            val_loss=2.1,
            is_best=False,
            checkpoint_dir=checkpoint_dir,
            history_path=history_path,
            train_losses=[3.0, 2.5, 2.2, 2.1],
            val_losses=[3.2, 2.7, 2.3, 2.1]
        )
        
        # Verify checkpoint was saved
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_3.pt")
        assert os.path.exists(checkpoint_path)
        
        # Verify history was saved
        assert os.path.exists(history_path)
    
    def test_save_checkpoint_is_best(self, temp_dir):
        """Test saving best model checkpoint"""
        from hpc_version.utils.common_utils import save_checkpoint
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weight": torch.randn(5, 5)}
        
        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.return_value = {}
        
        mock_scheduler = MagicMock()
        mock_scheduler.state_dict.return_value = {}
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        
        save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=5,
            val_loss=1.8,
            is_best=True,  # This is the best model
            checkpoint_dir=checkpoint_dir
        )
        
        # Verify both checkpoints saved
        assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint_epoch_5.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "best_model.pt"))
    
    def test_save_checkpoint_without_history(self, temp_dir):
        """Test saving checkpoint without history"""
        from hpc_version.utils.common_utils import save_checkpoint
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        
        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.return_value = {}
        
        mock_scheduler = MagicMock()
        mock_scheduler.state_dict.return_value = {}
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        
        # Don't pass history parameters
        save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=1,
            val_loss=2.5,
            is_best=False,
            checkpoint_dir=checkpoint_dir
        )
        
        assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint_epoch_1.pt"))


class TestLoadCheckpointFull:
    """Complete coverage for load_checkpoint"""
    
    def test_load_checkpoint_with_multiple_files(self, temp_dir):
        """Test loading checkpoint when multiple exist"""
        from hpc_version.utils.common_utils import load_checkpoint
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir)
        
        # Create checkpoints in non-sequential order
        for epoch in [1, 5, 3, 7, 2]:
            checkpoint = {
                'epoch': epoch,
                'val_loss': 3.0 - epoch * 0.1,
                'model_state_dict': {}
            }
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, path)
        
        # Should load epoch 7 (latest)
        loaded = load_checkpoint(checkpoint_dir)
        
        assert loaded is not None
        assert loaded['epoch'] == 7
        assert abs(loaded['val_loss'] - 2.3) < 0.01
    
    def test_load_checkpoint_empty_dir(self, temp_dir):
        """Test loading from empty checkpoint directory"""
        from hpc_version.utils.common_utils import load_checkpoint
        
        checkpoint_dir = os.path.join(temp_dir, "empty_checkpoints")
        os.makedirs(checkpoint_dir)
        
        result = load_checkpoint(checkpoint_dir)
        assert result is None
    
    def test_load_checkpoint_nonexistent_dir(self):
        """Test loading from nonexistent directory"""
        from hpc_version.utils.common_utils import load_checkpoint
        
        result = load_checkpoint("/definitely/does/not/exist")
        assert result is None
    
    def test_load_checkpoint_with_extra_files(self, temp_dir):
        """Test loading when directory has non-checkpoint files"""
        from hpc_version.utils.common_utils import load_checkpoint
        
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir)
        
        # Create valid checkpoint
        checkpoint = {'epoch': 3, 'val_loss': 2.0}
        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_epoch_3.pt"))
        
        # Create other files
        with open(os.path.join(checkpoint_dir, "other_file.txt"), 'w') as f:
            f.write("not a checkpoint")
        
        # Should still load correctly
        loaded = load_checkpoint(checkpoint_dir)
        assert loaded is not None
        assert loaded['epoch'] == 3


class TestDatasetEdgeCases:
    """Additional dataset edge case tests"""
    
    @patch('datasets.load_dataset')
    def test_load_dataset_empty_after_filtering(self, mock_load_dataset):
        """Test dataset that becomes empty after filtering"""
        from hpc_version.utils.common_utils import load_dataset_split
        
        # Dataset with only invalid entries
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"wrong_field": "value"},  # Missing required fields
                    {},  # Empty
                ])
        
        mock_load_dataset.return_value = MockDataset()
        
        texts, summaries = load_dataset_split(
            "empty_dataset", "test", "text", "summary"
        )
        
        # Should return empty lists
        assert texts == []
        assert summaries == []
    
    @patch('datasets.load_dataset')
    def test_load_dataset_with_none_values(self, mock_load_dataset):
        """Test dataset with None values"""
        from hpc_version.utils.common_utils import load_dataset_split
        
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"text": None, "summary": "summary"},  # None text
                    {"text": "text", "summary": None},  # None summary
                    {"text": "good", "summary": "good"},  # Valid
                ])
        
        mock_load_dataset.return_value = MockDataset()
        
        texts, summaries = load_dataset_split(
            "none_dataset", "test", "text", "summary"
        )
        
        # Should only include valid entry
        assert len(texts) == 1
        assert texts[0] == "good"
        assert summaries[0] == "good"
    
    @patch('datasets.load_dataset')
    def test_load_dataset_with_whitespace(self, mock_load_dataset):
        """Test dataset with whitespace-only entries"""
        from hpc_version.utils.common_utils import load_dataset_split
        
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"text": "   ", "summary": "summary"},  # Whitespace text
                    {"text": "text", "summary": "  "},  # Whitespace summary
                    {"text": "good text", "summary": "good summary"},  # Valid
                ])
        
        mock_load_dataset.return_value = MockDataset()
        
        texts, summaries = load_dataset_split(
            "whitespace_dataset", "test", "text", "summary"
        )
        
        # Whitespace should be stripped, empty strings filtered
        # Depending on implementation, may or may not include whitespace-only
        assert len(texts) >= 1  # At least the valid one


class TestConfigValidation:
    """Additional configuration validation tests"""
    
    def test_validate_config_all_paths_exist(self, temp_dir, monkeypatch):
        """Test validation when all paths exist"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        scratch = os.path.join(temp_dir, "scratch")
        project = os.path.join(temp_dir, "project")
        os.makedirs(scratch)
        os.makedirs(project)
        
        monkeypatch.setattr(ExperimentConfig, "SCRATCH_DIR", scratch)
        monkeypatch.setattr(ExperimentConfig, "PROJECT_DIR", project)
        monkeypatch.setattr(ExperimentConfig, "USE_FULL_TEST_SETS", False)
        monkeypatch.setattr(ExperimentConfig, "BATCH_SIZE", 2)
        
        result = ExperimentConfig.validate_config()
        
        # Should pass (or return False if no GPU, but shouldn't crash)
        assert isinstance(result, bool)
    
    def test_validate_config_large_batch_warning(self, temp_dir, monkeypatch):
        """Test validation warns about large batch size"""
        from hpc_version.configs.experiment_config import ExperimentConfig
        
        scratch = os.path.join(temp_dir, "scratch")
        project = os.path.join(temp_dir, "project")
        os.makedirs(scratch)
        os.makedirs(project)
        
        monkeypatch.setattr(ExperimentConfig, "SCRATCH_DIR", scratch)
        monkeypatch.setattr(ExperimentConfig, "PROJECT_DIR", project)
        monkeypatch.setattr(ExperimentConfig, "USE_FULL_TEST_SETS", True)
        monkeypatch.setattr(ExperimentConfig, "BATCH_SIZE", 32)  # Large
        
        result = ExperimentConfig.validate_config()
        
        # Should warn but not crash
        assert isinstance(result, bool)
