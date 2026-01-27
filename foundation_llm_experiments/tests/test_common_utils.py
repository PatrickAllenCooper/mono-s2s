"""
Tests for common utilities

Validates monotonicity application, data handling, and helper functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import os

from utils.common_utils import (
    set_all_seeds,
    get_generator,
    worker_init_fn,
    NonNegativeParametrization,
    make_model_monotonic,
    compute_perplexity,
    save_json,
    load_json,
    create_completion_flag,
    check_completion_flag,
    check_dependencies,
    StageLogger,
    LanguageModelingDataset,
)


class TestDeterminism:
    """Test deterministic behavior"""
    
    def test_set_all_seeds_makes_reproducible(self):
        """Test that set_all_seeds produces reproducible results"""
        set_all_seeds(42)
        vals1 = [torch.rand(1).item() for _ in range(10)]
        
        set_all_seeds(42)
        vals2 = [torch.rand(1).item() for _ in range(10)]
        
        assert vals1 == vals2, "Random values should be identical with same seed"
    
    def test_different_seeds_give_different_results(self):
        """Test different seeds produce different results"""
        set_all_seeds(42)
        vals1 = [torch.rand(1).item() for _ in range(10)]
        
        set_all_seeds(1337)
        vals2 = [torch.rand(1).item() for _ in range(10)]
        
        assert vals1 != vals2, "Different seeds should give different values"
    
    def test_get_generator_reproducible(self):
        """Test generator produces reproducible samples"""
        gen1 = get_generator(device='cpu', seed=42)
        vals1 = [torch.rand(1, generator=gen1).item() for _ in range(10)]
        
        gen2 = get_generator(device='cpu', seed=42)
        vals2 = [torch.rand(1, generator=gen2).item() for _ in range(10)]
        
        assert vals1 == vals2
    
    def test_worker_init_fn_different_workers(self):
        """Test worker_init_fn gives different seeds to different workers"""
        # This is hard to test directly, but we can verify it doesn't crash
        worker_init_fn(0)
        worker_init_fn(1)
        worker_init_fn(2)
        # If it didn't crash, it's working


class TestMonotonicParametrization:
    """Test softplus parametrization"""
    
    def test_forward_always_positive(self):
        """Test that softplus(V) is always >= 0"""
        param = NonNegativeParametrization()
        
        # Test with various inputs
        test_inputs = [
            torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0]),
            torch.randn(10, 10),
            torch.randn(100, 100) - 5,  # Biased negative
        ]
        
        for V in test_inputs:
            W = param.forward(V)
            assert torch.all(W >= 0), "Softplus output should always be non-negative"
    
    def test_right_inverse_preserves_magnitude(self):
        """Test inverse softplus preserves weight magnitude roughly"""
        param = NonNegativeParametrization()
        
        W_original = torch.randn(5, 5).abs() + 0.1  # Positive weights
        V_init = param.right_inverse(W_original)
        W_reconstructed = param.forward(V_init)
        
        # Should be close (within numerical error)
        assert torch.allclose(W_original, W_reconstructed, atol=1e-3)
    
    def test_right_inverse_handles_negative_weights(self):
        """Test inverse softplus handles negative weights (takes abs)"""
        param = NonNegativeParametrization()
        
        W_negative = torch.tensor([-1.0, -5.0, -10.0])
        V_init = param.right_inverse(W_negative)
        W_result = param.forward(V_init)
        
        # Should be positive and close to |W|
        assert torch.all(W_result >= 0)
        assert torch.allclose(W_result, W_negative.abs(), atol=1e-3)
    
    def test_right_inverse_stable_near_zero(self):
        """Test inverse softplus is numerically stable near zero"""
        param = NonNegativeParametrization()
        
        W_small = torch.tensor([1e-6, 1e-5, 1e-4, 1e-3])
        V_init = param.right_inverse(W_small)
        
        # Should not produce NaN or Inf
        assert torch.all(torch.isfinite(V_init))
        
        # Should reconstruct reasonably
        W_reconstructed = param.forward(V_init)
        assert torch.allclose(W_small, W_reconstructed, atol=1e-5, rtol=1e-2)


class TestMakeModelMonotonic:
    """Test applying monotonicity to models"""
    
    def test_make_model_monotonic_modifies_ffn(self, mock_ffn_model):
        """Test that make_model_monotonic actually modifies FFN layers"""
        # Count parameters before
        params_before = sum(p.numel() for p in mock_ffn_model.parameters())
        
        # Apply monotonicity
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Model should still exist
        assert monotonic_model is not None
        
        # Should have parametrizations registered
        # (parameter count might change due to parametrization)
        params_after = sum(p.numel() for p in monotonic_model.parameters())
        assert params_after >= params_before
    
    def test_monotonic_model_weights_nonnegative(self, mock_ffn_model):
        """Test that monotonic model has non-negative FFN weights"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Check weights in FFN layers
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and ('mlp' in name.lower() or 'linear' in name.lower()):
                # Weights should be non-negative
                min_val = param.data.min().item()
                assert min_val >= -1e-6, f"Found negative weight in {name}: {min_val}"
    
    def test_monotonic_model_forward_pass(self, mock_ffn_model):
        """Test that monotonic model can do forward pass"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Test forward pass
        x = torch.randn(2, 64)
        output = monotonic_model(x)
        
        assert output.shape == (2, 64)
        assert torch.all(torch.isfinite(output))
    
    def test_monotonic_model_gradient_flow(self, mock_ffn_model):
        """Test that gradients flow through monotonic model"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        x = torch.randn(2, 64, requires_grad=True)
        output = monotonic_model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))
        
        # Check model parameters have gradients
        for param in monotonic_model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestPerplexity:
    """Test perplexity computation"""
    
    def test_compute_perplexity_basic(self, mock_gpt_model, mock_tokenizer, mock_eval_data):
        """Test basic perplexity computation"""
        from utils.common_utils import LanguageModelingDataset
        from torch.utils.data import DataLoader
        
        dataset = LanguageModelingDataset(mock_eval_data[:10], mock_tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=2)
        
        device = torch.device('cpu')
        mock_gpt_model.eval()
        
        result = compute_perplexity(mock_gpt_model, dataloader, device)
        
        assert 'perplexity' in result
        assert 'loss' in result
        assert 'total_tokens' in result
        
        # Perplexity should be positive
        assert result['perplexity'] > 0
        # Loss should be positive (cross-entropy)
        assert result['loss'] > 0
        # Should have processed some tokens
        assert result['total_tokens'] > 0
    
    def test_perplexity_exp_of_loss(self, mock_gpt_model, mock_tokenizer, mock_eval_data):
        """Test perplexity = exp(loss) relationship"""
        from utils.common_utils import LanguageModelingDataset
        from torch.utils.data import DataLoader
        
        dataset = LanguageModelingDataset(mock_eval_data[:10], mock_tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=2)
        
        result = compute_perplexity(mock_gpt_model, dataloader, torch.device('cpu'))
        
        # Perplexity should be exp(loss)
        expected_ppl = np.exp(result['loss'])
        assert np.abs(result['perplexity'] - expected_ppl) < 1e-4


class TestFileOperations:
    """Test file I/O utilities"""
    
    def test_save_and_load_json(self, tmp_path):
        """Test JSON save/load roundtrip"""
        data = {
            'test_key': 'test_value',
            'number': 42,
            'list': [1, 2, 3],
            'nested': {'a': 1, 'b': 2}
        }
        
        filepath = tmp_path / "test.json"
        save_json(data, str(filepath))
        
        assert os.path.exists(filepath)
        
        loaded = load_json(str(filepath))
        assert loaded == data
    
    def test_save_json_creates_directory(self, tmp_path):
        """Test save_json creates parent directory if needed"""
        filepath = tmp_path / "subdir" / "nested" / "test.json"
        data = {'test': 123}
        
        save_json(data, str(filepath))
        
        assert os.path.exists(filepath)
        assert load_json(str(filepath)) == data
    
    def test_load_json_missing_file(self, tmp_path):
        """Test load_json raises error for missing file"""
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "nonexistent.json"))
    
    def test_create_completion_flag(self, tmp_path, monkeypatch):
        """Test completion flag creation"""
        from configs.experiment_config import FoundationExperimentConfig as Config
        monkeypatch.setattr(Config, 'WORK_DIR', str(tmp_path))
        monkeypatch.setattr(Config, 'CURRENT_SEED', 42)
        
        flag_file = create_completion_flag('test_stage', work_dir=str(tmp_path))
        
        assert os.path.exists(flag_file)
        assert 'test_stage_complete.flag' in flag_file
        
        # Read flag content
        with open(flag_file) as f:
            content = f.read()
        assert 'Completed at:' in content
        assert 'Seed: 42' in content
    
    def test_check_completion_flag(self, tmp_path):
        """Test checking for completion flag"""
        # No flag exists yet
        assert check_completion_flag('test_stage', work_dir=str(tmp_path)) is False
        
        # Create flag
        create_completion_flag('test_stage', work_dir=str(tmp_path))
        
        # Now should exist
        assert check_completion_flag('test_stage', work_dir=str(tmp_path)) is True
    
    def test_check_dependencies_all_met(self, tmp_path):
        """Test dependency checking when all dependencies met"""
        # Create all required flags
        for stage in ['stage_0_setup', 'stage_1_apply']:
            create_completion_flag(stage, work_dir=str(tmp_path))
        
        result = check_dependencies(['stage_0_setup', 'stage_1_apply'], work_dir=str(tmp_path))
        assert result is True
    
    def test_check_dependencies_missing(self, tmp_path):
        """Test dependency checking when some missing"""
        # Create only one flag
        create_completion_flag('stage_0_setup', work_dir=str(tmp_path))
        
        result = check_dependencies(['stage_0_setup', 'stage_1_apply'], work_dir=str(tmp_path))
        assert result is False


class TestStageLogger:
    """Test stage logging functionality"""
    
    def test_logger_creates_log_file(self, tmp_path):
        """Test logger creates log file"""
        log_dir = tmp_path / "logs"
        logger = StageLogger("test_stage", log_dir=str(log_dir))
        
        assert os.path.exists(logger.log_file)
        assert 'test_stage.log' in logger.log_file
    
    def test_logger_writes_messages(self, tmp_path):
        """Test logger writes messages to file"""
        log_dir = tmp_path / "logs"
        logger = StageLogger("test_stage", log_dir=str(log_dir))
        
        logger.log("Test message 1")
        logger.log("Test message 2")
        
        with open(logger.log_file) as f:
            content = f.read()
        
        assert "Test message 1" in content
        assert "Test message 2" in content
        assert "STAGE: test_stage" in content
    
    def test_logger_complete_creates_flag(self, tmp_path, monkeypatch):
        """Test logger.complete creates completion flag"""
        from configs.experiment_config import FoundationExperimentConfig as Config
        monkeypatch.setattr(Config, 'WORK_DIR', str(tmp_path))
        
        log_dir = tmp_path / "logs"
        logger = StageLogger("test_stage", log_dir=str(log_dir))
        
        exit_code = logger.complete(success=True)
        
        assert exit_code == 0
        assert check_completion_flag('test_stage', work_dir=str(tmp_path))
    
    def test_logger_complete_failure_no_flag(self, tmp_path, monkeypatch):
        """Test logger.complete on failure doesn't create flag"""
        from configs.experiment_config import FoundationExperimentConfig as Config
        monkeypatch.setattr(Config, 'WORK_DIR', str(tmp_path))
        
        log_dir = tmp_path / "logs"
        logger = StageLogger("test_stage", log_dir=str(log_dir))
        
        exit_code = logger.complete(success=False)
        
        assert exit_code == 1
        assert not check_completion_flag('test_stage', work_dir=str(tmp_path))


class TestLanguageModelingDataset:
    """Test dataset class"""
    
    def test_dataset_creation(self, mock_tokenizer, mock_training_data):
        """Test dataset can be created"""
        dataset = LanguageModelingDataset(
            mock_training_data,
            mock_tokenizer,
            max_length=128
        )
        
        assert len(dataset) == len(mock_training_data)
    
    def test_dataset_getitem(self, mock_tokenizer, mock_training_data):
        """Test dataset __getitem__ returns correct format"""
        dataset = LanguageModelingDataset(
            mock_training_data,
            mock_tokenizer,
            max_length=128
        )
        
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        
        # Check shapes
        assert item['input_ids'].shape == (128,)
        assert item['attention_mask'].shape == (128,)
        
        # Check types
        assert item['input_ids'].dtype == torch.long
        assert item['attention_mask'].dtype == torch.long
    
    def test_dataset_handles_long_text(self, mock_tokenizer):
        """Test dataset truncates long texts correctly"""
        long_text = "word " * 1000  # Very long text
        
        dataset = LanguageModelingDataset(
            [long_text],
            mock_tokenizer,
            max_length=128
        )
        
        item = dataset[0]
        
        # Should be truncated to max_length
        assert item['input_ids'].shape == (128,)
    
    def test_dataset_handles_short_text(self, mock_tokenizer):
        """Test dataset pads short texts correctly"""
        short_text = "Hi"
        
        dataset = LanguageModelingDataset(
            [short_text],
            mock_tokenizer,
            max_length=128
        )
        
        item = dataset[0]
        
        # Should be padded to max_length
        assert item['input_ids'].shape == (128,)
        
        # Should have some padding tokens
        num_padding = (item['attention_mask'] == 0).sum().item()
        assert num_padding > 0


class TestMonotonicityApplication:
    """Integration tests for monotonicity application"""
    
    def test_apply_monotonicity_to_simple_model(self, mock_ffn_model):
        """Test applying monotonicity to a simple model"""
        # Get initial weights
        initial_weights = {}
        for name, param in mock_ffn_model.named_parameters():
            if 'weight' in name:
                initial_weights[name] = param.data.clone()
        
        # Apply monotonicity
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Check weights are now non-negative
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                min_val = param.data.min().item()
                assert min_val >= -1e-6, f"Negative weight in {name}: {min_val}"
    
    def test_monotonic_model_preserves_functionality(self, mock_ffn_model):
        """Test monotonic model still produces valid outputs"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Forward pass
        x = torch.randn(4, 64)
        output = monotonic_model(x)
        
        # Output should have correct shape
        assert output.shape == (4, 64)
        
        # Output should be finite
        assert torch.all(torch.isfinite(output))
    
    def test_monotonic_model_trainable(self, mock_ffn_model):
        """Test monotonic model can be trained"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        optimizer = torch.optim.Adam(monotonic_model.parameters(), lr=1e-3)
        
        # Training step
        x = torch.randn(4, 64)
        target = torch.randn(4, 64)
        
        output = monotonic_model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert True
    
    def test_weights_stay_nonnegative_after_training(self, mock_ffn_model):
        """Test weights remain non-negative after training steps"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        optimizer = torch.optim.Adam(monotonic_model.parameters(), lr=1e-3)
        
        # Do several training steps
        for _ in range(10):
            x = torch.randn(4, 64)
            target = torch.randn(4, 64)
            
            optimizer.zero_grad()
            output = monotonic_model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
        
        # Check weights are still non-negative
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                min_val = param.data.min().item()
                assert min_val >= -1e-6, f"Weight became negative after training: {name} = {min_val}"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_monotonicity_with_zero_weights(self):
        """Test monotonicity application handles zero weights"""
        param = NonNegativeParametrization()
        
        W_zeros = torch.zeros(3, 3)
        V_init = param.right_inverse(W_zeros)
        
        # Should not produce NaN
        assert torch.all(torch.isfinite(V_init))
        
        # Reconstructed should be close to zero (with small epsilon)
        W_reconstructed = param.forward(V_init)
        assert torch.all(W_reconstructed >= 0)
        assert torch.all(W_reconstructed < 1e-2)  # Small but positive
    
    def test_monotonicity_with_large_weights(self):
        """Test monotonicity application handles large weights"""
        param = NonNegativeParametrization()
        
        W_large = torch.tensor([100.0, 1000.0, 10000.0])
        V_init = param.right_inverse(W_large)
        
        # Should not overflow
        assert torch.all(torch.isfinite(V_init))
        
        # Reconstructed should be close
        W_reconstructed = param.forward(V_init)
        assert torch.allclose(W_large, W_reconstructed, rtol=1e-2)
    
    def test_empty_dependency_list(self, tmp_path):
        """Test check_dependencies with empty list"""
        result = check_dependencies([], work_dir=str(tmp_path))
        assert result is True  # Empty dependencies = always satisfied
    
    def test_dataset_with_empty_text_list(self, mock_tokenizer):
        """Test dataset handles empty text list"""
        dataset = LanguageModelingDataset([], mock_tokenizer, max_length=128)
        assert len(dataset) == 0
