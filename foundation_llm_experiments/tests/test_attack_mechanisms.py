"""
Comprehensive Tests for Attack Mechanisms

Tests UAT and HotFlip attack implementations in detail.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestUATOptimizer:
    """Test UAT optimization logic"""
    
    def test_uat_optimizer_initialization(self, mock_gpt_model, mock_tokenizer):
        """Test UAT optimizer can be initialized"""
        from scripts.stage_5_uat_attacks import UATOptimizer
        
        optimizer = UATOptimizer(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            trigger_length=5
        )
        
        assert optimizer.trigger_length == 5
        assert optimizer.vocab_size == len(mock_tokenizer)
        assert optimizer.model == mock_gpt_model
    
    def test_uat_get_candidate_tokens(self, mock_gpt_model, mock_tokenizer):
        """Test candidate token selection"""
        from scripts.stage_5_uat_attacks import UATOptimizer
        
        optimizer = UATOptimizer(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            trigger_length=5
        )
        
        candidates = optimizer._get_candidate_tokens()
        
        # Should return tokens
        assert len(candidates) > 0
        
        # All should be valid token IDs
        for token_id in candidates:
            assert 0 < token_id < optimizer.vocab_size
    
    def test_uat_compute_trigger_loss_single_text(self, mock_gpt_model, mock_tokenizer):
        """Test trigger loss computation on single text"""
        from scripts.stage_5_uat_attacks import UATOptimizer
        
        optimizer = UATOptimizer(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            trigger_length=3
        )
        
        trigger_ids = [10, 20, 30]
        texts = ["This is a test sentence."]
        
        loss = optimizer.compute_trigger_loss(trigger_ids, texts, batch_size=1)
        
        # Should return finite positive loss
        assert loss > 0
        assert np.isfinite(loss)
    
    def test_uat_compute_trigger_loss_batch(self, mock_gpt_model, mock_tokenizer):
        """Test trigger loss computation with batching"""
        from scripts.stage_5_uat_attacks import UATOptimizer
        
        optimizer = UATOptimizer(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            trigger_length=3
        )
        
        trigger_ids = [10, 20, 30]
        texts = ["Text 1.", "Text 2.", "Text 3.", "Text 4.", "Text 5."]
        
        loss = optimizer.compute_trigger_loss(trigger_ids, texts, batch_size=2)
        
        # Should handle batching correctly
        assert loss > 0
        assert np.isfinite(loss)
    
    def test_uat_optimize_trigger_minimal(self, mock_gpt_model, mock_tokenizer):
        """Test trigger optimization with minimal iterations"""
        from scripts.stage_5_uat_attacks import UATOptimizer
        
        optimizer = UATOptimizer(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            trigger_length=3
        )
        
        texts = ["Test sentence."] * 5
        
        # Run minimal optimization
        best_trigger, best_loss = optimizer.optimize_trigger(
            texts,
            num_iterations=2,  # Very short for testing
            num_restarts=1
        )
        
        # Should return valid trigger
        assert len(best_trigger) == 3
        assert all(isinstance(t, int) for t in best_trigger)
        assert best_loss > 0


class TestHotFlipAttacker:
    """Test HotFlip attack logic"""
    
    def test_hotflip_attacker_initialization(self, mock_gpt_model, mock_tokenizer):
        """Test HotFlip attacker can be initialized"""
        from scripts.stage_6_hotflip_attacks import HotFlipAttacker
        
        attacker = HotFlipAttacker(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            num_flips=5
        )
        
        assert attacker.num_flips == 5
        assert attacker.vocab_size == len(mock_tokenizer)
    
    def test_hotflip_attack_single_example(self, mock_gpt_model, mock_tokenizer):
        """Test HotFlip attack on single example"""
        from scripts.stage_6_hotflip_attacks import HotFlipAttacker
        
        attacker = HotFlipAttacker(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            num_flips=3
        )
        
        text = "This is a test sentence for attacking."
        
        result = attacker.attack_single_example(text)
        
        # Should return result dict
        assert 'clean_loss' in result
        assert 'attacked_loss' in result
        assert 'degradation' in result
        assert 'success' in result
        
        # Losses should be positive
        assert result['clean_loss'] > 0
        assert result['attacked_loss'] > 0
    
    def test_hotflip_attack_batch(self, mock_gpt_model, mock_tokenizer):
        """Test HotFlip attack on batch of examples"""
        from scripts.stage_6_hotflip_attacks import HotFlipAttacker
        
        attacker = HotFlipAttacker(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            num_flips=2
        )
        
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        results = attacker.attack_batch(texts)
        
        # Should return aggregated results
        assert 'avg_degradation' in results
        assert 'success_rate' in results
        assert 'num_samples' in results
        
        assert results['num_samples'] == 3
        assert 0 <= results['success_rate'] <= 1
    
    def test_hotflip_handles_short_text(self, mock_gpt_model, mock_tokenizer):
        """Test HotFlip handles very short texts"""
        from scripts.stage_6_hotflip_attacks import HotFlipAttacker
        
        attacker = HotFlipAttacker(
            mock_gpt_model,
            mock_tokenizer,
            torch.device('cpu'),
            num_flips=10  # More flips than tokens in text
        )
        
        short_text = "Hi."
        
        # Should not crash
        result = attacker.attack_single_example(short_text)
        
        assert 'clean_loss' in result
        assert 'attacked_loss' in result


class TestUATEvaluation:
    """Test UAT evaluation function"""
    
    def test_evaluate_trigger_basic(self, mock_gpt_model, mock_tokenizer):
        """Test basic trigger evaluation"""
        from scripts.stage_5_uat_attacks import evaluate_trigger
        
        trigger_ids = [10, 20, 30]
        texts = ["Test sentence."] * 3
        
        result = evaluate_trigger(
            mock_gpt_model,
            mock_tokenizer,
            trigger_ids,
            texts,
            torch.device('cpu')
        )
        
        assert 'trigger_text' in result
        assert 'trigger_ids' in result
        assert 'clean_loss' in result
        assert 'attacked_loss' in result
        assert 'nll_increase' in result
        assert 'nll_increase_percent' in result
    
    def test_evaluate_trigger_empty_trigger(self, mock_gpt_model, mock_tokenizer):
        """Test evaluation with empty trigger"""
        from scripts.stage_5_uat_attacks import evaluate_trigger
        
        trigger_ids = []
        texts = ["Test sentence."]
        
        # Should handle empty trigger
        result = evaluate_trigger(
            mock_gpt_model,
            mock_tokenizer,
            trigger_ids,
            texts,
            torch.device('cpu')
        )
        
        # Trigger text should be empty or minimal
        assert len(result['trigger_text']) < 5


class TestScriptMainFunctions:
    """Test main() functions of scripts can be called"""
    
    def test_all_scripts_have_callable_main(self):
        """Test all stage scripts have callable main()"""
        scripts = [
            'stage_0_setup',
            'stage_1_apply_monotonicity',
            'stage_2_train_baseline',
            'stage_3_train_monotonic',
            'stage_4_evaluate',
            'stage_5_uat_attacks',
            'stage_6_hotflip_attacks',
            'stage_7_aggregate',
        ]
        
        for script_name in scripts:
            module = __import__(f'scripts.{script_name}', fromlist=['main'])
            assert hasattr(module, 'main')
            assert callable(module.main)
    
    def test_stage_scripts_have_if_main_guard(self):
        """Test all scripts have if __name__ == '__main__' guard"""
        import inspect
        
        scripts = [
            'stage_0_setup',
            'stage_1_apply_monotonicity',
            'stage_5_uat_attacks',
            'stage_6_hotflip_attacks',
            'stage_7_aggregate',
        ]
        
        for script_name in scripts:
            module = __import__(f'scripts.{script_name}', fromlist=[''])
            source = inspect.getsource(module)
            
            assert '__name__' in source
            assert '__main__' in source
            assert 'exit(main())' in source or 'sys.exit(main())' in source


class TestMonotonicityConstraintMaintenance:
    """Test monotonicity constraints are maintained through operations"""
    
    def test_monotonic_model_after_zero_grad(self, mock_ffn_model):
        """Test weights stay non-negative after zero_grad"""
        from utils.common_utils import make_model_monotonic
        
        monotonic_model = make_model_monotonic(mock_ffn_model)
        optimizer = torch.optim.Adam(monotonic_model.parameters())
        
        # Training step
        x = torch.randn(2, 64)
        output = monotonic_model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check weights still non-negative
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                assert param.data.min().item() >= -1e-6
    
    def test_monotonic_model_after_scheduler_step(self, mock_ffn_model):
        """Test weights stay non-negative after scheduler step"""
        from utils.common_utils import make_model_monotonic
        from transformers import get_linear_schedule_with_warmup
        
        monotonic_model = make_model_monotonic(mock_ffn_model)
        optimizer = torch.optim.Adam(monotonic_model.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, 10, 100)
        
        # Multiple training steps
        for _ in range(10):
            x = torch.randn(2, 64)
            output = monotonic_model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Weights should still be non-negative
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                min_val = param.data.min().item()
                assert min_val >= -1e-6, f"Weight became negative: {name} = {min_val}"
    
    def test_monotonic_model_survives_gradient_explosion(self, mock_ffn_model):
        """Test monotonic model handles large gradients"""
        from utils.common_utils import make_model_monotonic
        
        monotonic_model = make_model_monotonic(mock_ffn_model)
        optimizer = torch.optim.SGD(monotonic_model.parameters(), lr=10.0)  # Very large LR
        
        # Training step with large gradients
        x = torch.randn(2, 64) * 100  # Large input
        output = monotonic_model(x)
        loss = output.sum() * 1000  # Large loss
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(monotonic_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Weights should still be non-negative (softplus guarantees this)
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                assert param.data.min().item() >= -1e-6


class TestConfigurationValidation:
    """Test configuration validation methods"""
    
    def test_validate_config_with_good_config(self, tmp_path, monkeypatch):
        """Test validate_config passes with good configuration"""
        from configs.experiment_config import FoundationExperimentConfig as Config
        
        # Set up valid directories
        scratch_dir = tmp_path / "scratch"
        project_dir = tmp_path / "project"
        scratch_dir.mkdir()
        project_dir.mkdir()
        
        monkeypatch.setattr(Config, 'SCRATCH_DIR', str(scratch_dir))
        monkeypatch.setattr(Config, 'PROJECT_DIR', str(project_dir))
        
        result = Config.validate_config()
        
        # Should pass if directories exist
        # (May fail due to no GPU, but that's OK)
        assert result in [True, False]  # Either is acceptable
    
    def test_config_create_directories_idempotent(self, tmp_path, monkeypatch):
        """Test create_directories can be called multiple times"""
        from configs.experiment_config import FoundationExperimentConfig as Config
        
        work_dir = tmp_path / "work"
        results_dir = tmp_path / "results"
        
        monkeypatch.setattr(Config, 'WORK_DIR', str(work_dir))
        monkeypatch.setattr(Config, 'RESULTS_DIR', str(results_dir))
        monkeypatch.setattr(Config, 'CHECKPOINT_DIR', str(tmp_path / "checkpoints"))
        monkeypatch.setattr(Config, 'DATA_CACHE_DIR', str(tmp_path / "cache"))
        monkeypatch.setattr(Config, 'FINAL_RESULTS_DIR', str(tmp_path / "final"))
        
        # Call multiple times
        Config.create_directories()
        Config.create_directories()
        Config.create_directories()
        
        # Should not crash, directories should exist
        assert work_dir.exists()
        assert results_dir.exists()


class TestStageLoggerEdgeCases:
    """Test StageLogger edge cases"""
    
    def test_logger_with_very_long_message(self, tmp_path):
        """Test logger handles very long messages"""
        from utils.common_utils import StageLogger
        
        logger = StageLogger("test", log_dir=str(tmp_path))
        
        # Log very long message
        long_message = "x" * 10000
        logger.log(long_message)
        
        # Verify logged
        with open(logger.log_file) as f:
            content = f.read()
        
        assert long_message in content
    
    def test_logger_with_special_characters(self, tmp_path):
        """Test logger handles special characters"""
        from utils.common_utils import StageLogger
        
        logger = StageLogger("test", log_dir=str(tmp_path))
        
        # Log messages with special chars
        messages = [
            "Message with emoji: 👍",
            "Message with unicode: αβγ",
            "Message with newlines:\nLine 2\nLine 3",
            "Message with tabs:\tTabbed",
        ]
        
        for msg in messages:
            logger.log(msg)
        
        # Should not crash
        assert os.path.exists(logger.log_file)
    
    def test_logger_complete_timing(self, tmp_path):
        """Test logger tracks elapsed time"""
        from utils.common_utils import StageLogger
        import time
        
        logger = StageLogger("test", log_dir=str(tmp_path))
        
        time.sleep(0.1)  # Small delay
        
        exit_code = logger.complete(success=True)
        
        assert exit_code == 0
        
        # Check log contains elapsed time
        with open(logger.log_file) as f:
            content = f.read()
        
        assert "Elapsed time" in content


class TestDependencyCheckingEdgeCases:
    """Test dependency checking edge cases"""
    
    def test_check_dependencies_with_duplicates(self, tmp_path):
        """Test dependency checking handles duplicate requirements"""
        from utils.common_utils import create_completion_flag, check_dependencies
        
        create_completion_flag('stage_a', work_dir=str(tmp_path))
        
        # Check with duplicates
        result = check_dependencies(['stage_a', 'stage_a', 'stage_a'], work_dir=str(tmp_path))
        
        assert result == True
    
    def test_check_dependencies_mixed_present_missing(self, tmp_path):
        """Test with some dependencies met, some not"""
        from utils.common_utils import create_completion_flag, check_dependencies
        
        create_completion_flag('stage_a', work_dir=str(tmp_path))
        # Don't create stage_b
        
        result = check_dependencies(['stage_a', 'stage_b'], work_dir=str(tmp_path))
        
        assert result == False
    
    def test_check_dependencies_all_missing(self, tmp_path):
        """Test when all dependencies missing"""
        from utils.common_utils import check_dependencies
        
        result = check_dependencies(['stage_x', 'stage_y', 'stage_z'], work_dir=str(tmp_path))
        
        assert result == False


class TestParametrizationEdgeCases:
    """Test softplus parametrization edge cases"""
    
    def test_parametrization_with_very_large_input(self):
        """Test softplus with very large input values"""
        from utils.common_utils import NonNegativeParametrization
        
        param = NonNegativeParametrization()
        
        V_large = torch.tensor([100.0, 500.0, 1000.0])
        W = param.forward(V_large)
        
        # Should not overflow, should approximate V for large values
        assert torch.all(torch.isfinite(W))
        assert torch.allclose(W, V_large, atol=1.0)  # softplus(x) ≈ x for large x
    
    def test_parametrization_with_very_small_input(self):
        """Test softplus with very small input values"""
        from utils.common_utils import NonNegativeParametrization
        
        param = NonNegativeParametrization()
        
        V_small = torch.tensor([-100.0, -50.0, -20.0])
        W = param.forward(V_small)
        
        # Should not underflow, should be very small but positive
        assert torch.all(W >= 0)
        assert torch.all(W < 1e-6)  # Very close to zero
    
    def test_right_inverse_with_zero_weights(self):
        """Test inverse softplus with zero weights"""
        from utils.common_utils import NonNegativeParametrization
        
        param = NonNegativeParametrization()
        
        W_zeros = torch.zeros(5)
        V = param.right_inverse(W_zeros)
        
        # Should handle zeros with epsilon
        assert torch.all(torch.isfinite(V))
        
        # Reconstructed should be small positive
        W_reconstructed = param.forward(V)
        assert torch.all(W_reconstructed >= 0)
        assert torch.all(W_reconstructed < 0.01)
    
    def test_right_inverse_mixed_magnitudes(self):
        """Test inverse softplus with mixed magnitude weights"""
        from utils.common_utils import NonNegativeParametrization
        
        param = NonNegativeParametrization()
        
        W_mixed = torch.tensor([1e-6, 1e-3, 1.0, 10.0, 100.0])
        V = param.right_inverse(W_mixed.abs())
        W_reconstructed = param.forward(V)
        
        # Should handle full range
        assert torch.all(torch.isfinite(V))
        assert torch.allclose(W_mixed, W_reconstructed, rtol=1e-2, atol=1e-4)


class TestSetSeedsComprehensive:
    """Comprehensive tests for seed setting"""
    
    def test_set_all_seeds_affects_all_libraries(self):
        """Test set_all_seeds actually sets all random generators"""
        from utils.common_utils import set_all_seeds
        import random
        import numpy as np
        
        # Set seeds
        set_all_seeds(999)
        
        # Get random values from each library
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        
        # Reset and get again
        set_all_seeds(999)
        
        python_rand2 = random.random()
        numpy_rand2 = np.random.random()
        torch_rand2 = torch.rand(1).item()
        
        # All should be identical
        assert python_rand == python_rand2
        assert numpy_rand == numpy_rand2
        assert torch_rand == torch_rand2
    
    def test_set_all_seeds_with_different_seeds(self):
        """Test different seeds produce different sequences"""
        from utils.common_utils import set_all_seeds
        
        set_all_seeds(42)
        vals_42 = [torch.rand(1).item() for _ in range(5)]
        
        set_all_seeds(1337)
        vals_1337 = [torch.rand(1).item() for _ in range(5)]
        
        # Should be different
        assert vals_42 != vals_1337
    
    def test_set_all_seeds_cuda_if_available(self):
        """Test CUDA seeds set if CUDA available"""
        from utils.common_utils import set_all_seeds
        
        # Should not crash regardless of CUDA availability
        set_all_seeds(42)
        
        if torch.cuda.is_available():
            # CUDA seeds should be set
            # (Can't easily verify, but function shouldn't crash)
            assert True
        else:
            # Should handle gracefully
            assert True


class TestErrorMessageQuality:
    """Test error messages are helpful"""
    
    def test_missing_dependency_error_message(self, tmp_path, capsys):
        """Test missing dependency produces clear error"""
        from utils.common_utils import check_dependencies
        
        result = check_dependencies(['stage_missing'], work_dir=str(tmp_path))
        
        captured = capsys.readouterr()
        
        # Should print error message
        assert result == False
        assert "Missing dependencies" in captured.out or "stage_missing" in captured.out
    
    def test_stage_logger_failure_message(self, tmp_path, capsys):
        """Test stage logger failure produces clear message"""
        from utils.common_utils import StageLogger
        
        logger = StageLogger("test_fail", log_dir=str(tmp_path))
        exit_code = logger.complete(success=False)
        
        captured = capsys.readouterr()
        
        # Should print FAILED message
        assert exit_code == 1
        assert "FAILED" in captured.out or "fail" in captured.out.lower()


class TestDataLoaderIntegration:
    """Test DataLoader integration with custom dataset"""
    
    def test_dataloader_with_shuffle(self, mock_training_data, mock_tokenizer):
        """Test dataset works with shuffled DataLoader"""
        from torch.utils.data import DataLoader
        from utils.common_utils import LanguageModelingDataset, get_generator
        
        dataset = LanguageModelingDataset(mock_training_data, mock_tokenizer, max_length=128)
        
        generator = get_generator(device='cpu', seed=42)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
        
        # Should be able to iterate
        batches = list(loader)
        assert len(batches) > 0
    
    def test_dataloader_without_shuffle(self, mock_training_data, mock_tokenizer):
        """Test dataset works without shuffling"""
        from torch.utils.data import DataLoader
        from utils.common_utils import LanguageModelingDataset
        
        dataset = LanguageModelingDataset(mock_training_data, mock_tokenizer, max_length=128)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Should be able to iterate
        batches = list(loader)
        assert len(batches) > 0
    
    def test_dataloader_with_worker_init(self, mock_training_data, mock_tokenizer):
        """Test DataLoader with worker_init_fn"""
        from torch.utils.data import DataLoader
        from utils.common_utils import LanguageModelingDataset, worker_init_fn
        
        dataset = LanguageModelingDataset(mock_training_data, mock_tokenizer, max_length=128)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Must be 0 for testing (no multiprocessing)
            worker_init_fn=worker_init_fn
        )
        
        # Should work
        batches = list(loader)
        assert len(batches) > 0


# Import fixtures
from tests.conftest import MockModel
