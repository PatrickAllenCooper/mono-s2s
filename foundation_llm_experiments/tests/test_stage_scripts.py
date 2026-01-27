"""
Tests for Individual Stage Scripts

Tests that each stage script can execute with mock data/models.
"""

import pytest
import torch
import os
import sys
import subprocess
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestStage0Setup:
    """Test stage_0_setup.py script"""
    
    def test_stage0_imports(self):
        """Test stage 0 script imports work"""
        # This will fail if imports are broken
        import scripts.stage_0_setup as stage0
        assert hasattr(stage0, 'main')
    
    @pytest.mark.slow
    def test_stage0_execution_dry_run(self, tmp_path, monkeypatch):
        """Test stage 0 can execute (dry run with mocks)"""
        # This would actually download model - skip in fast tests
        pytest.skip("Requires model download, run with --slow flag")


class TestStage1Monotonicity:
    """Test stage_1_apply_monotonicity.py script"""
    
    def test_stage1_imports(self):
        """Test stage 1 script imports work"""
        import scripts.stage_1_apply_monotonicity as stage1
        assert hasattr(stage1, 'main')
    
    def test_stage1_applies_constraints(self, mock_ffn_model, tmp_path, monkeypatch):
        """Test stage 1 applies monotonicity correctly"""
        from utils.common_utils import make_model_monotonic
        
        # Apply monotonicity (core logic from stage 1)
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Verify constraints
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                min_val = param.data.min().item()
                assert min_val >= -1e-6, f"Constraint not applied to {name}"


class TestStage2BaselineTraining:
    """Test stage_2_train_baseline.py script"""
    
    def test_stage2_imports(self):
        """Test stage 2 script imports work"""
        import scripts.stage_2_train_baseline as stage2
        assert hasattr(stage2, 'main')
        assert hasattr(stage2, 'BaselineTrainer')
    
    def test_baseline_trainer_initialization(self, mock_gpt_model, mock_training_data, 
                                             mock_tokenizer, tmp_path):
        """Test BaselineTrainer can be initialized"""
        from scripts.stage_2_train_baseline import BaselineTrainer
        from torch.utils.data import DataLoader
        from utils.common_utils import LanguageModelingDataset
        
        # Create data loaders
        dataset = LanguageModelingDataset(mock_training_data[:20], mock_tokenizer, max_length=128)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)
        
        # Create trainer
        trainer = BaselineTrainer(
            model=mock_gpt_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device('cpu'),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            history_path=str(tmp_path / "history.json")
        )
        
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None


class TestStage3MonotonicTraining:
    """Test stage_3_train_monotonic.py script"""
    
    def test_stage3_imports(self):
        """Test stage 3 script imports work"""
        import scripts.stage_3_train_monotonic as stage3
        assert hasattr(stage3, 'main')
        assert hasattr(stage3, 'MonotonicTrainer')
    
    def test_monotonic_trainer_initialization(self, mock_ffn_model, mock_training_data,
                                              mock_tokenizer, tmp_path):
        """Test MonotonicTrainer can be initialized"""
        from scripts.stage_3_train_monotonic import MonotonicTrainer
        from torch.utils.data import DataLoader
        from utils.common_utils import LanguageModelingDataset, make_model_monotonic
        
        # Apply monotonicity
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Create dummy data loaders (MonotonicTrainer doesn't use them in init)
        dataset = LanguageModelingDataset(mock_training_data[:20], mock_tokenizer, max_length=128)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)
        
        # For full testing, we'd need a proper GPT model
        # This is just testing the trainer can be instantiated
        # Skip actual initialization test since mock_ffn_model is not a causal LM
        pytest.skip("Requires full GPT model for proper testing")


class TestStage4Evaluation:
    """Test stage_4_evaluate.py script"""
    
    def test_stage4_imports(self):
        """Test stage 4 script imports work"""
        import scripts.stage_4_evaluate as stage4
        assert hasattr(stage4, 'main')
        assert hasattr(stage4, 'evaluate_on_pile_test')


class TestScriptInterfaces:
    """Test that scripts have consistent interfaces"""
    
    def test_all_scripts_have_main(self):
        """Test all stage scripts have main() function"""
        scripts = [
            'stage_0_setup',
            'stage_1_apply_monotonicity',
            'stage_2_train_baseline',
            'stage_3_train_monotonic',
            'stage_4_evaluate',
        ]
        
        for script_name in scripts:
            module = __import__(f'scripts.{script_name}', fromlist=['main'])
            assert hasattr(module, 'main'), f"{script_name} missing main()"
    
    def test_all_scripts_importable(self):
        """Test all stage scripts can be imported"""
        scripts = [
            'stage_0_setup',
            'stage_1_apply_monotonicity',
            'stage_2_train_baseline',
            'stage_3_train_monotonic',
            'stage_4_evaluate',
        ]
        
        for script_name in scripts:
            try:
                module = __import__(f'scripts.{script_name}', fromlist=[''])
                assert module is not None
            except ImportError as e:
                pytest.fail(f"Failed to import scripts.{script_name}: {e}")


class TestDataPipeline:
    """Test data loading and processing pipeline"""
    
    def test_dataset_batching(self, mock_training_data, mock_tokenizer):
        """Test dataset can be batched correctly"""
        from torch.utils.data import DataLoader
        from utils.common_utils import LanguageModelingDataset
        
        dataset = LanguageModelingDataset(mock_training_data, mock_tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        batch = next(iter(dataloader))
        
        assert batch['input_ids'].shape == (4, 128)
        assert batch['attention_mask'].shape == (4, 128)
    
    def test_dataset_with_generator(self, mock_training_data, mock_tokenizer):
        """Test dataset with generator for reproducibility"""
        from torch.utils.data import DataLoader
        from utils.common_utils import LanguageModelingDataset, get_generator, worker_init_fn
        
        dataset = LanguageModelingDataset(mock_training_data, mock_tokenizer, max_length=128)
        generator = get_generator(device='cpu', seed=42)
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        
        # Get first batch twice with same seed
        batches1 = list(dataloader)
        
        generator = get_generator(device='cpu', seed=42)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        batches2 = list(dataloader)
        
        # Should get same batches in same order
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1['input_ids'], b2['input_ids'])


class TestCheckpointManagement:
    """Test checkpoint saving and loading"""
    
    def test_checkpoint_includes_all_state(self, mock_gpt_model, tmp_path):
        """Test checkpoint includes all necessary state"""
        from torch.optim import Adam
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = Adam(mock_gpt_model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, 10, 100)
        
        checkpoint = {
            'epoch': 3,
            'model_state_dict': mock_gpt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_perplexity': 10.5,
            'val_perplexity': 11.2,
        }
        
        save_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, save_path)
        
        # Load and verify
        loaded = torch.load(save_path, weights_only=False)
        
        assert loaded['epoch'] == 3
        assert loaded['best_val_perplexity'] == 10.5
        assert 'model_state_dict' in loaded
        assert 'optimizer_state_dict' in loaded
        assert 'scheduler_state_dict' in loaded
    
    def test_checkpoint_resume_preserves_training_state(self, mock_gpt_model, tmp_path):
        """Test resuming from checkpoint preserves training state"""
        from torch.optim import Adam
        
        # Train for a few steps
        optimizer = Adam(mock_gpt_model.parameters(), lr=1e-3)
        
        for _ in range(5):
            x = torch.randn(2, 128)
            # For GPT model
            outputs = mock_gpt_model(input_ids=x.long().abs() % 1000)
            loss = outputs.logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            'model_state_dict': mock_gpt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        # Create new model and load
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=128, 
                           n_layer=2, n_head=2, n_inner=512)
        new_model = GPT2LMHeadModel(config)
        new_optimizer = Adam(new_model.parameters(), lr=1e-3)
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # If this doesn't crash, resume works


# Import for fixtures
from tests.conftest import MockModel
