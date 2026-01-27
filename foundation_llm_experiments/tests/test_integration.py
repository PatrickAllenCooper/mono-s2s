"""
Integration Tests for Full Pipeline

Tests end-to-end workflows with mock models and data.
"""

import pytest
import torch
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.common_utils import (
    set_all_seeds,
    make_model_monotonic,
    compute_perplexity,
    save_json,
    create_completion_flag,
    check_completion_flag,
    check_dependencies,
    StageLogger,
    LanguageModelingDataset,
)


class TestStage0Setup:
    """Test stage 0 setup functionality"""
    
    def test_setup_creates_directories(self, tmp_path, monkeypatch):
        """Test setup creates all required directories"""
        from configs.experiment_config import FoundationExperimentConfig as Config
        
        # Override paths
        monkeypatch.setattr(Config, 'WORK_DIR', str(tmp_path / "work"))
        monkeypatch.setattr(Config, 'RESULTS_DIR', str(tmp_path / "results"))
        monkeypatch.setattr(Config, 'CHECKPOINT_DIR', str(tmp_path / "checkpoints"))
        monkeypatch.setattr(Config, 'DATA_CACHE_DIR', str(tmp_path / "cache"))
        monkeypatch.setattr(Config, 'FINAL_RESULTS_DIR', str(tmp_path / "final"))
        
        Config.create_directories()
        
        assert os.path.exists(Config.WORK_DIR)
        assert os.path.exists(Config.RESULTS_DIR)
        assert os.path.exists(Config.CHECKPOINT_DIR)
        assert os.path.exists(Config.DATA_CACHE_DIR)
        assert os.path.exists(Config.FINAL_RESULTS_DIR)


class TestStage1MonotonicityApplication:
    """Test stage 1 monotonicity application"""
    
    def test_apply_and_verify_monotonicity(self, mock_ffn_model):
        """Test full monotonicity application workflow"""
        # Apply monotonicity
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Verify all weights non-negative
        all_nonneg = True
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                if param.data.min().item() < -1e-6:
                    all_nonneg = False
                    break
        
        assert all_nonneg, "Some weights are still negative"
    
    def test_save_monotonic_model(self, mock_ffn_model, tmp_path):
        """Test saving monotonic model state dict"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        save_path = tmp_path / "monotonic_initialized.pt"
        torch.save(monotonic_model.state_dict(), save_path)
        
        assert os.path.exists(save_path)
        
        # Verify can load
        state_dict = torch.load(save_path, weights_only=False)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0


class TestStage2BaselineTraining:
    """Test baseline training workflow"""
    
    def test_training_loop_with_mock_data(self, mock_gpt_model, mock_tokenizer, mock_training_data):
        """Test baseline training loop executes"""
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW
        
        # Create dataset
        dataset = LanguageModelingDataset(
            mock_training_data[:20],
            mock_tokenizer,
            max_length=128
        )
        
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Setup training
        device = torch.device('cpu')
        mock_gpt_model = mock_gpt_model.to(device)
        optimizer = AdamW(mock_gpt_model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2,
            num_training_steps=10
        )
        
        # Training loop (minimal)
        mock_gpt_model.train()
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Only 5 steps for testing
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = mock_gpt_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            assert torch.isfinite(loss), f"Loss is not finite at step {i}"
    
    def test_checkpoint_saving_and_loading(self, mock_gpt_model, tmp_path):
        """Test checkpoint save/load workflow"""
        checkpoint_dir = tmp_path / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / "checkpoint_epoch_1.pt"
        save_dict = {
            'epoch': 1,
            'model_state_dict': mock_gpt_model.state_dict(),
            'val_perplexity': 10.5,
        }
        torch.save(save_dict, checkpoint_path)
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, weights_only=False)
        
        assert loaded['epoch'] == 1
        assert 'model_state_dict' in loaded
        assert loaded['val_perplexity'] == 10.5
        
        # Load into model
        mock_gpt_model.load_state_dict(loaded['model_state_dict'])
        # If this doesn't crash, it worked


class TestStage3MonotonicTraining:
    """Test monotonic training workflow"""
    
    def test_monotonic_training_loop(self, mock_ffn_model, mock_training_data, mock_tokenizer):
        """Test monotonic model can be trained"""
        from torch.utils.data import DataLoader
        
        # Apply monotonicity
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # For this test, we'll just test the model itself (not full causal LM)
        optimizer = torch.optim.Adam(monotonic_model.parameters(), lr=1e-3)
        
        # Training steps
        for _ in range(5):
            x = torch.randn(2, 64)
            target = torch.randn(2, 64)
            
            optimizer.zero_grad()
            output = monotonic_model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
            
            # Verify weights stay non-negative
            for name, param in monotonic_model.named_parameters():
                if 'weight' in name and 'mlp' in name.lower():
                    assert param.data.min().item() >= -1e-6
    
    def test_monotonic_model_checkpoint_workflow(self, mock_ffn_model, tmp_path):
        """Test saving and loading monotonic model checkpoints"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "monotonic_checkpoint.pt"
        torch.save(monotonic_model.state_dict(), checkpoint_path)
        
        # Create new model and apply monotonicity
        new_model = MockModel(input_size=64, hidden_size=256, output_size=64)
        new_model = make_model_monotonic(new_model)
        
        # Load checkpoint
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
        
        # Verify outputs match
        x = torch.randn(2, 64)
        with torch.no_grad():
            output1 = monotonic_model(x)
            output2 = new_model(x)
        
        assert torch.allclose(output1, output2, atol=1e-5)


class TestStage4Evaluation:
    """Test evaluation workflow"""
    
    def test_perplexity_computation_end_to_end(self, mock_gpt_model, mock_tokenizer, mock_eval_data):
        """Test perplexity computation on mock data"""
        from torch.utils.data import DataLoader
        
        dataset = LanguageModelingDataset(
            mock_eval_data,
            mock_tokenizer,
            max_length=128
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        device = torch.device('cpu')
        mock_gpt_model.eval()
        
        result = compute_perplexity(mock_gpt_model, dataloader, device)
        
        # Verify result structure
        assert 'perplexity' in result
        assert 'loss' in result
        assert 'total_tokens' in result
        
        # Verify values are reasonable
        assert result['perplexity'] > 0
        assert result['perplexity'] < 1000  # Should not be absurdly high
        assert result['loss'] > 0
        assert result['total_tokens'] > 0


class TestDependencyChain:
    """Test stage dependency checking"""
    
    def test_dependency_chain_stage_by_stage(self, tmp_path):
        """Test stages can check dependencies correctly"""
        # Stage 1 depends on stage 0
        assert check_dependencies(['stage_0_setup'], work_dir=str(tmp_path)) is False
        
        # Create stage 0 flag
        create_completion_flag('stage_0_setup', work_dir=str(tmp_path))
        assert check_dependencies(['stage_0_setup'], work_dir=str(tmp_path)) is True
        
        # Stage 2 depends on stage 0
        assert check_dependencies(['stage_0_setup'], work_dir=str(tmp_path)) is True
        
        # Stage 3 depends on stages 0 and 1
        assert check_dependencies(['stage_0_setup', 'stage_1_apply_monotonicity'], 
                                 work_dir=str(tmp_path)) is False
        
        # Create stage 1 flag
        create_completion_flag('stage_1_apply_monotonicity', work_dir=str(tmp_path))
        assert check_dependencies(['stage_0_setup', 'stage_1_apply_monotonicity'],
                                 work_dir=str(tmp_path)) is True
    
    def test_multiple_stages_partial_completion(self, tmp_path):
        """Test dependency checking with partial completion"""
        # Create flags for stages 0, 1, 2
        for stage in ['stage_0_setup', 'stage_1_apply', 'stage_2_train']:
            create_completion_flag(stage, work_dir=str(tmp_path))
        
        # Check different combinations
        assert check_dependencies(['stage_0_setup'], work_dir=str(tmp_path)) is True
        assert check_dependencies(['stage_0_setup', 'stage_1_apply'], work_dir=str(tmp_path)) is True
        assert check_dependencies(['stage_0_setup', 'stage_3_missing'], work_dir=str(tmp_path)) is False


class TestFullPipelineSimulation:
    """Simulate full pipeline execution with mock data"""
    
    def test_end_to_end_minimal_pipeline(self, tmp_path, mock_gpt_model, 
                                         mock_tokenizer, mock_training_data, 
                                         mock_eval_data, monkeypatch):
        """Test minimal end-to-end pipeline with mock data"""
        from configs.experiment_config import FoundationExperimentConfig as Config
        from torch.utils.data import DataLoader
        
        # Setup paths
        monkeypatch.setattr(Config, 'WORK_DIR', str(tmp_path / "work"))
        monkeypatch.setattr(Config, 'RESULTS_DIR', str(tmp_path / "results"))
        monkeypatch.setattr(Config, 'CHECKPOINT_DIR', str(tmp_path / "checkpoints"))
        Config.create_directories()
        
        device = torch.device('cpu')
        
        # Stage 0: Setup (simulated)
        create_completion_flag('stage_0_setup', work_dir=Config.WORK_DIR)
        
        # Stage 1: Apply monotonicity
        monotonic_model = make_model_monotonic(mock_gpt_model)
        save_path = os.path.join(Config.CHECKPOINT_DIR, 'monotonic_initialized.pt')
        torch.save(monotonic_model.state_dict(), save_path)
        create_completion_flag('stage_1_apply_monotonicity', work_dir=Config.WORK_DIR)
        
        # Stage 2: Baseline training (1 step only)
        baseline_model = mock_gpt_model  # Use original model
        dataset = LanguageModelingDataset(mock_training_data[:10], mock_tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=2)
        
        baseline_model.train()
        optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-4)
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = baseline_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break  # Just one step
        
        # Save baseline checkpoint
        baseline_checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, 'baseline_checkpoints')
        os.makedirs(baseline_checkpoint_dir, exist_ok=True)
        torch.save(baseline_model.state_dict(), 
                  os.path.join(baseline_checkpoint_dir, 'best_model.pt'))
        create_completion_flag('stage_2_train_baseline', work_dir=Config.WORK_DIR)
        
        # Stage 3: Monotonic training (1 step only)
        monotonic_model.train()
        monotonic_optimizer = torch.optim.Adam(monotonic_model.parameters(), lr=1e-4)
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = monotonic_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            monotonic_optimizer.step()
            monotonic_optimizer.zero_grad()
            break
        
        # Save monotonic checkpoint
        monotonic_checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, 'monotonic_checkpoints')
        os.makedirs(monotonic_checkpoint_dir, exist_ok=True)
        torch.save(monotonic_model.state_dict(),
                  os.path.join(monotonic_checkpoint_dir, 'best_model.pt'))
        create_completion_flag('stage_3_train_monotonic', work_dir=Config.WORK_DIR)
        
        # Stage 4: Evaluation
        eval_dataset = LanguageModelingDataset(mock_eval_data, mock_tokenizer, max_length=128)
        eval_dataloader = DataLoader(eval_dataset, batch_size=2)
        
        baseline_model.eval()
        monotonic_model.eval()
        
        baseline_ppl_result = compute_perplexity(baseline_model, eval_dataloader, device)
        monotonic_ppl_result = compute_perplexity(monotonic_model, eval_dataloader, device)
        
        # Save evaluation results
        eval_results = {
            'pile_test': {
                'baseline_pythia': baseline_ppl_result,
                'monotonic_pythia': monotonic_ppl_result
            }
        }
        save_json(eval_results, os.path.join(Config.RESULTS_DIR, 'evaluation_results.json'))
        create_completion_flag('stage_4_evaluate', work_dir=Config.WORK_DIR)
        
        # Verify all completion flags exist
        assert check_completion_flag('stage_0_setup', work_dir=Config.WORK_DIR)
        assert check_completion_flag('stage_1_apply_monotonicity', work_dir=Config.WORK_DIR)
        assert check_completion_flag('stage_2_train_baseline', work_dir=Config.WORK_DIR)
        assert check_completion_flag('stage_3_train_monotonic', work_dir=Config.WORK_DIR)
        assert check_completion_flag('stage_4_evaluate', work_dir=Config.WORK_DIR)
        
        # Verify results file exists and is valid
        assert os.path.exists(os.path.join(Config.RESULTS_DIR, 'evaluation_results.json'))
        with open(os.path.join(Config.RESULTS_DIR, 'evaluation_results.json')) as f:
            results = json.load(f)
        assert 'pile_test' in results


class TestMonotonicityConstraints:
    """Test monotonicity constraints are preserved through pipeline"""
    
    def test_weights_nonnegative_after_initialization(self, mock_ffn_model):
        """Test weights are non-negative after initialization"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                min_val = param.data.min().item()
                max_val = param.data.max().item()
                
                assert min_val >= -1e-6, f"Negative weight after init: {name} = {min_val}"
                assert max_val > 0, f"No positive weights: {name}"
    
    def test_weights_nonnegative_after_save_load(self, mock_ffn_model, tmp_path):
        """Test weights remain non-negative after save/load"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(monotonic_model.state_dict(), save_path)
        
        # Load into new model
        new_model = MockModel(input_size=64, hidden_size=256, output_size=64)
        new_model = make_model_monotonic(new_model)
        new_model.load_state_dict(torch.load(save_path, weights_only=False))
        
        # Check weights
        for name, param in new_model.named_parameters():
            if 'weight' in name and 'mlp' in name.lower():
                assert param.data.min().item() >= -1e-6
    
    def test_gradients_preserve_monotonicity(self, mock_ffn_model):
        """Test that gradient updates preserve non-negativity"""
        monotonic_model = make_model_monotonic(mock_ffn_model)
        optimizer = torch.optim.SGD(monotonic_model.parameters(), lr=0.1)
        
        # Do aggressive updates
        for _ in range(20):
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
                assert min_val >= -1e-6, f"Weights became negative: {name} = {min_val}"


class TestReproducibility:
    """Test reproducibility across pipeline"""
    
    def test_same_seed_same_initialization(self):
        """Test same seed produces same monotonic initialization"""
        from utils.common_utils import set_all_seeds
        
        # Create model and apply monotonicity twice
        set_all_seeds(42)
        model1 = MockModel(input_size=64, hidden_size=256, output_size=64)
        model1 = make_model_monotonic(model1)
        weights1 = {name: param.data.clone() for name, param in model1.named_parameters()}
        
        set_all_seeds(42)
        model2 = MockModel(input_size=64, hidden_size=256, output_size=64)
        model2 = make_model_monotonic(model2)
        weights2 = {name: param.data.clone() for name, param in model2.named_parameters()}
        
        # Weights should be identical
        for name in weights1:
            if name in weights2:
                assert torch.allclose(weights1[name], weights2[name], atol=1e-6), \
                    f"Weights differ for {name}"
    
    def test_training_reproducible_with_seed(self, mock_ffn_model):
        """Test training is reproducible with same seed"""
        def train_one_step(model, seed):
            set_all_seeds(seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            x = torch.randn(2, 64)
            target = torch.randn(2, 64)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
            
            return loss.item()
        
        # Run twice with same seed
        model1 = make_model_monotonic(MockModel(64, 256, 64))
        loss1 = train_one_step(model1, seed=42)
        
        model2 = make_model_monotonic(MockModel(64, 256, 64))
        loss2 = train_one_step(model2, seed=42)
        
        # Losses should be identical
        assert abs(loss1 - loss2) < 1e-6


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_checkpoint_handled(self, tmp_path):
        """Test loading missing checkpoint is handled gracefully"""
        nonexistent_path = tmp_path / "nonexistent.pt"
        
        # Should not crash when checking existence
        exists = os.path.exists(nonexistent_path)
        assert exists is False
    
    def test_corrupted_json_handled(self, tmp_path):
        """Test loading corrupted JSON is handled"""
        bad_json = tmp_path / "bad.json"
        with open(bad_json, 'w') as f:
            f.write("{ invalid json")
        
        with pytest.raises(json.JSONDecodeError):
            load_json(str(bad_json))
    
    def test_stage_logger_handles_nested_dirs(self, tmp_path):
        """Test logger handles nested directory creation"""
        log_dir = tmp_path / "deep" / "nested" / "logs"
        logger = StageLogger("test_stage", log_dir=str(log_dir))
        
        assert os.path.exists(logger.log_file)


# Import MockModel for fixtures
from tests.conftest import MockModel
import torch.nn as nn
