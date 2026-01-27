"""
Tests for experiment configuration

Validates all configuration parameters and directory setup.
"""

import pytest
import os
import torch
from configs.experiment_config import FoundationExperimentConfig as Config


class TestExperimentConfig:
    """Test configuration validation"""
    
    def test_config_has_required_attributes(self):
        """Verify all required config attributes exist"""
        required_attrs = [
            'MODEL_NAME',
            'WORK_DIR',
            'RESULTS_DIR',
            'CHECKPOINT_DIR',
            'BATCH_SIZE',
            'RECOVERY_LR',
            'MONOTONIC_RECOVERY_LR',
            'MAX_SEQ_LENGTH',
            'RANDOM_SEEDS',
            'CURRENT_SEED',
        ]
        
        for attr in required_attrs:
            assert hasattr(Config, attr), f"Missing required attribute: {attr}"
    
    def test_model_name_valid(self):
        """Test model name is valid"""
        assert isinstance(Config.MODEL_NAME, str)
        assert len(Config.MODEL_NAME) > 0
        assert "pythia" in Config.MODEL_NAME.lower()
    
    def test_batch_size_positive(self):
        """Test batch size is positive integer"""
        assert isinstance(Config.BATCH_SIZE, int)
        assert Config.BATCH_SIZE > 0
        assert Config.BATCH_SIZE <= 32  # Sanity check
    
    def test_learning_rates_valid(self):
        """Test learning rates are in reasonable range"""
        assert 1e-6 <= Config.RECOVERY_LR <= 1e-3
        assert 1e-6 <= Config.MONOTONIC_RECOVERY_LR <= 1e-3
    
    def test_warmup_ratios_valid(self):
        """Test warmup ratios are valid fractions"""
        assert 0 < Config.RECOVERY_WARMUP_RATIO < 1
        assert 0 < Config.MONOTONIC_RECOVERY_WARMUP_RATIO < 1
        # Monotonic should have more warmup
        assert Config.MONOTONIC_RECOVERY_WARMUP_RATIO >= Config.RECOVERY_WARMUP_RATIO
    
    def test_sequence_length_reasonable(self):
        """Test sequence length is reasonable for Pythia"""
        assert Config.MAX_SEQ_LENGTH > 0
        assert Config.MAX_SEQ_LENGTH <= 4096  # Pythia max context
    
    def test_random_seeds_valid(self):
        """Test random seeds are valid"""
        assert isinstance(Config.RANDOM_SEEDS, list)
        assert len(Config.RANDOM_SEEDS) == 5
        assert all(isinstance(s, int) for s in Config.RANDOM_SEEDS)
        assert 42 in Config.RANDOM_SEEDS  # Should include default
    
    def test_current_seed_in_seed_list(self):
        """Test current seed is in the seed list"""
        assert Config.CURRENT_SEED in Config.RANDOM_SEEDS
    
    def test_attack_params_valid(self):
        """Test attack parameters are reasonable"""
        assert Config.ATTACK_TRIGGER_LENGTH > 0
        assert Config.ATTACK_TRIGGER_LENGTH <= 20
        assert Config.ATTACK_NUM_ITERATIONS > 0
        assert Config.ATTACK_NUM_RESTARTS > 0
        assert Config.HOTFLIP_NUM_FLIPS > 0
    
    def test_to_dict_method(self):
        """Test configuration can be exported as dict"""
        config_dict = Config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'MODEL_NAME' in config_dict
        assert 'BATCH_SIZE' in config_dict
    
    def test_get_device_method(self):
        """Test device detection works"""
        device = Config.get_device()
        assert isinstance(device, torch.device)
        # Should be cuda if available, cpu otherwise
        assert device.type in ['cuda', 'cpu']
    
    def test_create_directories_method(self, tmp_path, monkeypatch):
        """Test directory creation works"""
        # Temporarily override paths
        test_work_dir = tmp_path / "test_work"
        monkeypatch.setattr(Config, 'WORK_DIR', str(test_work_dir))
        monkeypatch.setattr(Config, 'RESULTS_DIR', str(tmp_path / "results"))
        monkeypatch.setattr(Config, 'CHECKPOINT_DIR', str(tmp_path / "checkpoints"))
        monkeypatch.setattr(Config, 'DATA_CACHE_DIR', str(tmp_path / "cache"))
        monkeypatch.setattr(Config, 'FINAL_RESULTS_DIR', str(tmp_path / "final"))
        
        Config.create_directories()
        
        assert os.path.exists(Config.WORK_DIR)
        assert os.path.exists(Config.RESULTS_DIR)
        assert os.path.exists(Config.CHECKPOINT_DIR)
    
    def test_time_limits_reasonable(self):
        """Test SLURM time limits are reasonable"""
        # Parse time strings (HH:MM:SS format)
        def parse_time(time_str):
            parts = time_str.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        setup_time = parse_time(Config.TIME_SETUP)
        train_baseline_time = parse_time(Config.TIME_TRAIN_BASELINE)
        train_monotonic_time = parse_time(Config.TIME_TRAIN_MONOTONIC)
        
        # Basic sanity checks
        assert setup_time > 0
        assert train_baseline_time > setup_time  # Training takes longer
        assert train_monotonic_time >= train_baseline_time  # Monotonic needs more time
        assert train_monotonic_time <= 48 * 3600  # Not more than 48 hours
    
    def test_eval_batch_size_reasonable(self):
        """Test evaluation batch size is reasonable"""
        assert Config.EVAL_BATCH_SIZE >= Config.BATCH_SIZE
        assert Config.EVAL_BATCH_SIZE <= 64


class TestConfigConsistency:
    """Test configuration consistency and compatibility"""
    
    def test_monotonic_epochs_vs_baseline(self):
        """Test monotonic gets at least as many epochs as baseline"""
        assert Config.MONOTONIC_RECOVERY_EPOCHS >= Config.RECOVERY_EPOCHS
    
    def test_attack_batch_size_fits_memory(self):
        """Test attack batch size won't cause OOM"""
        assert Config.ATTACK_LOSS_BATCH_SIZE <= Config.EVAL_BATCH_SIZE
    
    def test_quick_mode_sample_sizes(self):
        """Test quick mode has reasonable sample sizes"""
        if hasattr(Config, 'QUICK_TRAINING_SAMPLES'):
            assert Config.QUICK_TRAINING_SAMPLES > 0
            assert Config.QUICK_TRAINING_SAMPLES < 50000  # Should be quick
        
        if hasattr(Config, 'QUICK_EVAL_SIZE'):
            assert Config.QUICK_EVAL_SIZE > 0
            assert Config.QUICK_EVAL_SIZE < 1000
    
    def test_slurm_config_valid(self):
        """Test SLURM configuration is valid"""
        assert Config.SLURM_NODES == 1  # Single node for now
        assert Config.SLURM_GPUS_PER_NODE == 1  # Single GPU
        assert Config.SLURM_MEM_GB > 0
        assert Config.SLURM_MEM_GB <= 512  # Sanity check
        
        # Partition should be string
        assert isinstance(Config.SLURM_PARTITION, str)
        assert len(Config.SLURM_PARTITION) > 0


class TestConfigEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_config_handles_missing_env_vars(self, monkeypatch):
        """Test config handles missing environment variables"""
        # Remove environment variables
        monkeypatch.delenv('SCRATCH', raising=False)
        monkeypatch.delenv('PROJECT', raising=False)
        monkeypatch.delenv('USER', raising=False)
        
        # Should still have default values
        assert Config.SCRATCH_DIR is not None
        assert Config.PROJECT_DIR is not None
    
    def test_config_to_dict_excludes_private(self):
        """Test to_dict excludes private/callable attributes"""
        config_dict = Config.to_dict()
        
        # Should not include private attributes
        assert not any(k.startswith('_') for k in config_dict.keys())
        
        # Should not include methods
        assert 'to_dict' not in config_dict
        assert 'create_directories' not in config_dict
        assert 'get_device' not in config_dict
    
    def test_validate_config_with_missing_dirs(self, tmp_path, monkeypatch):
        """Test validation fails gracefully with missing directories"""
        # Point to non-existent directories
        monkeypatch.setattr(Config, 'SCRATCH_DIR', str(tmp_path / "nonexistent"))
        monkeypatch.setattr(Config, 'PROJECT_DIR', str(tmp_path / "also_nonexistent"))
        
        # Should return False, not crash
        result = Config.validate_config()
        assert result is False
