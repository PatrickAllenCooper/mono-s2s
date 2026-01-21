"""
Tests for ExperimentConfig

Comprehensive tests for configuration class and helper functions.
"""
import os
import sys
import pytest
import torch
from pathlib import Path

# Add hpc_version to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))

from hpc_version.configs.experiment_config import ExperimentConfig, get_config, get_paths


class TestExperimentConfig:
    """Tests for ExperimentConfig class"""
    
    def test_default_values(self):
        """Test that default configuration values are set correctly"""
        assert ExperimentConfig.MODEL_NAME == "t5-small"
        assert ExperimentConfig.LEARNING_RATE == 5e-5
        assert ExperimentConfig.NUM_EPOCHS == 5
        assert ExperimentConfig.BATCH_SIZE == 4
        assert ExperimentConfig.MAX_INPUT_LENGTH == 512
        assert ExperimentConfig.MAX_TARGET_LENGTH == 128
    
    def test_monotonic_hyperparameters(self):
        """Test monotonic-specific hyperparameters"""
        assert ExperimentConfig.MONOTONIC_NUM_EPOCHS == 7
        assert ExperimentConfig.MONOTONIC_WARMUP_RATIO == 0.15
        assert ExperimentConfig.MONOTONIC_LEARNING_RATE == 5e-5
    
    def test_decode_parameters(self):
        """Test decoding parameters"""
        assert ExperimentConfig.DECODE_NUM_BEAMS == 4
        assert ExperimentConfig.DECODE_LENGTH_PENALTY == 1.2
        assert ExperimentConfig.DECODE_MIN_NEW_TOKENS == 10
        assert ExperimentConfig.DECODE_MAX_NEW_TOKENS == 80
        assert ExperimentConfig.DECODE_NO_REPEAT_NGRAM_SIZE == 3
        assert ExperimentConfig.DECODE_EARLY_STOPPING is True
    
    def test_rouge_configuration(self):
        """Test ROUGE configuration"""
        assert "rouge1" in ExperimentConfig.ROUGE_METRICS
        assert "rouge2" in ExperimentConfig.ROUGE_METRICS
        assert "rougeLsum" in ExperimentConfig.ROUGE_METRICS
        assert ExperimentConfig.ROUGE_USE_STEMMER is True
        assert ExperimentConfig.ROUGE_BOOTSTRAP_SAMPLES == 1000
    
    def test_attack_configuration(self):
        """Test attack configuration parameters"""
        assert ExperimentConfig.ATTACK_TRIGGER_LENGTH == 5
        assert ExperimentConfig.ATTACK_NUM_CANDIDATES == 100
        assert ExperimentConfig.ATTACK_NUM_GRAD_STEPS == 50
        assert ExperimentConfig.ATTACK_LOSS_BATCH_SIZE == 8
    
    def test_random_seeds(self):
        """Test random seed configuration"""
        assert len(ExperimentConfig.RANDOM_SEEDS) == 5
        assert 42 in ExperimentConfig.RANDOM_SEEDS
        assert isinstance(ExperimentConfig.CURRENT_SEED, int)
    
    def test_paths_exist(self):
        """Test that path attributes are defined"""
        assert hasattr(ExperimentConfig, 'SCRATCH_DIR')
        assert hasattr(ExperimentConfig, 'PROJECT_DIR')
        assert hasattr(ExperimentConfig, 'WORK_DIR')
        assert hasattr(ExperimentConfig, 'RESULTS_DIR')
        assert hasattr(ExperimentConfig, 'CHECKPOINT_DIR')
        assert hasattr(ExperimentConfig, 'DATA_CACHE_DIR')
    
    def test_dataset_configuration(self):
        """Test dataset configuration"""
        assert isinstance(ExperimentConfig.TRAIN_DATASETS, list)
        assert len(ExperimentConfig.TRAIN_DATASETS) > 0
        assert isinstance(ExperimentConfig.TEST_DATASETS, list)
        assert len(ExperimentConfig.TEST_DATASETS) > 0
        
        # Check dataset retry configuration
        assert ExperimentConfig.DATASET_MAX_RETRIES == 3
        assert ExperimentConfig.DATASET_RETRY_DELAY == 10
        assert ExperimentConfig.DATASET_ALLOW_PARTIAL is True
    
    def test_slurm_configuration(self):
        """Test SLURM/HPC configuration"""
        assert hasattr(ExperimentConfig, 'SLURM_PARTITION')
        assert hasattr(ExperimentConfig, 'SLURM_QOS')
        assert ExperimentConfig.SLURM_NODES == 1
        assert ExperimentConfig.SLURM_TASKS_PER_NODE == 1
        assert ExperimentConfig.SLURM_GPUS_PER_NODE == 1
    
    def test_to_dict(self):
        """Test configuration export to dictionary"""
        config_dict = ExperimentConfig.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'MODEL_NAME' in config_dict
        assert 'LEARNING_RATE' in config_dict
        assert 'NUM_EPOCHS' in config_dict
        
        # Should not include private or callable attributes
        assert not any(k.startswith('_') for k in config_dict.keys())
        assert not any(callable(v) for v in config_dict.values())
    
    def test_create_directories(self, temp_dir, monkeypatch):
        """Test directory creation"""
        work_dir = os.path.join(temp_dir, "work")
        results_dir = os.path.join(temp_dir, "results")
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        
        monkeypatch.setattr(ExperimentConfig, "WORK_DIR", work_dir)
        monkeypatch.setattr(ExperimentConfig, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(ExperimentConfig, "CHECKPOINT_DIR", checkpoint_dir)
        monkeypatch.setattr(ExperimentConfig, "DATA_CACHE_DIR", os.path.join(temp_dir, "cache"))
        monkeypatch.setattr(ExperimentConfig, "FINAL_RESULTS_DIR", os.path.join(temp_dir, "final"))
        
        ExperimentConfig.create_directories()
        
        assert os.path.exists(work_dir)
        assert os.path.exists(results_dir)
        assert os.path.exists(checkpoint_dir)
        assert os.path.exists(os.path.join(checkpoint_dir, 'baseline_checkpoints'))
        assert os.path.exists(os.path.join(checkpoint_dir, 'monotonic_checkpoints'))
    
    def test_get_device(self):
        """Test device detection"""
        device = ExperimentConfig.get_device()
        
        assert isinstance(device, torch.device)
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"
    
    def test_validate_config(self, temp_dir, monkeypatch):
        """Test configuration validation"""
        # Setup valid paths
        scratch = os.path.join(temp_dir, "scratch")
        project = os.path.join(temp_dir, "project")
        os.makedirs(scratch, exist_ok=True)
        os.makedirs(project, exist_ok=True)
        
        monkeypatch.setattr(ExperimentConfig, "SCRATCH_DIR", scratch)
        monkeypatch.setattr(ExperimentConfig, "PROJECT_DIR", project)
        monkeypatch.setattr(ExperimentConfig, "USE_FULL_TEST_SETS", False)
        monkeypatch.setattr(ExperimentConfig, "BATCH_SIZE", 4)
        
        # Should pass with valid paths
        result = ExperimentConfig.validate_config()
        # Result may be False if no GPU, but should not raise
        assert isinstance(result, bool)
    
    def test_quick_test_configuration(self):
        """Test quick test size configuration"""
        assert hasattr(ExperimentConfig, 'QUICK_TEST_SIZE')
        assert hasattr(ExperimentConfig, 'TRIGGER_OPT_SIZE_QUICK')
        assert hasattr(ExperimentConfig, 'TRIGGER_EVAL_SIZE_QUICK')
        assert hasattr(ExperimentConfig, 'TRIGGER_OPT_SIZE_FULL')
        assert hasattr(ExperimentConfig, 'TRIGGER_EVAL_SIZE_FULL')


class TestConfigHelpers:
    """Tests for configuration helper functions"""
    
    def test_get_config(self):
        """Test get_config function"""
        config = get_config()
        assert config == ExperimentConfig
    
    def test_get_paths(self):
        """Test get_paths function"""
        paths = get_paths()
        
        assert isinstance(paths, dict)
        assert 'work_dir' in paths
        assert 'results_dir' in paths
        assert 'checkpoint_dir' in paths
        assert 'data_cache_dir' in paths
        assert 'final_results_dir' in paths
        assert 'baseline_checkpoint_dir' in paths
        assert 'monotonic_checkpoint_dir' in paths
