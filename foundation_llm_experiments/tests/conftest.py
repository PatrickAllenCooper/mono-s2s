"""
Pytest Configuration and Shared Fixtures

Provides mock models, datasets, and utilities for testing.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config


@pytest.fixture(scope="session")
def temp_work_dir():
    """Create temporary work directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="foundation_llm_test_")
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_work_dir):
    """Mock configuration with temporary paths"""
    class MockConfig:
        WORK_DIR = os.path.join(temp_work_dir, "work")
        RESULTS_DIR = os.path.join(temp_work_dir, "results")
        CHECKPOINT_DIR = os.path.join(temp_work_dir, "checkpoints")
        DATA_CACHE_DIR = os.path.join(temp_work_dir, "cache")
        FINAL_RESULTS_DIR = os.path.join(temp_work_dir, "final")
        
        MODEL_NAME = "EleutherAI/pythia-70m"  # Tiny model for testing
        CURRENT_SEED = 42
        MAX_SEQ_LENGTH = 512
        BATCH_SIZE = 2
        GRADIENT_ACCUMULATION_STEPS = 1
        MAX_GRAD_NORM = 1.0
        
        RECOVERY_EPOCHS = 1
        RECOVERY_LR = 1e-5
        RECOVERY_WARMUP_RATIO = 0.1
        RECOVERY_WEIGHT_DECAY = 0.01
        
        MONOTONIC_RECOVERY_EPOCHS = 1
        MONOTONIC_RECOVERY_LR = 1e-5
        MONOTONIC_RECOVERY_WARMUP_RATIO = 0.15
        MONOTONIC_RECOVERY_WEIGHT_DECAY = 0.01
        
        EVAL_BATCH_SIZE = 4
        USE_FULL_EVAL_SETS = False
        
        ATTACK_TRIGGER_LENGTH = 5
        ATTACK_NUM_ITERATIONS = 10
        ATTACK_NUM_RESTARTS = 2
        HOTFLIP_NUM_FLIPS = 3
        HOTFLIP_NUM_SAMPLES = 10
        
        @classmethod
        def create_directories(cls):
            for d in [cls.WORK_DIR, cls.RESULTS_DIR, cls.CHECKPOINT_DIR,
                     cls.DATA_CACHE_DIR, cls.FINAL_RESULTS_DIR]:
                os.makedirs(d, exist_ok=True)
        
        @classmethod
        def get_device(cls):
            return torch.device("cpu")
    
    # Create directories
    MockConfig.create_directories()
    
    return MockConfig


@pytest.fixture
def mock_gpt_model():
    """Create a tiny GPT-style model for testing"""
    from transformers import GPT2Config, GPT2LMHeadModel
    
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
        n_inner=512,  # FFN intermediate size
    )
    
    model = GPT2LMHeadModel(config)
    return model


@pytest.fixture
def mock_pythia_model():
    """Create a tiny Pythia-style model for testing"""
    try:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        
        config = GPTNeoXConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=128,
        )
        
        model = GPTNeoXForCausalLM(config)
        return model
    except ImportError:
        # Fallback to GPT2 if GPTNeoX not available
        pytest.skip("GPTNeoX not available, using GPT2 instead")


@pytest.fixture
def mock_tokenizer():
    """Create a simple tokenizer for testing"""
    from transformers import AutoTokenizer
    
    # Use a small, fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


@pytest.fixture
def mock_training_data():
    """Generate mock training data"""
    texts = [
        "This is a test sentence.",
        "Another example text for training.",
        "Machine learning is fascinating.",
        "Neural networks can learn patterns.",
        "Testing is important for reliability.",
    ] * 20  # 100 samples total
    
    return texts


@pytest.fixture
def mock_eval_data():
    """Generate mock evaluation data"""
    texts = [
        "Evaluation example one.",
        "Evaluation example two.",
        "Testing model performance.",
    ] * 10  # 30 samples
    
    return texts


@pytest.fixture
def set_test_seed():
    """Set all random seeds for reproducibility in tests"""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@pytest.fixture
def mock_checkpoint_dir(temp_work_dir):
    """Create mock checkpoint directory"""
    checkpoint_dir = os.path.join(temp_work_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


class MockModel(nn.Module):
    """Minimal model for unit testing monotonicity"""
    def __init__(self, input_size=128, hidden_size=512, output_size=128):
        super().__init__()
        # FFN structure
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # dense_h_to_4h
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)   # dense_4h_to_h
        )
    
    def forward(self, x):
        return self.mlp(x)


@pytest.fixture
def mock_ffn_model():
    """Create a simple FFN model for testing"""
    return MockModel(input_size=64, hidden_size=256, output_size=64)


@pytest.fixture
def sample_weights():
    """Generate sample weight matrices for testing"""
    return {
        'positive': torch.randn(10, 10).abs(),  # All positive
        'mixed': torch.randn(10, 10),           # Mixed pos/neg
        'negative': -torch.randn(10, 10).abs(), # All negative
    }
