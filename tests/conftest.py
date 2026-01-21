"""
Pytest Configuration and Fixtures

Shared test fixtures and configuration for the entire test suite.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hpc_version"))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory"""
    return PROJECT_ROOT


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files"""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_work_dir(temp_dir, monkeypatch):
    """Create a temporary work directory and patch config"""
    work_dir = os.path.join(temp_dir, "work")
    results_dir = os.path.join(temp_dir, "results")
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")
    data_cache_dir = os.path.join(temp_dir, "data_cache")
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_cache_dir, exist_ok=True)
    
    # Patch ExperimentConfig paths
    from hpc_version.configs.experiment_config import ExperimentConfig
    monkeypatch.setattr(ExperimentConfig, "WORK_DIR", work_dir)
    monkeypatch.setattr(ExperimentConfig, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(ExperimentConfig, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(ExperimentConfig, "DATA_CACHE_DIR", data_cache_dir)
    
    yield {
        "work_dir": work_dir,
        "results_dir": results_dir,
        "checkpoint_dir": checkpoint_dir,
        "data_cache_dir": data_cache_dir
    }


@pytest.fixture(scope="function")
def mock_model():
    """Create a minimal mock T5 model for testing"""
    from transformers import T5Config, T5ForConditionalGeneration
    
    # Create minimal config
    config = T5Config(
        vocab_size=100,
        d_model=64,
        d_kv=8,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
    )
    
    model = T5ForConditionalGeneration(config)
    return model


@pytest.fixture(scope="function")
def mock_tokenizer():
    """Create a minimal mock tokenizer for testing"""
    from transformers import T5Tokenizer
    # Use a tiny model for fast testing
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return tokenizer


@pytest.fixture(scope="function")
def sample_texts():
    """Sample texts for testing"""
    return [
        "This is a sample document. It contains some text.",
        "Another document with different content.",
        "A third document for testing purposes."
    ]


@pytest.fixture(scope="function")
def sample_summaries():
    """Sample summaries for testing"""
    return [
        "Sample summary.",
        "Different summary.",
        "Third summary."
    ]


@pytest.fixture(scope="function")
def mock_dataset(sample_texts, sample_summaries):
    """Create a mock dataset"""
    return {
        'texts': sample_texts,
        'summaries': sample_summaries
    }


@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset random seeds before each test"""
    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)


@pytest.fixture(scope="function")
def mock_huggingface_dataset(monkeypatch):
    """Mock HuggingFace datasets.load_dataset function"""
    class MockDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    def mock_load_dataset(name, *args, **kwargs):
        # Return mock data based on dataset name
        if "cnn_dailymail" in name:
            return MockDataset([
                {"article": f"Article {i}", "highlights": f"Summary {i}"}
                for i in range(10)
            ])
        elif "xsum" in name:
            return MockDataset([
                {"document": f"Document {i}", "summary": f"Summary {i}"}
                for i in range(10)
            ])
        elif "samsum" in name:
            return MockDataset([
                {"dialogue": f"Dialogue {i}", "summary": f"Summary {i}"}
                for i in range(10)
            ])
        else:
            return MockDataset([
                {"text": f"Text {i}", "summary": f"Summary {i}"}
                for i in range(10)
            ])
    
    return mock_load_dataset


@pytest.fixture(scope="function")
def gpu_available():
    """Check if GPU is available"""
    return torch.cuda.is_available()


@pytest.fixture(scope="function")
def device():
    """Return appropriate device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mark GPU tests to skip if no GPU available
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if GPU not available"""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)
