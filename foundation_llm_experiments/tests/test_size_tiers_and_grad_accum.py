"""Pythia size-tier presets and gradient-accumulation math."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import optimizer_step_count, prune_epoch_checkpoints


def test_default_model_is_pythia_14b():
    assert Config.MODEL_NAME == "EleutherAI/pythia-1.4b"
    assert Config.HIDDEN_SIZE == 2048
    assert Config.NUM_LAYERS == 24


def test_larger_arch_tables():
    assert Config._PYTHIA_ARCH["EleutherAI/pythia-2.8b"]["HIDDEN_SIZE"] == 2560
    assert Config._PYTHIA_ARCH["EleutherAI/pythia-6.9b"]["NUM_LAYERS"] == 32
    assert Config._PYTHIA_SIZE_TIERS["EleutherAI/pythia-6.9b"]["GRADIENT_ACCUMULATION_STEPS"] == 16


def test_optimizer_step_count_and_prune(tmp_path):
    assert optimizer_step_count(32, 8, 2) == 8
    for epoch in (0, 1, 2):
        (tmp_path / f"checkpoint_epoch_{epoch}.pt").write_bytes(b"x")
    removed = prune_epoch_checkpoints(str(tmp_path), keep_last_n=1)
    assert len(removed) == 2
    assert (tmp_path / "checkpoint_epoch_2.pt").exists()
