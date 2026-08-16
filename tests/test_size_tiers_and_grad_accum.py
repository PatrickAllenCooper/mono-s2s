"""Size-tier presets, gradient-accumulation math, and checkpoint pruning."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import optimizer_step_count, prune_epoch_checkpoints, training_autocast


class TestT5SizeTiers:
    def test_default_is_t5_small_legacy_hparams(self):
        assert ExperimentConfig.MODEL_NAME == "t5-small"
        assert ExperimentConfig.BATCH_SIZE == 4
        assert ExperimentConfig.GRADIENT_ACCUMULATION_STEPS == 1
        assert ExperimentConfig.USE_BF16 is False
        assert ExperimentConfig.USE_GRADIENT_CHECKPOINTING is False
        assert ExperimentConfig.EVAL_BATCH_SIZE == 8

    def test_larger_tiers_enable_memory_hygiene(self):
        large = ExperimentConfig._T5_SIZE_TIERS["t5-large"]
        assert large["USE_BF16"] is True
        assert large["USE_GRADIENT_CHECKPOINTING"] is True
        assert large["GRADIENT_ACCUMULATION_STEPS"] >= 4
        assert large["BATCH_SIZE"] <= 2

    def test_full_test_sets_default_on(self):
        assert ExperimentConfig.USE_FULL_TEST_SETS is True

    def test_five_seeds_listed(self):
        assert ExperimentConfig.RANDOM_SEEDS == [42, 1337, 2024, 8888, 12345]


class TestOptimizerStepCount:
    def test_accum_one_matches_batch_count(self):
        assert optimizer_step_count(10, 1, 3) == 30

    def test_accum_divides_batches(self):
        assert optimizer_step_count(10, 2, 1) == 5

    def test_leftover_microbatch_still_steps(self):
        assert optimizer_step_count(11, 2, 1) == 6


class TestPruneEpochCheckpoints:
    def test_keeps_newest_and_best(self, tmp_path):
        for epoch in (1, 2, 3, 4):
            (tmp_path / f"checkpoint_epoch_{epoch}.pt").write_bytes(b"x")
        (tmp_path / "best_model.pt").write_bytes(b"best")
        removed = prune_epoch_checkpoints(str(tmp_path), keep_last_n=1)
        assert len(removed) == 3
        assert (tmp_path / "checkpoint_epoch_4.pt").exists()
        assert (tmp_path / "best_model.pt").exists()
        assert not (tmp_path / "checkpoint_epoch_1.pt").exists()

    def test_missing_dir_is_noop(self, tmp_path):
        assert prune_epoch_checkpoints(str(tmp_path / "nope"), keep_last_n=1) == []


class TestTrainingAutocast:
    def test_cpu_returns_nullcontext(self):
        class _Dev:
            type = "cpu"
        with training_autocast(_Dev(), use_bf16=True):
            pass
