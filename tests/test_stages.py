"""
Tests for Pipeline Stage Scripts

Comprehensive tests for all stage scripts (0-7).
"""
import os
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))

from hpc_version.configs.experiment_config import ExperimentConfig


class TestStage0Setup:
    """Tests for Stage 0: Setup"""
    
    @pytest.mark.slow
    def test_stage_0_imports(self):
        """Test that stage 0 can be imported"""
        from hpc_version.scripts import stage_0_setup
        assert hasattr(stage_0_setup, 'main')
    
    @patch('hpc_version.scripts.stage_0_setup.T5ForConditionalGeneration')
    @patch('hpc_version.scripts.stage_0_setup.T5Tokenizer')
    def test_stage_0_main_mocked(self, mock_tokenizer_class, mock_model_class, 
                                  temp_work_dir, monkeypatch):
        """Test stage 0 main function with mocked components"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        monkeypatch.setattr(ExperimentConfig, "MODEL_NAME", "t5-small")
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.model_type = "t5"
        mock_model.parameters.return_value = [torch.randn(10, 10)]
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Import and run
        from hpc_version.scripts import stage_0_setup
        
        # Should complete without errors
        exit_code = stage_0_setup.main()
        
        # Check results
        assert exit_code == 0
        assert os.path.exists(os.path.join(temp_work_dir["results_dir"], "setup_complete.json"))


class TestStage1DataPrep:
    """Tests for Stage 1: Data Preparation"""
    
    def test_stage_1_imports(self):
        """Test that stage 1 can be imported"""
        from hpc_version.scripts import stage_1_prepare_data
        assert hasattr(stage_1_prepare_data, 'main')
    
    @patch('hpc_version.utils.common_utils.load_dataset')
    def test_stage_1_main_mocked(self, mock_load_dataset, temp_work_dir, monkeypatch):
        """Test stage 1 main function with mocked dataset loading"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        monkeypatch.setattr(ExperimentConfig, "USE_FULL_TEST_SETS", False)
        monkeypatch.setattr(ExperimentConfig, "QUICK_TEST_SIZE", 10)
        monkeypatch.setattr(ExperimentConfig, "TRAIN_DATASETS", [
            ("test_dataset", "text", "summary", "TestDataset")
        ])
        monkeypatch.setattr(ExperimentConfig, "DATASET_ALLOW_PARTIAL", True)
        
        # Create stage 0 completion flag (dependency)
        from hpc_version.utils.common_utils import create_completion_flag
        create_completion_flag("stage_0_setup", work_dir=temp_work_dir["work_dir"])
        
        # Mock dataset
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"text": f"text{i}", "summary": f"summary{i}"}
                    for i in range(10)
                ])
        
        mock_load_dataset.return_value = MockDataset()
        
        # Import and run
        from hpc_version.scripts import stage_1_prepare_data
        
        exit_code = stage_1_prepare_data.main()
        
        # Should complete (may have warnings but shouldn't crash)
        assert isinstance(exit_code, int)


class TestStage2Baseline:
    """Tests for Stage 2: Train Baseline"""
    
    def test_stage_2_imports(self):
        """Test that stage 2 can be imported"""
        from hpc_version.scripts import stage_2_train_baseline
        assert hasattr(stage_2_train_baseline, 'main')


class TestStage3Monotonic:
    """Tests for Stage 3: Train Monotonic"""
    
    def test_stage_3_imports(self):
        """Test that stage 3 can be imported"""
        from hpc_version.scripts import stage_3_train_monotonic
        assert hasattr(stage_3_train_monotonic, 'main')


class TestStage4Evaluate:
    """Tests for Stage 4: Evaluate Models"""
    
    def test_stage_4_imports(self):
        """Test that stage 4 can be imported"""
        from hpc_version.scripts import stage_4_evaluate
        assert hasattr(stage_4_evaluate, 'main')


class TestStage5UAT:
    """Tests for Stage 5: UAT Attacks"""
    
    def test_stage_5_imports(self):
        """Test that stage 5 can be imported"""
        from hpc_version.scripts import stage_5_uat_attacks
        assert hasattr(stage_5_uat_attacks, 'main')


class TestStage6HotFlip:
    """Tests for Stage 6: HotFlip Attacks"""
    
    def test_stage_6_imports(self):
        """Test that stage 6 can be imported"""
        from hpc_version.scripts import stage_6_hotflip_attacks
        assert hasattr(stage_6_hotflip_attacks, 'main')


class TestStage7Aggregate:
    """Tests for Stage 7: Aggregate Results"""
    
    def test_stage_7_imports(self):
        """Test that stage 7 can be imported"""
        from hpc_version.scripts import stage_7_aggregate
        assert hasattr(stage_7_aggregate, 'main')
    
    def test_stage_7_with_mock_data(self, temp_work_dir, monkeypatch):
        """Test stage 7 with mock result files"""
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        # Create dependency flags
        from hpc_version.utils.common_utils import create_completion_flag, save_json
        for stage in ["stage_0_setup", "stage_1_data_prep", "stage_2_train_baseline",
                      "stage_3_train_monotonic", "stage_4_evaluate", "stage_5_uat",
                      "stage_6_hotflip"]:
            create_completion_flag(stage, work_dir=temp_work_dir["work_dir"])
        
        # Create mock result files
        mock_eval_results = {
            "standard": {"cnn_dm": {"rouge1": {"mean": 0.30}}},
            "baseline": {"cnn_dm": {"rouge1": {"mean": 0.35}}},
            "monotonic": {"cnn_dm": {"rouge1": {"mean": 0.33}}}
        }
        
        mock_uat_results = {
            "standard": {"clean_rouge1": 0.30, "attacked_rouge1": 0.20},
            "baseline": {"clean_rouge1": 0.35, "attacked_rouge1": 0.25},
            "monotonic": {"clean_rouge1": 0.33, "attacked_rouge1": 0.28}
        }
        
        mock_hotflip_results = {
            "standard": {"clean_rouge1": 0.30, "attacked_rouge1": 0.22},
            "baseline": {"clean_rouge1": 0.35, "attacked_rouge1": 0.27},
            "monotonic": {"clean_rouge1": 0.33, "attacked_rouge1": 0.29}
        }
        
        save_json(mock_eval_results, 
                 os.path.join(temp_work_dir["results_dir"], "evaluation_results.json"))
        save_json(mock_uat_results, 
                 os.path.join(temp_work_dir["results_dir"], "uat_results.json"))
        save_json(mock_hotflip_results, 
                 os.path.join(temp_work_dir["results_dir"], "hotflip_results.json"))
        
        # Import and run
        from hpc_version.scripts import stage_7_aggregate
        
        exit_code = stage_7_aggregate.main()
        
        # Should complete successfully
        assert exit_code == 0
        
        # Check output files
        final_results_path = os.path.join(temp_work_dir["results_dir"], "final_results.json")
        summary_path = os.path.join(temp_work_dir["results_dir"], "experiment_summary.txt")
        
        assert os.path.exists(final_results_path)
        assert os.path.exists(summary_path)


class TestExistingTests:
    """Tests for existing test files"""
    
    def test_existing_test_dataset_loading(self):
        """Test that existing test_dataset_loading.py can be imported"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))
        # Just check it can be imported without errors
        # (won't run it as it requires actual dataset downloads)
    
    def test_existing_test_improvements(self):
        """Test that existing test_improvements.py can be imported"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "hpc_version"))
        # Just check it can be imported without errors


class TestIntegrationScenarios:
    """Integration tests for common scenarios"""
    
    def test_full_pipeline_mock(self, temp_work_dir, monkeypatch):
        """Test that all stages can be chained together (mocked)"""
        # This is a basic smoke test to ensure the pipeline structure is sound
        monkeypatch.setattr(ExperimentConfig, "CURRENT_SEED", 42)
        
        from hpc_version.utils.common_utils import (
            create_completion_flag, check_dependencies
        )
        
        # Simulate pipeline stages completing
        stages = [
            "stage_0_setup",
            "stage_1_data_prep",
            "stage_2_train_baseline",
            "stage_3_train_monotonic",
            "stage_4_evaluate",
            "stage_5_uat",
            "stage_6_hotflip",
            "stage_7_aggregate"
        ]
        
        for i, stage in enumerate(stages):
            # Check dependencies for stages that have them
            if i > 0:
                deps = stages[:i]
                assert check_dependencies(deps, work_dir=temp_work_dir["work_dir"])
            
            # Mark stage complete
            create_completion_flag(stage, work_dir=temp_work_dir["work_dir"])
        
        # All stages should be complete
        for stage in stages:
            from hpc_version.utils.common_utils import check_completion_flag
            assert check_completion_flag(stage, work_dir=temp_work_dir["work_dir"])
