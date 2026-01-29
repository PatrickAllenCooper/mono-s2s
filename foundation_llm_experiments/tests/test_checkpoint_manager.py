#!/usr/bin/env python3
"""
Tests for CheckpointManager and TrainingTimer

Critical functionality for long-running jobs with timeout handling.
"""

import os
import sys
import time
import tempfile
import torch
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.checkpoint_manager import CheckpointManager, TrainingTimer


class TestCheckpointManager:
    """Test suite for CheckpointManager"""
    
    def test_initialization(self, tmp_path):
        """Test CheckpointManager initialization"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            keep_last_n=3,
            save_every_n_steps=10,
            save_every_n_minutes=1,
            prefix="test_ckpt"
        )
        
        assert manager.checkpoint_dir == str(tmp_path)
        assert manager.keep_last_n == 3
        assert manager.save_every_n_steps == 10
        assert manager.save_every_n_minutes == 1
        assert manager.prefix == "test_ckpt"
        assert os.path.exists(tmp_path)
    
    def test_should_save_step_based(self, tmp_path):
        """Test step-based checkpoint trigger"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            save_every_n_steps=10,
            save_every_n_minutes=1000  # Very long to avoid time trigger
        )
        
        # Should trigger at multiples of save_every_n_steps
        assert manager.should_save(0) == False
        assert manager.should_save(5) == False
        assert manager.should_save(10) == True
        
        # Update tracking
        manager.last_save_step = 10
        
        assert manager.should_save(15) == False
        assert manager.should_save(20) == True
    
    def test_should_save_time_based(self, tmp_path):
        """Test time-based checkpoint trigger"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            save_every_n_steps=1000,  # Very high to avoid step trigger
            save_every_n_minutes=0.01  # 0.6 seconds
        )
        
        # Should not trigger immediately
        assert manager.should_save(1) == False
        
        # Wait for time trigger
        time.sleep(1.0)  # 1 second > 0.6 seconds
        assert manager.should_save(2) == True
    
    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            keep_last_n=3,
            prefix="test"
        )
        
        # Create dummy state
        state = {
            'model_state_dict': {'layer1': torch.randn(10, 10)},
            'optimizer_state_dict': {'param_groups': []},
            'scheduler_state_dict': {},
        }
        
        # Save checkpoint
        ckpt_path = manager.save_checkpoint(
            state_dict=state,
            step=100,
            epoch=1,
            best_metric=0.95
        )
        
        # Verify file exists
        assert os.path.exists(ckpt_path)
        assert 'test_epoch1_step100' in ckpt_path
        
        # Load and verify contents
        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded['step'] == 100
        assert loaded['epoch'] == 1
        assert loaded['best_metric'] == 0.95
        assert 'model_state_dict' in loaded
        assert 'timestamp' in loaded
    
    def test_checkpoint_cleanup(self, tmp_path):
        """Test automatic cleanup of old checkpoints"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            keep_last_n=3,
            prefix="test"
        )
        
        state = {'model_state_dict': {}}
        
        # Create 5 checkpoints
        for i in range(5):
            manager.save_checkpoint(state, step=i*100, epoch=i)
            time.sleep(0.1)  # Ensure different timestamps
        
        # Should only keep last 3
        checkpoints = sorted(
            [f for f in os.listdir(tmp_path) if f.startswith('test_')],
            key=lambda x: os.path.getmtime(os.path.join(tmp_path, x))
        )
        
        assert len(checkpoints) == 3
        
        # Verify we kept the most recent ones (steps 200, 300, 400)
        loaded = torch.load(os.path.join(tmp_path, checkpoints[-1]), weights_only=False)
        assert loaded['step'] == 400
    
    def test_load_latest_checkpoint_empty(self, tmp_path):
        """Test loading when no checkpoint exists"""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        
        result = manager.load_latest_checkpoint()
        assert result is None
    
    def test_load_latest_checkpoint(self, tmp_path):
        """Test loading the most recent checkpoint"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            prefix="test"
        )
        
        # Create multiple checkpoints
        for i in range(3):
            state = {'model_state_dict': {}, 'value': i}
            manager.save_checkpoint(state, step=i*100, epoch=i)
            time.sleep(0.1)
        
        # Load latest
        loaded = manager.load_latest_checkpoint()
        
        assert loaded is not None
        assert loaded['step'] == 200  # Last checkpoint
        assert loaded['epoch'] == 2
        assert loaded['value'] == 2
    
    def test_save_best_model(self, tmp_path):
        """Test saving best model separately"""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        
        model_state = {'layer1': torch.randn(5, 5)}
        
        manager.save_best_model(
            model_state_dict=model_state,
            metric_value=0.987,
            metric_name="accuracy"
        )
        
        # Verify best_model.pt exists
        best_path = os.path.join(tmp_path, 'best_model.pt')
        assert os.path.exists(best_path)
        
        # Verify contents
        loaded = torch.load(best_path, weights_only=False)
        assert 'model_state_dict' in loaded
        assert loaded['best_accuracy'] == 0.987
        assert 'timestamp' in loaded
    
    def test_best_model_not_cleaned_up(self, tmp_path):
        """Test that best_model.pt is not deleted during cleanup"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            keep_last_n=2,
            prefix="test"
        )
        
        # Save best model
        manager.save_best_model({'dummy': 1}, metric_value=0.95)
        
        # Create multiple regular checkpoints (should trigger cleanup)
        state = {'model_state_dict': {}}
        for i in range(5):
            manager.save_checkpoint(state, step=i*100)
            time.sleep(0.1)
        
        # best_model.pt should still exist
        best_path = os.path.join(tmp_path, 'best_model.pt')
        assert os.path.exists(best_path)
        
        # But only 2 regular checkpoints should remain
        regular_ckpts = [f for f in os.listdir(tmp_path) 
                        if f.startswith('test_') and f != 'best_model.pt']
        assert len(regular_ckpts) == 2
    
    def test_get_resume_info(self, tmp_path):
        """Test getting resume information"""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        
        # No checkpoint
        info = manager.get_resume_info()
        assert info is None
        
        # Create checkpoint
        state = {'model_state_dict': {}}
        manager.save_checkpoint(state, step=250, epoch=2)
        
        # Get info
        info = manager.get_resume_info()
        assert info is not None
        assert info['step'] == 250
        assert info['epoch'] == 2
        assert 'timestamp' in info


class TestTrainingTimer:
    """Test suite for TrainingTimer"""
    
    def test_initialization(self):
        """Test TrainingTimer initialization"""
        timer = TrainingTimer(max_runtime_minutes=60, warning_minutes=10)
        
        assert timer.max_runtime_minutes == 60
        assert timer.warning_minutes == 10
        assert timer.warning_issued == False
    
    def test_elapsed_minutes(self):
        """Test elapsed time calculation"""
        timer = TrainingTimer()
        
        time.sleep(0.1)  # 0.1 seconds
        elapsed = timer.elapsed_minutes()
        
        # Should be roughly 0.1/60 = 0.00167 minutes
        assert 0.001 < elapsed < 0.01
    
    def test_remaining_minutes_no_limit(self):
        """Test remaining time with no limit set"""
        timer = TrainingTimer(max_runtime_minutes=None)
        
        remaining = timer.remaining_minutes()
        assert remaining == float('inf')
    
    def test_remaining_minutes_with_limit(self):
        """Test remaining time calculation with limit"""
        timer = TrainingTimer(max_runtime_minutes=60)
        
        time.sleep(0.1)
        remaining = timer.remaining_minutes()
        
        # Should be close to 60 minutes
        assert 59.9 < remaining < 60.0
    
    def test_should_stop_no_limit(self):
        """Test should_stop with no time limit"""
        timer = TrainingTimer(max_runtime_minutes=None, warning_minutes=10)
        
        assert timer.should_stop() == False
    
    def test_should_stop_with_time_remaining(self):
        """Test should_stop when plenty of time remains"""
        timer = TrainingTimer(max_runtime_minutes=60, warning_minutes=10)
        
        assert timer.should_stop() == False
        assert timer.warning_issued == False
    
    def test_should_stop_approaching_limit(self):
        """Test should_stop when approaching time limit"""
        # Set limit very close to elapsed time
        timer = TrainingTimer(max_runtime_minutes=0.01, warning_minutes=0.02)
        
        time.sleep(0.1)  # Sleep to ensure we're past the limit
        
        should_stop = timer.should_stop()
        assert should_stop == True
        assert timer.warning_issued == True
    
    def test_get_status_no_limit(self):
        """Test status string with no limit"""
        timer = TrainingTimer(max_runtime_minutes=None)
        
        time.sleep(0.1)
        status = timer.get_status()
        
        assert 'Elapsed:' in status
        assert 'Remaining' not in status
    
    def test_get_status_with_limit(self):
        """Test status string with limit"""
        timer = TrainingTimer(max_runtime_minutes=60, warning_minutes=10)
        
        time.sleep(0.1)
        status = timer.get_status()
        
        assert 'Elapsed:' in status
        assert 'Remaining:' in status


class TestCheckpointManagerIntegration:
    """Integration tests for CheckpointManager with realistic scenarios"""
    
    def test_resume_from_checkpoint(self, tmp_path):
        """Test full save and resume workflow"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            prefix="training"
        )
        
        # Simulate training session 1
        model_state = {'weights': torch.randn(10, 10)}
        optimizer_state = {'lr': 0.001}
        
        manager.save_checkpoint(
            state_dict={
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
            },
            step=500,
            epoch=1,
            best_val_loss=0.25
        )
        
        # Simulate training session 2 (resume after timeout)
        manager2 = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            prefix="training"
        )
        
        checkpoint = manager2.load_latest_checkpoint()
        
        assert checkpoint is not None
        assert checkpoint['step'] == 500
        assert checkpoint['epoch'] == 1
        assert checkpoint['best_val_loss'] == 0.25
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
    
    def test_multiple_checkpoint_saves_with_cleanup(self, tmp_path):
        """Test realistic training with multiple saves and cleanup"""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            keep_last_n=3,
            save_every_n_steps=100,
            prefix="train"
        )
        
        # Simulate 1000 training steps
        for step in range(0, 1001, 100):
            if manager.should_save(step):
                state = {
                    'model_state_dict': {'dummy': step},
                    'optimizer_state_dict': {},
                }
                manager.save_checkpoint(state, step=step, epoch=step//100)
        
        # Should only have last 3 checkpoints
        checkpoints = [f for f in os.listdir(tmp_path) if f.startswith('train_')]
        assert len(checkpoints) == 3
        
        # Most recent should be step 1000
        latest = manager.load_latest_checkpoint()
        assert latest['step'] == 1000
    
    def test_timeout_scenario_simulation(self, tmp_path):
        """Test checkpoint behavior during timeout scenario"""
        # Session 1: Training interrupted at step 750
        manager1 = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            save_every_n_steps=250,
            prefix="interrupted"
        )
        
        # Save checkpoints at steps 0, 250, 500, 750
        for step in [0, 250, 500, 750]:
            state = {'model_state_dict': {'step': step}}
            manager1.save_checkpoint(state, step=step)
        
        # Session 2: Resume after resubmission
        manager2 = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            save_every_n_steps=250,
            prefix="interrupted"
        )
        
        checkpoint = manager2.load_latest_checkpoint()
        resume_step = checkpoint['step']
        
        assert resume_step == 750  # Resume from last checkpoint
        
        # Continue training from resume_step
        for step in [1000]:  # Next checkpoint
            state = {'model_state_dict': {'step': step}}
            manager2.save_checkpoint(state, step=step)
        
        # Verify we can load the continued checkpoint
        final_checkpoint = manager2.load_latest_checkpoint()
        assert final_checkpoint['step'] == 1000


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for testing"""
    return tmp_path_factory.mktemp("checkpoints")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
