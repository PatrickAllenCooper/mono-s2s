#!/usr/bin/env python3
"""
Enhanced Checkpoint Manager for Long-Running Jobs

Handles:
- Time-based checkpointing (every N minutes)
- Step-based checkpointing (every N steps)
- Automatic cleanup of old checkpoints
- Resume detection and loading
- Timeout-safe checkpoint saving

Critical for 24+ hour jobs that may timeout and need resubmission.
"""

import os
import time
import glob
import torch
from datetime import datetime, timedelta


class CheckpointManager:
    """
    Manages checkpoints with automatic cleanup and resume capability.
    
    Features:
    - Saves checkpoints every N steps OR every N minutes (whichever comes first)
    - Keeps only last K checkpoints to save disk space
    - Handles resume automatically
    - Time-based saving ensures we don't lose too much progress on timeout
    """
    
    def __init__(self, checkpoint_dir, keep_last_n=3,
                 save_every_n_steps=500, save_every_n_minutes=30,
                 prefix="checkpoint"):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep (older deleted)
            save_every_n_steps: Save checkpoint every N training steps
            save_every_n_minutes: Save checkpoint every N minutes (timeout protection)
            prefix: Prefix for checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_minutes = save_every_n_minutes
        self.prefix = prefix
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track when we last saved
        self.last_save_time = time.time()
        self.last_save_step = 0
        
        print(f"CheckpointManager initialized:")
        print(f"  Directory: {checkpoint_dir}")
        print(f"  Keep last {keep_last_n} checkpoints")
        print(f"  Save every {save_every_n_steps} steps OR {save_every_n_minutes} minutes")
    
    def should_save(self, current_step):
        """
        Check if we should save a checkpoint now.
        
        Returns True if:
        - current_step is a multiple of save_every_n_steps, OR
        - save_every_n_minutes has elapsed since last save
        """
        # Step-based trigger
        if current_step - self.last_save_step >= self.save_every_n_steps:
            return True
        
        # Time-based trigger (critical for timeout protection)
        elapsed_minutes = (time.time() - self.last_save_time) / 60.0
        if elapsed_minutes >= self.save_every_n_minutes:
            return True
        
        return False
    
    def save_checkpoint(self, state_dict, step, epoch=None, **extra_info):
        """
        Save checkpoint with given state.
        
        Args:
            state_dict: Dictionary with model/optimizer/scheduler states
            step: Current training step (for filename)
            epoch: Optional epoch number
            **extra_info: Any additional info to save (e.g., best_metric)
        
        Returns:
            Path to saved checkpoint
        """
        # Create filename with step and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if epoch is not None:
            filename = f"{self.prefix}_epoch{epoch}_step{step}_{timestamp}.pt"
        else:
            filename = f"{self.prefix}_step{step}_{timestamp}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Add metadata
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp,
            'save_time_unix': time.time(),
            **state_dict,
            **extra_info
        }
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        
        # Update tracking
        self.last_save_time = time.time()
        self.last_save_step = step
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"  [Checkpoint] Saved: {filename}")
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        pattern = os.path.join(self.checkpoint_dir, f"{self.prefix}_*.pt")
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        # Keep best_model.pt if it exists
        checkpoints = [c for c in checkpoints if not c.endswith('best_model.pt')]
        
        # Remove old checkpoints
        if len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[:-self.keep_last_n]
            for ckpt in to_remove:
                try:
                    os.remove(ckpt)
                    print(f"  [Cleanup] Removed old checkpoint: {os.path.basename(ckpt)}")
                except OSError:
                    pass  # Ignore errors
    
    def load_latest_checkpoint(self, device='cpu'):
        """
        Load the most recent checkpoint.
        
        Returns:
            checkpoint dict if found, None otherwise
        """
        pattern = os.path.join(self.checkpoint_dir, f"{self.prefix}_*.pt")
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        # Exclude best_model.pt from resume (it's for final evaluation)
        checkpoints = [c for c in checkpoints if not c.endswith('best_model.pt')]
        
        if not checkpoints:
            print("  [Resume] No checkpoint found, starting from scratch")
            return None
        
        latest_checkpoint = checkpoints[-1]
        print(f"  [Resume] Loading checkpoint: {os.path.basename(latest_checkpoint)}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            
            # Print resume info
            step = checkpoint.get('step', 'unknown')
            epoch = checkpoint.get('epoch', 'unknown')
            timestamp = checkpoint.get('timestamp', 'unknown')
            
            print(f"  [Resume] Checkpoint info:")
            print(f"    Step: {step}")
            print(f"    Epoch: {epoch}")
            print(f"    Saved: {timestamp}")
            
            return checkpoint
            
        except Exception as e:
            print(f"  [Resume] Error loading checkpoint: {e}")
            return None
    
    def save_best_model(self, model_state_dict, metric_value, metric_name="metric"):
        """
        Save best model separately (not subject to cleanup).
        
        Args:
            model_state_dict: Model state dict to save
            metric_value: Value of the metric (e.g., validation loss)
            metric_name: Name of the metric for logging
        """
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        
        save_dict = {
            'model_state_dict': model_state_dict,
            f'best_{metric_name}': metric_value,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        
        torch.save(save_dict, best_path)
        print(f"  [Best Model] Saved with {metric_name}={metric_value:.4f}")
    
    def get_resume_info(self):
        """
        Get information about available checkpoint for resume.
        
        Returns:
            Dictionary with resume info or None if no checkpoint exists
        """
        checkpoint = self.load_latest_checkpoint()
        if checkpoint is None:
            return None
        
        return {
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'checkpoint_path': os.path.join(
                self.checkpoint_dir,
                f"{self.prefix}_step{checkpoint.get('step', 0)}_*.pt"
            )
        }


class TrainingTimer:
    """
    Helps track if we're approaching job time limit.
    
    Useful for gracefully stopping before SLURM kills the job,
    ensuring we have time to save a final checkpoint.
    """
    
    def __init__(self, max_runtime_minutes=None, warning_minutes=10):
        """
        Args:
            max_runtime_minutes: Maximum job runtime (from SLURM time limit)
            warning_minutes: Issue warning when this many minutes remain
        """
        self.start_time = time.time()
        self.max_runtime_minutes = max_runtime_minutes
        self.warning_minutes = warning_minutes
        self.warning_issued = False
    
    def elapsed_minutes(self):
        """Get elapsed time in minutes."""
        return (time.time() - self.start_time) / 60.0
    
    def remaining_minutes(self):
        """Get remaining time in minutes (if max_runtime set)."""
        if self.max_runtime_minutes is None:
            return float('inf')
        return self.max_runtime_minutes - self.elapsed_minutes()
    
    def should_stop(self):
        """
        Check if we should stop training to save final checkpoint.
        
        Returns True if we're within warning_minutes of the time limit.
        """
        if self.max_runtime_minutes is None:
            return False
        
        remaining = self.remaining_minutes()
        
        if remaining <= self.warning_minutes and not self.warning_issued:
            print(f"\n⚠️  WARNING: Only {remaining:.1f} minutes remaining!")
            print(f"⚠️  Approaching job time limit. Will save checkpoint and exit gracefully.")
            self.warning_issued = True
        
        return remaining <= self.warning_minutes
    
    def get_status(self):
        """Get human-readable status string."""
        elapsed = self.elapsed_minutes()
        
        if self.max_runtime_minutes:
            remaining = self.remaining_minutes()
            return f"Elapsed: {elapsed:.1f}m, Remaining: {remaining:.1f}m"
        else:
            return f"Elapsed: {elapsed:.1f}m"


if __name__ == "__main__":
    # Test checkpoint manager
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Testing CheckpointManager...")
        
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            keep_last_n=3,
            save_every_n_steps=10,
            save_every_n_minutes=1
        )
        
        # Simulate training
        for step in range(50):
            if manager.should_save(step):
                state = {
                    'model_state_dict': {'dummy': step},
                    'optimizer_state_dict': {},
                }
                manager.save_checkpoint(state, step, epoch=step // 10)
            
            time.sleep(0.1)  # Simulate training step
        
        # Test resume
        print("\nTesting resume...")
        checkpoint = manager.load_latest_checkpoint()
        if checkpoint:
            print(f"Resumed from step: {checkpoint['step']}")
        
        # Test best model save
        manager.save_best_model({'dummy': 'best'}, 0.95, metric_name="accuracy")
        
        print("\nCheckpoint manager test complete!")
