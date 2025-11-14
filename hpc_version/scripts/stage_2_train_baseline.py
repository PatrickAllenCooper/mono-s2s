#!/usr/bin/env python3
"""
Stage 2: Train Baseline Model (Unconstrained T5)

This stage trains the FAIR BASELINE model - an unconstrained T5 model
fine-tuned with the SAME data and hyperparameters as the monotonic model.

This addresses the critical methodological issue of comparing monotonic models
to pre-trained (not fine-tuned) models.

Inputs:
- train_data.pt (from stage 1)
- val_data.pt (from stage 1)
- Standard T5 checkpoint (downloads if needed)

Outputs:
- baseline_checkpoints/best_model.pt (best model weights)
- baseline_checkpoints/checkpoint_epoch_*.pt (epoch checkpoints)
- baseline_training_history.json (loss curves)
- stage_2_train_baseline_complete.flag
"""

# Set environment variables BEFORE importing torch
import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, create_completion_flag, check_dependencies,
    save_json, StageLogger, worker_init_fn, get_generator
)

# Import transformers AFTER environment setup
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW


class BaselineT5Trainer:
    """
    Trainer for baseline (unconstrained) T5 model.
    Uses identical hyperparameters as monotonic trainer for fair comparison.
    """
    def __init__(self, model, train_loader, val_loader, device, 
                 checkpoint_dir, history_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.history_path = history_path
        
        # Use config hyperparameters
        self.learning_rate = ExperimentConfig.LEARNING_RATE
        self.weight_decay = ExperimentConfig.WEIGHT_DECAY
        self.num_epochs = ExperimentConfig.NUM_EPOCHS
        self.max_grad_norm = ExperimentConfig.MAX_GRAD_NORM
        self.warmup_ratio = ExperimentConfig.WARMUP_RATIO
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Attempt to load checkpoint
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load latest checkpoint if it exists"""
        if not os.path.exists(self.checkpoint_dir):
            print("\nNo checkpoint directory found. Starting from scratch.")
            return
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        
        if not checkpoints:
            print("\nNo checkpoint found. Starting training from epoch 0.")
            return
        
        # Get latest checkpoint
        epochs = [int(f.replace('checkpoint_epoch_', '').replace('.pt', '')) 
                 for f in checkpoints]
        latest_epoch = max(epochs)
        latest_checkpoint = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{latest_epoch}.pt'
        )
        
        print(f"\n🔄 Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load history
        if os.path.exists(self.history_path):
            history = torch.load(self.history_path)
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
        
        print(f"✓ Resuming from epoch {self.start_epoch}")
        print(f"  Best validation loss so far: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
        }
        
        torch.save(save_dict, checkpoint_path)
        print(f"  ✓ Checkpoint saved: epoch_{epoch}.pt")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_path)
            print(f"  ✓ Best model saved: best_model.pt")
        
        # Save history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        torch.save(history, self.history_path)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training Baseline T5")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Full training loop"""
        print(f"\nStarting training from epoch {self.start_epoch + 1}/{self.num_epochs}")
        print("="*80)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-"*80)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ New best validation loss!")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        print("\n" + "="*80)
        print("✓ Training complete!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print("="*80)
        
        return self.train_losses, self.val_losses


def main():
    """Run baseline model training"""
    logger = StageLogger("stage_2_train_baseline")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_0_setup', 'stage_1_data_prep']):
            logger.complete(success=False)
            return 1
        
        # Set seeds
        logger.log("Setting random seeds...")
        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        
        # Get device
        device = ExperimentConfig.get_device()
        
        # Load tokenizer
        logger.log("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(ExperimentConfig.MODEL_NAME)
        
        # Load data
        logger.log("Loading training and validation data...")
        data_cache_dir = ExperimentConfig.DATA_CACHE_DIR
        
        train_data = torch.load(os.path.join(data_cache_dir, 'train_data.pt'))
        val_data = torch.load(os.path.join(data_cache_dir, 'val_data.pt'))
        
        logger.log(f"  Training samples: {len(train_data['texts'])}")
        logger.log(f"  Validation samples: {len(val_data['texts'])}")
        
        # Create datasets
        from utils.common_utils import SummarizationDataset
        
        train_dataset = SummarizationDataset(
            train_data['texts'],
            train_data['summaries'],
            tokenizer
        )
        
        val_dataset = SummarizationDataset(
            val_data['texts'],
            val_data['summaries'],
            tokenizer
        )
        
        # Create data loaders with deterministic settings
        generator = get_generator(device='cpu')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=ExperimentConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Set to 0 for full reproducibility
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=ExperimentConfig.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        logger.log("Initializing baseline model (unconstrained T5)...")
        model = T5ForConditionalGeneration.from_pretrained(
            ExperimentConfig.MODEL_NAME
        ).to(device)
        
        # Verify T5 architecture
        assert model.config.model_type == "t5", \
            f"ERROR: Expected T5, got {model.config.model_type}"
        
        logger.log(f"✓ Model initialized: {ExperimentConfig.MODEL_NAME}")
        logger.log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.log(f"  Model type: Baseline (NO monotonic constraints)")
        
        # Setup checkpoint paths
        checkpoint_dir = os.path.join(
            ExperimentConfig.CHECKPOINT_DIR, 
            'baseline_checkpoints'
        )
        history_path = os.path.join(
            ExperimentConfig.RESULTS_DIR,
            'baseline_training_history.json'
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create trainer
        logger.log("Creating trainer...")
        trainer = BaselineT5Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=checkpoint_dir,
            history_path=history_path
        )
        
        # Train
        logger.log("Starting training...")
        logger.log(f"  Epochs: {ExperimentConfig.NUM_EPOCHS}")
        logger.log(f"  Batch size: {ExperimentConfig.BATCH_SIZE}")
        logger.log(f"  Learning rate: {ExperimentConfig.LEARNING_RATE}")
        logger.log(f"  Weight decay: {ExperimentConfig.WEIGHT_DECAY}")
        logger.log(f"  Warmup ratio: {ExperimentConfig.WARMUP_RATIO}")
        logger.log(f"  Max grad norm: {ExperimentConfig.MAX_GRAD_NORM}")
        
        start_time = time.time()
        train_losses, val_losses = trainer.train()
        training_time = time.time() - start_time
        
        # Save results
        logger.log("Saving training results...")
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': trainer.best_val_loss,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'num_epochs': ExperimentConfig.NUM_EPOCHS,
            'hyperparameters': {
                'learning_rate': ExperimentConfig.LEARNING_RATE,
                'weight_decay': ExperimentConfig.WEIGHT_DECAY,
                'batch_size': ExperimentConfig.BATCH_SIZE,
                'warmup_ratio': ExperimentConfig.WARMUP_RATIO,
                'max_grad_norm': ExperimentConfig.MAX_GRAD_NORM,
            },
            'seed': ExperimentConfig.CURRENT_SEED,
        }
        
        save_json(
            results,
            os.path.join(ExperimentConfig.RESULTS_DIR, 'baseline_training_history.json')
        )
        
        logger.log(f"\n✓ Baseline training complete!")
        logger.log(f"  Training time: {training_time/60:.1f} minutes")
        logger.log(f"  Best validation loss: {trainer.best_val_loss:.4f}")
        logger.log(f"  Final train loss: {train_losses[-1]:.4f}")
        logger.log(f"  Final val loss: {val_losses[-1]:.4f}")
        
        # Mark complete
        logger.complete(success=True)
        return 0
        
    except Exception as e:
        logger.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())

