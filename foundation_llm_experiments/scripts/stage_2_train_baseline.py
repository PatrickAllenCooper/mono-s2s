#!/usr/bin/env python3
"""
Stage 2: Train Baseline Model (Standard Pythia-1.4B Recovery)

Finetunes standard Pythia-1.4B on Pile training data for 1 epoch.
This serves as the fair comparison baseline.

Inputs:
- Pythia-1.4B from cache

Outputs:
- baseline_checkpoints/best_model.pt
- baseline_training_history.json
- stage_2_train_baseline_complete.flag
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, create_completion_flag, save_json,
    StageLogger, check_dependencies, get_generator, worker_init_fn,
    LanguageModelingDataset, compute_perplexity
)

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW


class BaselineTrainer:
    """Trainer for standard (unconstrained) Pythia model"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 checkpoint_dir, history_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.history_path = history_path
        
        # Hyperparameters
        self.learning_rate = Config.RECOVERY_LR
        self.weight_decay = Config.RECOVERY_WEIGHT_DECAY
        self.num_epochs = Config.RECOVERY_EPOCHS
        self.max_grad_norm = Config.MAX_GRAD_NORM
        self.warmup_ratio = Config.RECOVERY_WARMUP_RATIO
        
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
        self.val_perplexities = []
        self.best_val_perplexity = float('inf')
        
        # CRITICAL: Load checkpoint if exists (for resume after timeout)
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load latest checkpoint if it exists (enables resume after timeout)"""
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
        checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_perplexity = checkpoint.get('best_val_perplexity', float('inf'))
        
        # Load history
        if os.path.exists(self.history_path):
            import json
            try:
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to pickle
                history = torch.load(self.history_path, weights_only=False)
            self.train_losses = history.get('train_losses', [])
            self.val_perplexities = history.get('val_perplexities', [])
        
        print(f"✓ Resuming from epoch {self.start_epoch}")
        print(f"  Best validation perplexity so far: {self.best_val_perplexity:.2f}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training Baseline Pythia")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # For causal LM, labels are input_ids shifted
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
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
        """Validate model - compute perplexity"""
        result = compute_perplexity(self.model, self.val_loader, self.device)
        return result['perplexity'], result['loss']
    
    def save_checkpoint(self, epoch, val_ppl, is_best=False):
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
            'best_val_perplexity': self.best_val_perplexity,
            'val_perplexity': val_ppl,
        }
        
        torch.save(save_dict, checkpoint_path)
        print(f"  ✓ Checkpoint saved: epoch_{epoch}.pt")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_path)
            print(f"  ✓ Best model saved: best_model.pt")
        
        # Save history
        history = {
            'train_losses': self.train_losses,
            'val_perplexities': self.val_perplexities,
            'best_val_perplexity': self.best_val_perplexity
        }
        save_json(history, self.history_path)
    
    def train(self, max_epochs_per_run=None):
        """Full training loop with optional epoch limit for job time constraints"""
        print(f"\nStarting training from epoch {self.start_epoch + 1}/{self.num_epochs}")
        print(f"Resume state: start_epoch={self.start_epoch}, target={self.num_epochs}")
        print("="*80)
        
        epochs_run = 0
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Check if we reached max epochs for this run (handles job time limits)
            if max_epochs_per_run is not None and epochs_run >= max_epochs_per_run:
                print(f"\nReached max epochs per run ({max_epochs_per_run}). Stopping.")
                print(f"Progress: {epoch}/{self.num_epochs} epochs completed")
                print(f"To resume: Re-submit this job (will auto-resume from epoch {epoch})")
                break
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-"*80)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_ppl, val_loss = self.validate()
            self.val_perplexities.append(val_ppl)
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Perplexity: {val_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Check if best
            is_best = val_ppl < self.best_val_perplexity
            if is_best:
                self.best_val_perplexity = val_ppl
                print(f"  ✓ New best validation perplexity!")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_ppl, is_best)
            
            epochs_run += 1
        
        # Calculate completion status
        total_epochs_completed = self.start_epoch + epochs_run
        is_complete = total_epochs_completed >= self.num_epochs
        
        print(f"\n[COMPLETION CHECK]")
        print(f"  start_epoch={self.start_epoch}")
        print(f"  epochs_run={epochs_run}")
        print(f"  total_epochs_completed={total_epochs_completed}")
        print(f"  target_epochs={self.num_epochs}")
        print(f"  is_complete={is_complete}")
        
        if is_complete:
            print("\n" + "="*80)
            print("✓ ALL TRAINING COMPLETE!")
            print(f"  Total epochs completed: {total_epochs_completed}/{self.num_epochs}")
            print(f"  Best validation perplexity: {self.best_val_perplexity:.2f}")
            print("="*80)
        else:
            print(f"\n[INFO] Partial training complete ({total_epochs_completed}/{self.num_epochs} epochs)")
            print(f"[INFO] Re-submit job to continue from epoch {total_epochs_completed + 1}")
        
        return self.train_losses, self.val_perplexities, is_complete


def main():
    """Run baseline training"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs_per_run", type=int, default=None,
                       help="Maximum number of epochs to run in this job (for time limits)")
    args = parser.parse_args()
    
    logger = StageLogger("stage_2_train_baseline")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_0_setup']):
            logger.complete(success=False)
            return 1
        
        # Set seeds
        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()
        
        # Load tokenizer
        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR
        )
        # Set pad_token for Pythia tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load training data
        logger.log("Loading training data from Pile...")
        
        # Load Pile dataset
        from datasets import load_dataset
        
        if Config.USE_FULL_EVAL_SETS and hasattr(Config, 'TRAINING_SAMPLES') and Config.TRAINING_SAMPLES is None:
            logger.log("  Loading full Pile training data (streaming)...")
            logger.log("  ⚠️  WARNING: This will process ~300B tokens")
            logger.log("  ⚠️  Consider using TRAINING_SAMPLES limit for testing")
            
            pile = load_dataset(
                Config.TRAINING_DATASET,
                split="train",
                streaming=True,
                cache_dir=Config.DATA_CACHE_DIR,
                trust_remote_code=True
            )
            
            # Stream and collect samples
            train_texts = []
            logger.log("  Streaming samples...")
            for i, example in enumerate(pile):
                if i % 10000 == 0:
                    logger.log(f"    Loaded {i:,} samples...")
                train_texts.append(example['text'])
                # Limit for safety
                if i >= 1000000:  # 1M samples max
                    break
        else:
            # Quick mode or limited samples
            logger.log(f"  Loading Pile data (limited samples for fast training)...")
            
            # Try validation split first, fall back to train split if not available
            try:
                pile = load_dataset(
                    Config.TRAINING_DATASET,
                    split="validation",
                    streaming=False,
                    cache_dir=Config.DATA_CACHE_DIR,
                    trust_remote_code=True
                )
                logger.log(f"    Using validation split")
            except (ValueError, KeyError):
                logger.log(f"    Validation split not available, using train split")
                pile = load_dataset(
                    Config.TRAINING_DATASET,
                    split="train",
                    streaming=True,
                    cache_dir=Config.DATA_CACHE_DIR,
                    trust_remote_code=True
                )
            
            max_samples = getattr(Config, 'TRAINING_SAMPLES', None) or getattr(Config, 'QUICK_TRAINING_SAMPLES', 10000)
            train_texts = [example['text'] for i, example in enumerate(pile) if i < max_samples]
        
        logger.log(f"  ✓ Loaded {len(train_texts)} training samples")
        
        # Split into train/val
        split_idx = int(len(train_texts) * 0.9)
        train_subset = train_texts[:split_idx]
        val_subset = train_texts[split_idx:]
        
        logger.log(f"  Train: {len(train_subset)} samples")
        logger.log(f"  Val: {len(val_subset)} samples")
        
        train_dataset = LanguageModelingDataset(
            train_subset,
            tokenizer
        )
        val_dataset = LanguageModelingDataset(
            val_subset,
            tokenizer
        )
        
        # Create data loaders
        generator = get_generator(device='cpu')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        logger.log(f"  Training batches: {len(train_loader)}")
        logger.log(f"  Validation batches: {len(val_loader)}")
        
        # Load model
        logger.log("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            low_cpu_mem_usage=True
        ).to(device)
        
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        
        logger.log(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.log(f"  Using bfloat16 precision + gradient checkpointing")
        
        # Setup checkpoint paths
        checkpoint_dir = os.path.join(
            Config.CHECKPOINT_DIR,
            'baseline_checkpoints'
        )
        history_path = os.path.join(
            Config.RESULTS_DIR,
            'baseline_training_history.json'
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create trainer
        logger.log("\nCreating trainer...")
        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=checkpoint_dir,
            history_path=history_path
        )
        
        # Train
        logger.log("Starting training...")
        logger.log(f"  Epochs: {Config.RECOVERY_EPOCHS}")
        logger.log(f"  Batch size: {Config.BATCH_SIZE}")
        logger.log(f"  Gradient accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
        logger.log(f"  Learning rate: {Config.RECOVERY_LR}")
        logger.log(f"  Warmup ratio: {Config.RECOVERY_WARMUP_RATIO}")
        if args.max_epochs_per_run:
            logger.log(f"  Max epochs this run: {args.max_epochs_per_run}")
        
        start_time = time.time()
        train_losses, val_perplexities, is_complete = trainer.train(
            max_epochs_per_run=args.max_epochs_per_run
        )
        training_time = time.time() - start_time
        
        # Save results
        logger.log("\nSaving training results...")
        results = {
            'train_losses': train_losses,
            'val_perplexities': val_perplexities,
            'best_val_perplexity': trainer.best_val_perplexity,
            'training_time_seconds': training_time,
            'training_time_hours': training_time / 3600,
            'num_epochs': Config.RECOVERY_EPOCHS,
            'hyperparameters': {
                'learning_rate': Config.RECOVERY_LR,
                'weight_decay': Config.RECOVERY_WEIGHT_DECAY,
                'batch_size': Config.BATCH_SIZE,
                'warmup_ratio': Config.RECOVERY_WARMUP_RATIO,
                'max_grad_norm': Config.MAX_GRAD_NORM,
            },
            'seed': Config.CURRENT_SEED,
        }
        
        save_json(results, history_path)
        
        logger.log(f"\n✓ Baseline training phase complete!")
        logger.log(f"  Training time: {training_time/3600:.1f} hours")
        logger.log(f"  Best validation perplexity: {trainer.best_val_perplexity:.2f}")
        
        # Mark complete only if actually finished all epochs
        if is_complete:
            logger.complete(success=True)
        else:
            logger.log(f"\nJob finished (partial run). Resubmit to continue training.")
            logger.log(f"Checkpoint saved at epoch {trainer.start_epoch + len(train_losses)}")
        
        return 0
        
    except Exception as e:
        logger.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())
