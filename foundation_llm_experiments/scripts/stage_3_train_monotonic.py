#!/usr/bin/env python3
"""
Stage 3: Train Monotonic Model (Pythia-1.4B with W>=0 FFN Constraints)

Finetunes monotonic-initialized Pythia-1.4B on Pile training data.
Uses extended warmup for stability under softplus parametrization.
Checkpoints every SAVE_CHECKPOINT_EVERY_N_STEPS steps and on SIGUSR1
so that wall-time kills do not waste compute.

Inputs:
- monotonic_initialized.pt (from stage 1)
- Pile training data

Outputs:
- monotonic_checkpoints/best_model.pt
- monotonic_training_history.json
- stage_3_train_monotonic_complete.flag
"""

import json
import os
import signal
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
    LanguageModelingDataset, compute_perplexity, make_model_monotonic
)

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW


class MonotonicTrainer:
    """Trainer for monotonic (W>=0 FFN) Pythia model.

    Checkpoint granularity is at the step level so that a 24-hour SLURM
    wall-time kill never discards more than SAVE_CHECKPOINT_EVERY_N_STEPS
    steps of work.  The SIGUSR1 handler (registered in main()) forces an
    immediate emergency save before SLURM sends SIGTERM.
    """

    STEP_CKPT_NAME = "checkpoint_step.pt"

    def __init__(self, model, train_loader, val_loader, device,
                 checkpoint_dir, history_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.history_path = history_path

        self.learning_rate = Config.MONOTONIC_RECOVERY_LR
        self.weight_decay = Config.MONOTONIC_RECOVERY_WEIGHT_DECAY
        self.num_epochs = Config.MONOTONIC_RECOVERY_EPOCHS
        self.max_grad_norm = Config.MAX_GRAD_NORM
        self.warmup_ratio = Config.MONOTONIC_RECOVERY_WARMUP_RATIO
        self.ckpt_every_n_steps = Config.SAVE_CHECKPOINT_EVERY_N_STEPS

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training state - may be overwritten by load_checkpoint
        self.start_epoch = 0
        self.start_step = 0
        self.global_step = 0
        self.train_losses = []
        self.val_perplexities = []
        self.best_val_perplexity = float('inf')

        self._save_and_exit = False

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.load_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _step_ckpt_path(self):
        return os.path.join(self.checkpoint_dir, self.STEP_CKPT_NAME)

    def _epoch_ckpt_path(self, epoch):
        return os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

    def _build_save_dict(self, epoch, step_in_epoch, val_ppl=None):
        return {
            'epoch': epoch,
            'step_in_epoch': step_in_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_perplexity': self.best_val_perplexity,
            'val_perplexity': val_ppl,
            'train_losses': self.train_losses,
            'val_perplexities': self.val_perplexities,
        }

    def save_step_checkpoint(self, epoch, step_in_epoch):
        tmp = self._step_ckpt_path() + ".tmp"
        torch.save(self._build_save_dict(epoch, step_in_epoch), tmp)
        os.replace(tmp, self._step_ckpt_path())

    def save_epoch_checkpoint(self, epoch, val_ppl, is_best=False):
        path = self._epoch_ckpt_path(epoch)
        torch.save(self._build_save_dict(epoch, 0, val_ppl), path)
        print(f"  Checkpoint saved: checkpoint_epoch_{epoch}.pt")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_path)
            print(f"  Best model saved: best_model.pt")

        history = {
            'train_losses': self.train_losses,
            'val_perplexities': self.val_perplexities,
            'best_val_perplexity': self.best_val_perplexity,
            'constraint_info': {
                'type': 'softplus_parametrization',
                'formula': 'W = softplus(V) >= 0',
                'scope': 'FFN sublayers only',
                'globally_monotonic': False,
            },
        }
        save_json(history, self.history_path)

        if os.path.exists(self._step_ckpt_path()):
            os.remove(self._step_ckpt_path())

    def load_checkpoint(self):
        step_path = self._step_ckpt_path()
        epoch_ckpts = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
        ]

        candidate = None
        if os.path.exists(step_path):
            candidate = step_path
            print(f"\nFound step-level checkpoint: {step_path}")
        elif epoch_ckpts:
            latest_epoch = max(
                int(f.replace('checkpoint_epoch_', '').replace('.pt', ''))
                for f in epoch_ckpts
            )
            candidate = self._epoch_ckpt_path(latest_epoch)
            print(f"\nFound epoch checkpoint: {candidate}")

        if candidate is None:
            print("\nNo checkpoint found. Starting training from scratch.")
            return

        ckpt = torch.load(candidate, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch']
        self.start_step = ckpt.get('step_in_epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_perplexity = ckpt.get('best_val_perplexity', float('inf'))
        self.train_losses = ckpt.get('train_losses', [])
        self.val_perplexities = ckpt.get('val_perplexities', [])

        print(f"Resuming from epoch {self.start_epoch}, step {self.start_step}")
        print(f"  Best val perplexity so far: {self.best_val_perplexity:.2f}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def validate(self):
        result = compute_perplexity(self.model, self.val_loader, self.device)
        return result['perplexity'], result['loss']

    def train_epoch(self, epoch, skip_steps=0):
        """Train one epoch.  skip_steps > 0 fast-forwards past already-done
        batches when resuming mid-epoch."""
        self.model.train()
        total_loss = 0.0
        steps_this_epoch = 0
        last_ckpt_time = time.time()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Monotonic epoch {epoch + 1}/{self.num_epochs}",
            initial=skip_steps,
            total=len(self.train_loader),
        )

        for step, batch in enumerate(self.train_loader):
            if step < skip_steps:
                progress_bar.update(1)
                continue

            if self._save_and_exit:
                print(f"\nSIGUSR1 received - saving emergency checkpoint at step {step}.")
                self.save_step_checkpoint(epoch, step)
                print("Emergency checkpoint saved. Exiting cleanly.")
                sys.exit(0)

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # W = softplus(V) >= 0 is maintained automatically; no projection needed.

            total_loss += loss.item()
            steps_this_epoch += 1
            self.global_step += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })
            progress_bar.update(1)

            minutes_since_save = (time.time() - last_ckpt_time) / 60
            if (steps_this_epoch % self.ckpt_every_n_steps == 0 or
                    minutes_since_save >= Config.SAVE_CHECKPOINT_EVERY_N_MINUTES):
                self.save_step_checkpoint(epoch, step + 1)
                last_ckpt_time = time.time()

        progress_bar.close()
        avg_loss = total_loss / max(steps_this_epoch, 1)
        return avg_loss

    def train(self, max_epochs_per_run=None):
        if self.start_step >= len(self.train_loader):
            self.start_epoch += 1
            self.start_step = 0
            print(f"Epoch {self.start_epoch} training was complete; resuming from epoch {self.start_epoch + 1}.")

        print(f"\nStarting monotonic training from epoch {self.start_epoch + 1}/{self.num_epochs}")
        print(f"  start_epoch={self.start_epoch}, start_step={self.start_step}")
        print(f"  Extended warmup ratio: {self.warmup_ratio}")
        print("=" * 80)

        epochs_run = 0

        for epoch in range(self.start_epoch, self.num_epochs):
            if max_epochs_per_run is not None and epochs_run >= max_epochs_per_run:
                print(f"\nReached max epochs per run ({max_epochs_per_run}). Stopping.")
                break

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 80)

            skip = self.start_step if epoch == self.start_epoch else 0
            train_loss = self.train_epoch(epoch, skip_steps=skip)
            self.train_losses.append(train_loss)

            val_ppl, val_loss = self.validate()
            self.val_perplexities.append(val_ppl)

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Perplexity: {val_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f}")

            is_best = val_ppl < self.best_val_perplexity
            if is_best:
                self.best_val_perplexity = val_ppl
                print(f"  New best validation perplexity!")

            self.save_epoch_checkpoint(epoch + 1, val_ppl, is_best)
            epochs_run += 1

        total_completed = self.start_epoch + epochs_run
        is_complete = total_completed >= self.num_epochs

        print(f"\n[COMPLETION CHECK]")
        print(f"  total_epochs_completed={total_completed}, target={self.num_epochs}")
        print(f"  is_complete={is_complete}")

        if is_complete:
            print("\n" + "=" * 80)
            print("ALL TRAINING COMPLETE!")
            print(f"  Best val perplexity: {self.best_val_perplexity:.2f}")
            print("=" * 80)
        else:
            print(f"\n[INFO] Partial run ({total_completed}/{self.num_epochs} epochs).")
            print(f"[INFO] Re-submit job to continue.")

        return self.train_losses, self.val_perplexities, is_complete


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs_per_run", type=int, default=None)
    args = parser.parse_args()

    logger = StageLogger("stage_3_train_monotonic")

    try:
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_0_setup', 'stage_1_apply_monotonicity']):
            logger.complete(success=False)
            return 1

        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()

        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.log("Loading training data from Pile...")
        from datasets import load_dataset

        # Always use a bounded sample count so epochs fit within wall time.
        max_samples = Config.TRAINING_SAMPLES or Config.QUICK_TRAINING_SAMPLES
        logger.log(f"  Sample limit: {max_samples:,}")

        try:
            pile = load_dataset(
                Config.TRAINING_DATASET,
                split="train",
                streaming=True,
                cache_dir=Config.DATA_CACHE_DIR,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.log(f"  WARNING: Could not load train split ({e}), trying validation split")
            pile = load_dataset(
                Config.TRAINING_DATASET,
                split="validation",
                streaming=False,
                cache_dir=Config.DATA_CACHE_DIR,
                trust_remote_code=True,
            )

        train_texts = []
        for i, example in enumerate(pile):
            if i >= max_samples:
                break
            train_texts.append(example['text'])
            if i % 10000 == 0:
                logger.log(f"    Loaded {i:,} samples...")

        logger.log(f"  Loaded {len(train_texts):,} training samples")

        split_idx = int(len(train_texts) * 0.9)
        train_subset = train_texts[:split_idx]
        val_subset = train_texts[split_idx:]

        logger.log(f"  Train: {len(train_subset):,}  Val: {len(val_subset):,}")

        train_dataset = LanguageModelingDataset(train_subset, tokenizer)
        val_dataset = LanguageModelingDataset(val_subset, tokenizer)

        generator = get_generator(device='cpu')
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        logger.log(f"  Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

        logger.log("Loading monotonic-initialized model...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        logger.log("Applying monotonicity constraints...")
        model = make_model_monotonic(model)

        init_path = os.path.join(Config.CHECKPOINT_DIR, 'monotonic_initialized.pt')
        if os.path.exists(init_path):
            logger.log(f"Loading initialized weights from: {init_path}")
            state_dict = torch.load(init_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            logger.log("  Loaded monotonic-initialized weights")
        else:
            logger.log("WARNING: No initialized weights found, using freshly applied constraints")

        model = model.to(device)
        model.gradient_checkpointing_enable()
        logger.log(f"  bfloat16 precision + gradient checkpointing enabled")

        checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, 'monotonic_checkpoints')
        history_path = os.path.join(Config.RESULTS_DIR, 'monotonic_training_history.json')
        os.makedirs(checkpoint_dir, exist_ok=True)

        trainer = MonotonicTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=checkpoint_dir,
            history_path=history_path,
        )

        def _sigusr1_handler(signum, frame):
            trainer._save_and_exit = True

        signal.signal(signal.SIGUSR1, _sigusr1_handler)

        logger.log("Starting monotonic recovery training...")
        logger.log(f"  Epochs: {Config.MONOTONIC_RECOVERY_EPOCHS}")
        logger.log(f"  Extended warmup: {Config.MONOTONIC_RECOVERY_WARMUP_RATIO}")
        logger.log(f"  Step checkpoint every: {Config.SAVE_CHECKPOINT_EVERY_N_STEPS} steps "
                   f"/ {Config.SAVE_CHECKPOINT_EVERY_N_MINUTES} min")
        if args.max_epochs_per_run:
            logger.log(f"  Max epochs this run: {args.max_epochs_per_run}")

        start_time = time.time()
        train_losses, val_perplexities, is_complete = trainer.train(
            max_epochs_per_run=args.max_epochs_per_run,
        )
        training_time = time.time() - start_time

        logger.log("\nSaving training results...")
        results = {
            'train_losses': train_losses,
            'val_perplexities': val_perplexities,
            'best_val_perplexity': trainer.best_val_perplexity,
            'training_time_seconds': training_time,
            'training_time_hours': training_time / 3600,
            'num_epochs': Config.MONOTONIC_RECOVERY_EPOCHS,
            'training_samples': max_samples,
            'hyperparameters': {
                'learning_rate': Config.MONOTONIC_RECOVERY_LR,
                'weight_decay': Config.MONOTONIC_RECOVERY_WEIGHT_DECAY,
                'batch_size': Config.BATCH_SIZE,
                'warmup_ratio': Config.MONOTONIC_RECOVERY_WARMUP_RATIO,
                'max_grad_norm': Config.MAX_GRAD_NORM,
            },
            'seed': Config.CURRENT_SEED,
            'constraint_info': {
                'type': 'softplus_parametrization',
                'formula': 'W = softplus(V) >= 0',
                'scope': 'FFN sublayers only',
                'globally_monotonic': False,
            },
        }
        save_json(results, history_path)

        logger.log(f"  Training time: {training_time / 3600:.1f} hours")
        logger.log(f"  Best val perplexity: {trainer.best_val_perplexity:.2f}")

        if is_complete:
            logger.complete(success=True)
        else:
            logger.log("Job finished (partial run). Resubmit to continue.")

        return 0

    except Exception as e:
        logger.log(f"\nERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())
