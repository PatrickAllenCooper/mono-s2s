"""
Common Utility Functions for HPC Mono-S2S Experiments

This module contains shared functions used across all stages:
- Determinism setup
- ROUGE computation with bootstrap CIs
- Length statistics
- Model creation and loading
- Dataset loading helpers
"""

import os
import sys
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import torch.nn.utils.parametrize as P

# Add configs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.experiment_config import ExperimentConfig


# ======================================================================
# DETERMINISM FUNCTIONS
# ======================================================================

def set_all_seeds(seed=42):
    """
    Set all random seeds for comprehensive reproducibility.
    
    Note: Environment variables (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG, etc.)
    should be set in job scripts BEFORE importing torch.
    """
    # Update Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Disable TF32 on Ampere+ GPUs
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    
    # PyTorch deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  Note: deterministic_algorithms not fully supported: {e}")
    
    print(f"✓ All random seeds set to: {seed}")


def get_generator(device='cpu', seed=None):
    """Create PyTorch Generator for reproducible sampling"""
    if seed is None:
        seed = ExperimentConfig.CURRENT_SEED
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id):
    """Initialize each DataLoader worker with deterministic seed"""
    worker_seed = ExperimentConfig.CURRENT_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def log_environment():
    """Log environment details for reproducibility"""
    info = {
        "hostname": os.uname().nodename,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "gpu_count": torch.cuda.device_count(),
        })
    
    print("\n" + "="*80)
    print("ENVIRONMENT INFORMATION")
    print("="*80)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    return info


# ======================================================================
# ROUGE & EVALUATION FUNCTIONS
# ======================================================================

def compute_rouge_with_ci(predictions, references, metrics=None, use_stemmer=True, 
                         n_bootstrap=1000, confidence=0.95):
    """
    Compute ROUGE scores with bootstrap confidence intervals.
    """
    from rouge_score import rouge_scorer
    
    if metrics is None:
        metrics = ExperimentConfig.ROUGE_METRICS
    
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=use_stemmer)
    
    # Compute scores for all examples
    all_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        all_scores.append({k: v.fmeasure for k, v in scores.items()})
    
    # Compute means
    mean_scores = {
        metric: np.mean([s[metric] for s in all_scores])
        for metric in metrics
    }
    
    # Bootstrap confidence intervals
    np.random.seed(ExperimentConfig.CURRENT_SEED)
    bootstrap_means = {metric: [] for metric in metrics}
    
    n_samples = len(all_scores)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled = [all_scores[i] for i in indices]
        
        for metric in metrics:
            bootstrap_mean = np.mean([s[metric] for s in resampled])
            bootstrap_means[metric].append(bootstrap_mean)
    
    # Compute confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_scores = {}
    for metric in metrics:
        lower = np.percentile(bootstrap_means[metric], lower_percentile)
        upper = np.percentile(bootstrap_means[metric], upper_percentile)
        ci_scores[metric] = {
            "mean": mean_scores[metric],
            "lower": lower,
            "upper": upper,
            "ci_width": upper - lower
        }
    
    return ci_scores, all_scores


def compute_length_statistics(texts, tokenizer=None):
    """Compute length statistics for texts (token-level if tokenizer provided)"""
    if tokenizer is not None:
        lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        unit = "tokens"
    else:
        lengths = [len(text.split()) for text in texts]
        unit = "words"
    
    return {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "total": int(np.sum(lengths)),
        "unit": unit
    }


def compute_brevity_penalty(predictions, references, tokenizer=None):
    """Compute brevity penalty to detect length biases"""
    if tokenizer is not None:
        pred_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in predictions]
        ref_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in references]
    else:
        pred_lengths = [len(text.split()) for text in predictions]
        ref_lengths = [len(text.split()) for text in references]
    
    total_pred = sum(pred_lengths)
    total_ref = sum(ref_lengths)
    
    length_ratio = total_pred / total_ref if total_ref > 0 else 0
    
    if length_ratio >= 1.0:
        bp = 1.0
    else:
        bp = np.exp(1 - total_ref / total_pred) if total_pred > 0 else 0
    
    return {
        "brevity_penalty": float(bp),
        "length_ratio": float(length_ratio),
        "avg_pred_length": float(np.mean(pred_lengths)),
        "avg_ref_length": float(np.mean(ref_lengths))
    }


# ======================================================================
# MODEL CREATION & LOADING
# ======================================================================

class NonNegativeParametrization(nn.Module):
    """
    Improved softplus parametrization: W = softplus(V) >= 0
    
    Uses inverse softplus initialization to preserve pretrained weights:
    V_init = inverse_softplus(|W_pretrained| + eps)
    
    This minimizes disruption to pretrained model knowledge while enforcing W >= 0.
    """
    def __init__(self, init_weight=None):
        super().__init__()
        self.init_weight = init_weight
        
    def forward(self, V):
        return F.softplus(V)
    
    def right_inverse(self, W):
        """
        Initialize V from pretrained W to preserve learned features.
        
        For numerical stability:
        - Use |W| to ensure positive input to inverse softplus
        - Add small epsilon to avoid log(0)
        - Clamp to reasonable range
        """
        eps = 1e-4
        W_abs = torch.abs(W) + eps
        # inverse_softplus(x) = log(exp(x) - 1) 
        # For numerical stability, use: log(expm1(x)) for x > 0
        # But simpler: inverse_softplus(x) ≈ x for large x, log(x) + log(2) for small x
        # We use: log(exp(x) - 1 + eps) for stability
        V = torch.log(torch.exp(W_abs) - 1.0 + eps)
        return V


def make_model_monotonic(model):
    """
    Apply non-negative weight constraints to FFN layers using softplus parametrization.
    
    Covers both standard (wi, wo) and gated (wi_0, wi_1, wo) FFN variants.
    
    Note: This does NOT make the full model globally monotonic due to LayerNorm,
    residual connections, and unconstrained attention.
    """
    # Try importing the FFN class - handle both old and new transformers versions
    try:
        from transformers.models.t5.modeling_t5 import T5DenseReluDense
        FFN_CLASS = T5DenseReluDense
    except ImportError:
        try:
            # Newer transformers versions use T5DenseActDense
            from transformers.models.t5.modeling_t5 import T5DenseActDense
            FFN_CLASS = T5DenseActDense
        except ImportError:
            try:
                # Even newer versions use T5DenseGatedActDense
                from transformers.models.t5.modeling_t5 import T5DenseGatedActDense
                FFN_CLASS = T5DenseGatedActDense
            except ImportError:
                # Fallback: Use duck typing - find modules with FFN attributes
                FFN_CLASS = None
    
    modified_count = 0
    
    # If we found a class, use isinstance
    if FFN_CLASS is not None:
        for module in model.modules():
            if isinstance(module, FFN_CLASS):
                # Parametrize all FFN weight sublayers with preserved initialization
                for param_name in ["wi", "wi_0", "wi_1", "wo"]:
                    if hasattr(module, param_name):
                        sub_module = getattr(module, param_name)
                        if hasattr(sub_module, "weight"):
                            # Get current weight for initialization
                            current_weight = sub_module.weight.data.clone()
                            # Register with improved initialization
                            P.register_parametrization(
                                sub_module, "weight", 
                                NonNegativeParametrization(init_weight=current_weight)
                            )
                            modified_count += 1
    else:
        # Fallback: Use duck typing - find modules with FFN-like structure
        print("  Using duck typing to find FFN layers...")
        for name, module in model.named_modules():
            # Check if module has FFN-like attributes (wi/wo or wi_0/wi_1/wo)
            has_ffn_structure = (
                (hasattr(module, "wi") and hasattr(module, "wo")) or
                (hasattr(module, "wi_0") and hasattr(module, "wi_1") and hasattr(module, "wo"))
            )
            
            if has_ffn_structure and "DenseActDense" in type(module).__name__:
                # Parametrize all FFN weight sublayers with preserved initialization
                for param_name in ["wi", "wi_0", "wi_1", "wo"]:
                    if hasattr(module, param_name):
                        sub_module = getattr(module, param_name)
                        if hasattr(sub_module, "weight"):
                            # Get current weight for initialization
                            current_weight = sub_module.weight.data.clone()
                            # Register with improved initialization
                            P.register_parametrization(
                                sub_module, "weight",
                                NonNegativeParametrization(init_weight=current_weight)
                            )
                            modified_count += 1
    
    if modified_count == 0:
        raise RuntimeError("No FFN layers found to make monotonic! Check transformers version compatibility.")
    
    print(f"✓ Applied softplus parametrization to {modified_count} weight matrices")
    print(f"  Covers: wi, wi_0, wi_1, wo (handles gated variants)")
    print(f"  ⚠️  Note: Model is NOT globally monotonic (LayerNorm + residuals + attention)")
    return model


def load_model(model_type, checkpoint_path=None, device='cuda'):
    """
    Load model: standard, baseline, or monotonic.
    
    Args:
        model_type: 'standard', 'baseline', or 'monotonic'
        checkpoint_path: Path to checkpoint (None for pre-trained)
        device: Device to load model on
    
    Returns:
        model, is_pretrained_only
    """
    from transformers import T5ForConditionalGeneration
    
    print(f"\nLoading {model_type} model...")
    
    # Load base model
    model = T5ForConditionalGeneration.from_pretrained(ExperimentConfig.MODEL_NAME).to(device)
    
    # Verify T5 architecture
    assert model.config.model_type == "t5", \
        f"ERROR: Expected T5, got {model.config.model_type}"
    
    # Apply monotonic constraints if needed
    if model_type == 'monotonic':
        model = make_model_monotonic(model)
    
    # Load fine-tuned weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        is_pretrained_only = False
        print(f"✓ Loaded fine-tuned {model_type} model")
    else:
        if checkpoint_path:
            print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
        model.eval()
        is_pretrained_only = True
        print(f"✓ Using pre-trained {model_type} model")
    
    return model, is_pretrained_only


# ======================================================================
# GENERATION & EVALUATION
# ======================================================================

def generate_summary_fixed_params(model, text, tokenizer, device):
    """Generate summary with FIXED decoding parameters for fair comparison"""
    inputs = tokenizer(
        "summarize: " + text.strip(),
        return_tensors="pt",
        max_length=ExperimentConfig.MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=ExperimentConfig.DECODE_MAX_NEW_TOKENS,
            min_new_tokens=ExperimentConfig.DECODE_MIN_NEW_TOKENS,
            num_beams=ExperimentConfig.DECODE_NUM_BEAMS,
            length_penalty=ExperimentConfig.DECODE_LENGTH_PENALTY,
            no_repeat_ngram_size=ExperimentConfig.DECODE_NO_REPEAT_NGRAM_SIZE,
            early_stopping=ExperimentConfig.DECODE_EARLY_STOPPING
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def compute_avg_loss(model, data_loader, device):
    """Compute average loss on a data loader"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


# ======================================================================
# FILE & LOGGING HELPERS
# ======================================================================

def save_json(data, filepath, indent=2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    print(f"✓ Saved to: {filepath}")


def load_json(filepath):
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)


def create_completion_flag(stage_name, work_dir=None):
    """Create completion flag file for stage"""
    if work_dir is None:
        work_dir = ExperimentConfig.WORK_DIR
    flag_file = os.path.join(work_dir, f"{stage_name}_complete.flag")
    with open(flag_file, 'w') as f:
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {ExperimentConfig.CURRENT_SEED}\n")
    print(f"✓ Created completion flag: {flag_file}")
    return flag_file


def check_completion_flag(stage_name, work_dir=None):
    """Check if stage completed"""
    if work_dir is None:
        work_dir = ExperimentConfig.WORK_DIR
    flag_file = os.path.join(work_dir, f"{stage_name}_complete.flag")
    return os.path.exists(flag_file)


def check_dependencies(required_stages, work_dir=None):
    """
    Check that all required previous stages completed.
    Returns True if all dependencies met, False otherwise.
    """
    if work_dir is None:
        work_dir = ExperimentConfig.WORK_DIR
    
    missing = []
    for stage in required_stages:
        if not check_completion_flag(stage, work_dir):
            missing.append(stage)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print(f"   Please run these stages first before proceeding.")
        return False
    else:
        print(f"✓ All dependencies met: {', '.join(required_stages)}")
        return True


# ======================================================================
# DATASET LOADING HELPERS
# ======================================================================

def load_dataset_split(dataset_name, split, text_field, summary_field, 
                       config=None, max_samples=None, max_retries=None, retry_delay=None):
    """
    Generic dataset loader with retry logic and better error handling.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split ('train', 'validation', 'test')
        text_field: Field name for input text
        summary_field: Field name for summary
        config: Dataset config (e.g., "3.0.0" for CNN/DM)
        max_samples: Limit number of samples (None for all)
        max_retries: Number of retry attempts (defaults to ExperimentConfig)
        retry_delay: Delay between retries in seconds (defaults to ExperimentConfig)
    
    Returns:
        texts, summaries (lists)
    """
    from datasets import load_dataset
    import time
    
    # Use config defaults if not specified
    if max_retries is None:
        max_retries = getattr(ExperimentConfig, 'DATASET_MAX_RETRIES', 3)
    if retry_delay is None:
        retry_delay = getattr(ExperimentConfig, 'DATASET_RETRY_DELAY', 10)
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if config:
                dataset = load_dataset(dataset_name, config, split=split, 
                                      trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, split=split,
                                      trust_remote_code=True)
            
            texts = []
            summaries = []
            
            for i, example in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                text = example.get(text_field, "")
                summary = example.get(summary_field, "")
                
                if text and summary:
                    texts.append(text.strip())
                    summaries.append(summary.strip())
            
            print(f"  ✓ Loaded {len(texts)} samples from {dataset_name} ({split})")
            return texts, summaries
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"  ⚠️  Attempt {attempt + 1}/{max_retries} failed for {dataset_name} ({split}): {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  ❌ Failed to load {dataset_name} ({split}) after {max_retries} attempts")
                print(f"  Last error: {last_error}")
                
                # Check if partial results are allowed
                if getattr(ExperimentConfig, 'DATASET_ALLOW_PARTIAL', True):
                    print(f"  Continuing with other datasets (DATASET_ALLOW_PARTIAL=True)")
                    return [], []
                else:
                    raise last_error


# ======================================================================
# CHECKPOINT MANAGEMENT
# ======================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, is_best, 
                   checkpoint_dir, history_path=None, train_losses=None, val_losses=None):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save epoch checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(save_dict, checkpoint_path)
    print(f"  ✓ Checkpoint saved: epoch_{epoch}.pt")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(model.state_dict(), best_path)
        print(f"  ✓ Best model saved: best_model.pt")
    
    # Save training history
    if history_path and train_losses is not None and val_losses is not None:
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': val_loss if is_best else None
        }
        save_json(history, history_path)


def load_checkpoint(checkpoint_dir):
    """Load latest checkpoint from directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                   if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    if not checkpoints:
        return None
    
    # Get latest
    epochs = [int(f.replace('checkpoint_epoch_', '').replace('.pt', '')) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')
    
    print(f"  Found checkpoint: epoch {latest_epoch}")
    return torch.load(latest_checkpoint, map_location='cpu', weights_only=False)


# ======================================================================
# PROGRESS TRACKING
# ======================================================================

class StageLogger:
    """Logger for tracking stage progress"""
    
    def __init__(self, stage_name, log_dir=None):
        self.stage_name = stage_name
        if log_dir is None:
            log_dir = os.path.join(ExperimentConfig.WORK_DIR, 'stage_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"{stage_name}.log")
        self.start_time = time.time()
        
        self.log(f"="*80)
        self.log(f"STAGE: {stage_name}")
        self.log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Seed: {ExperimentConfig.CURRENT_SEED}")
        self.log(f"="*80)
    
    def log(self, message):
        """Log message to file and print"""
        timestamp = time.strftime('%H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
    
    def complete(self, success=True):
        """Mark stage as complete"""
        elapsed = time.time() - self.start_time
        status = "SUCCESS" if success else "FAILED"
        self.log(f"\n{'='*80}")
        self.log(f"Stage {self.stage_name}: {status}")
        self.log(f"Elapsed time: {elapsed/60:.1f} minutes")
        self.log(f"{'='*80}")
        
        if success:
            create_completion_flag(self.stage_name)
        
        return 0 if success else 1  # Return int exit code, not bool


# ======================================================================
# DATASET CLASSES
# ======================================================================

class SummarizationDataset(torch.utils.data.Dataset):
    """Dataset class for T5 summarization"""
    
    def __init__(self, texts, summaries, tokenizer, 
                 max_input_length=None, max_target_length=None):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length or ExperimentConfig.MAX_INPUT_LENGTH
        self.max_target_length = max_target_length or ExperimentConfig.MAX_TARGET_LENGTH
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = "summarize: " + self.texts[idx]
        summary = self.summaries[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            summary,
            return_tensors="pt",
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
        )
        
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }


if __name__ == "__main__":
    print("Common Utilities Module")
    print("This module is imported by other stages")
    print(f"Using config: {ExperimentConfig.MODEL_NAME}")
    print(f"Work dir: {ExperimentConfig.WORK_DIR}")

