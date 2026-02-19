"""
Common Utilities for Foundation LLM Monotonicity Experiments

Adapted from hpc_version/utils/common_utils.py to work with decoder-only models
like Pythia instead of encoder-decoder models like T5.
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
from configs.experiment_config import FoundationExperimentConfig as Config

# ======================================================================
# DETERMINISM FUNCTIONS (Copied from main project)
# ======================================================================

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  Note: deterministic_algorithms not fully supported: {e}")
    
    print(f"✓ All random seeds set to: {seed}")


def get_generator(device='cpu', seed=None):
    """Create PyTorch Generator for reproducible sampling"""
    if seed is None:
        seed = Config.CURRENT_SEED
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id):
    """Initialize each DataLoader worker with deterministic seed"""
    worker_seed = Config.CURRENT_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ======================================================================
# MONOTONIC NEURAL NETWORK COMPONENTS
# ======================================================================

class NonNegativeParametrization(nn.Module):
    """
    Softplus parametrization: W = softplus(V) >= 0
    Identical to main project implementation
    """
    def __init__(self, init_weight=None):
        super().__init__()
        self.init_weight = init_weight
        
    def forward(self, V):
        return F.softplus(V)
    
    def right_inverse(self, W):
        """Initialize V from pretrained W"""
        eps = 1e-4
        W_abs = torch.abs(W) + eps
        V = torch.log(torch.exp(W_abs) - 1.0 + eps)
        return V


def make_model_monotonic(model):
    """
    Apply non-negative weight constraints to FFN layers in decoder-only model.
    
    For Pythia/GPT-style models:
    - FFN structure: fc_in (d_model -> d_ff), fc_out (d_ff -> d_model)
    - Apply constraints to both fc_in and fc_out weight matrices
    
    Note: This is adapted from T5 implementation for decoder-only architectures
    """
    modified_count = 0
    
    # Pythia uses GPTNeoX architecture with specific naming
    # FFN layers are in: model.gpt_neox.layers[i].mlp.dense_h_to_4h and dense_4h_to_h
    
    for name, module in model.named_modules():
        # Check for MLP/FFN layers
        # Pythia: dense_h_to_4h (input projection), dense_4h_to_h (output projection)
        # GPT-2: c_fc (input), c_proj (output)
        # General: mlp, fc_in, fc_out patterns
        
        if any(ffn_pattern in name.lower() for ffn_pattern in 
               ['mlp', 'dense_h_to_4h', 'dense_4h_to_h', 'c_fc', 'c_proj', 'fc_in', 'fc_out']):
            
            # Apply to Linear layers only
            if isinstance(module, nn.Linear):
                # Get current weight for initialization
                current_weight = module.weight.data.clone()
                
                # Register parametrization
                P.register_parametrization(
                    module, "weight",
                    NonNegativeParametrization(init_weight=current_weight)
                )
                modified_count += 1
                print(f"  ✓ Applied monotonicity to: {name}")
    
    if modified_count == 0:
        raise RuntimeError(
            "No FFN layers found to make monotonic! "
            "Check model architecture compatibility."
        )
    
    print(f"\n✓ Applied softplus parametrization to {modified_count} weight matrices")
    print(f"  ⚠️  Note: Model is NOT globally monotonic (attention + residuals unconstrained)")
    
    return model


# ======================================================================
# EVALUATION FUNCTIONS
# ======================================================================

def compute_perplexity(model, dataloader, device):
    """
    Compute perplexity on a dataset.

    Perplexity = exp(average cross-entropy loss)
    """
    model.eval()
    # Free training cache before the eval forward passes so the full logit
    # tensor (batch * seq * vocab) does not trigger OOM on a GPU that was
    # full during the training epoch.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # For causal LM, labels are shifted input_ids
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Count actual tokens (excluding padding)
            num_tokens = (attention_mask == 1).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'total_tokens': total_tokens
    }


def evaluate_language_modeling(model, tokenizer, dataset_name, device, max_samples=None):
    """
    Evaluate on language modeling benchmarks (LAMBADA, etc.)
    Returns accuracy metrics
    """
    # Placeholder - implement specific benchmark evaluation
    # Would load datasets from EleutherAI/lm-evaluation-harness
    
    print(f"Evaluating on {dataset_name}...")
    # TODO: Implement using lm-evaluation-harness
    
    return {
        'dataset': dataset_name,
        'accuracy': 0.0,  # Placeholder
        'num_samples': 0
    }


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
        work_dir = Config.WORK_DIR
    flag_file = os.path.join(work_dir, f"{stage_name}_complete.flag")
    with open(flag_file, 'w') as f:
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {Config.CURRENT_SEED}\n")
    print(f"✓ Created completion flag: {flag_file}")
    return flag_file


def check_completion_flag(stage_name, work_dir=None):
    """Check if stage completed"""
    if work_dir is None:
        work_dir = Config.WORK_DIR
    flag_file = os.path.join(work_dir, f"{stage_name}_complete.flag")
    return os.path.exists(flag_file)


def check_dependencies(required_stages, work_dir=None):
    """Check that all required previous stages completed"""
    if work_dir is None:
        work_dir = Config.WORK_DIR
    
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


class StageLogger:
    """Logger for tracking stage progress"""
    
    def __init__(self, stage_name, log_dir=None):
        self.stage_name = stage_name
        if log_dir is None:
            log_dir = os.path.join(Config.WORK_DIR, 'stage_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"{stage_name}.log")
        self.start_time = time.time()
        
        self.log(f"="*80)
        self.log(f"STAGE: {stage_name}")
        self.log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Seed: {Config.CURRENT_SEED}")
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
        
        return 0 if success else 1


# ======================================================================
# DATASET CLASSES
# ======================================================================

class LanguageModelingDataset(torch.utils.data.Dataset):
    """Dataset for causal language modeling (Pythia-style)"""
    
    def __init__(self, texts, tokenizer, max_length=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length or Config.MAX_SEQ_LENGTH
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze()
        }


if __name__ == "__main__":
    print("Foundation LLM Common Utilities")
    print(f"Using model: {Config.MODEL_NAME}")
    print(f"Work dir: {Config.WORK_DIR}")
