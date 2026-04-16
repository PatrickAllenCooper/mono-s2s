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
        
        # Only constrain input projection (dense_h_to_4h / c_fc / fc_in), NOT output projection.
        # This preserves more model expressiveness while still enforcing monotonicity
        # on the critical FFN expansion step.
        if any(ffn_pattern in name.lower() for ffn_pattern in
               ['dense_h_to_4h', 'c_fc', 'fc_in']):
            
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
            
            # Use smaller chunks to avoid OOM during validation
            try:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            except torch.cuda.OutOfMemoryError:
                # If OOM, try with half the batch
                torch.cuda.empty_cache()
                mid = input_ids.size(0) // 2
                if mid == 0:
                    continue
                outputs = model(input_ids=input_ids[:mid], labels=labels[:mid])
                loss = outputs.loss
                attention_mask = attention_mask[:mid]
            
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


def compute_perplexity_resumable(
    model,
    dataloader,
    device,
    progress_path=None,
    flush_every=10,
    log_fn=None,
):
    """
    Perplexity computation with durable per-batch checkpointing.

    On deallocation-triggered restart, state is reloaded from
    `progress_path` and already-processed batches are skipped. The
    dataloader MUST be deterministic (shuffle=False) for skip-resume
    to be correct; evaluation loaders satisfy this.

    Progress file schema:
      {"batches_done": int, "total_loss_weighted": float, "total_tokens": int}
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    state = {"batches_done": 0, "total_loss_weighted": 0.0, "total_tokens": 0}
    if progress_path and os.path.exists(progress_path):
        loaded = load_json_safe(progress_path)
        if isinstance(loaded, dict) and "batches_done" in loaded:
            state.update(loaded)
            if log_fn:
                log_fn(f"  Resuming from batch {state['batches_done']} "
                       f"({state['total_tokens']} tokens already scored)")

    start_batch = state["batches_done"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < start_batch:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            try:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                mid = input_ids.size(0) // 2
                if mid == 0:
                    state["batches_done"] = batch_idx + 1
                    continue
                outputs = model(input_ids=input_ids[:mid], labels=labels[:mid])
                loss = outputs.loss
                attention_mask = attention_mask[:mid]

            num_tokens = int((attention_mask == 1).sum().item())
            state["total_loss_weighted"] += float(loss.item()) * num_tokens
            state["total_tokens"] += num_tokens
            state["batches_done"] = batch_idx + 1

            if progress_path and (batch_idx + 1) % flush_every == 0:
                atomic_save_json(state, progress_path)

    if progress_path:
        atomic_save_json(state, progress_path)

    tot = state["total_tokens"]
    avg_loss = state["total_loss_weighted"] / tot if tot > 0 else float('inf')
    return {
        "perplexity": float(np.exp(avg_loss)),
        "loss": avg_loss,
        "total_tokens": tot,
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

def atomic_save_json(data, filepath, indent=2, verbose=False):
    """
    Atomically write JSON to filepath.

    Writes to a sibling .tmp file, fsyncs, then renames over the target.
    This ensures that a partially written file is never observed by a
    resuming process (critical for Azure spot deallocations).
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=indent)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, filepath)
    if verbose:
        print(f"✓ Saved to: {filepath}")


def save_json(data, filepath, indent=2):
    """Save data to JSON file (atomic write, resilient to mid-write crashes)."""
    atomic_save_json(data, filepath, indent=indent, verbose=True)


def load_json(filepath):
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)


def load_json_safe(filepath, default=None):
    """
    Load JSON, returning a default on missing or corrupt file.

    Used by resume paths where a half-flushed file from a killed process
    must not abort the run.
    """
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def append_jsonl(filepath, record):
    """
    Append a single JSON record as one line with flush + fsync.

    Enables incremental result persistence for long attack loops: even
    if the VM is deallocated mid-loop, all completed records are durable
    and a restarted run will skip them.

    Defensively handles the case where a previous interrupted write left
    a partial line without a trailing newline by starting a fresh line
    before the new record. The malformed fragment is preserved but
    isolated; `load_jsonl` will stop at it on the next read.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    line = json.dumps(record)
    prefix = ""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefix = "\n"
    with open(filepath, 'a') as f:
        f.write(prefix + line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def load_jsonl(filepath):
    """
    Load all valid records from a JSONL file.

    Malformed lines are skipped (not fatal). This is the right behaviour
    for our resume pattern: a crash can leave a truncated line, but the
    subsequent restart appends valid records afterwards and the caller
    uses per-record indices to de-duplicate, so silently dropping the
    corrupt fragment is both safe and correct.
    """
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def partial_results_dir(subdir="partial"):
    """
    Return a per-stage partial-results directory on the persistent
    results disk. All resume state (per-batch perplexity progress,
    per-model attack output, per-restart UAT state) lives here so a
    spot deallocation never loses finished work.
    """
    base = os.path.join(Config.RESULTS_DIR, subdir)
    os.makedirs(base, exist_ok=True)
    return base


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
