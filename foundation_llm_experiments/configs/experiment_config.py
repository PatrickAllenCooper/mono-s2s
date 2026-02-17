"""
Configuration for Foundation LLM Monotonicity Experiments

This extends the mono-s2s experimental framework to general-purpose LLMs.
Key differences:
- Uses Pythia-1.4B instead of T5
- Evaluates on diverse benchmarks (not just summarization)
- Includes recovery training phase
- Different attack evaluation strategy
"""

import os
import torch

class FoundationExperimentConfig:
    """Configuration for foundation model experiments"""
    
    # ======================================================================
    # PATHS (CUSTOMIZE FOR YOUR HPC ENVIRONMENT)
    # ======================================================================
    
    SCRATCH_DIR = os.environ.get("SCRATCH", f"/scratch/alpine/{os.environ.get('USER', 'your_username')}")
    PROJECT_DIR = os.environ.get("PROJECT", f"/projects/{os.environ.get('USER', 'your_username')}")
    
    WORK_DIR = os.path.join(SCRATCH_DIR, "foundation_llm_work")
    RESULTS_DIR = os.path.join(SCRATCH_DIR, "foundation_llm_results")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    DATA_CACHE_DIR = os.path.join(WORK_DIR, "data_cache")
    
    FINAL_RESULTS_DIR = os.path.join(PROJECT_DIR, "foundation_llm_final_results")
    
    # ======================================================================
    # MODEL CONFIGURATION
    # ======================================================================
    
    # Foundation model selection
    MODEL_NAME = "EleutherAI/pythia-1.4b"  # 1.4B parameters, fits A100
    MODEL_REVISION = "main"  # Use latest checkpoint
    
    # Model architecture info
    HIDDEN_SIZE = 2048
    NUM_LAYERS = 24
    NUM_ATTENTION_HEADS = 16
    FFN_INTERMEDIATE_SIZE = 8192  # 4x hidden size
    
    # Estimated FFN parameters: ~560M (40% of 1.4B total)
    # FFN_PARAMS = 2 * HIDDEN_SIZE * FFN_INTERMEDIATE_SIZE * NUM_LAYERS
    #            = 2 * 2048 * 8192 * 24 ≈ 805M parameters
    
    # ======================================================================
    # RANDOM SEEDS
    # ======================================================================
    
    RANDOM_SEEDS = [42, 1337, 2024, 8888, 12345]
    CURRENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))
    
    # ======================================================================
    # TRAINING HYPERPARAMETERS
    # ======================================================================
    
    # Recovery training (restore perplexity after monotonicity init)
    RECOVERY_EPOCHS = 5  # 5 epochs for better baseline performance
    RECOVERY_LR = 1e-5
    RECOVERY_WARMUP_RATIO = 0.10
    RECOVERY_WEIGHT_DECAY = 0.01
    
    # Monotonic recovery (same data, extended warmup)
    MONOTONIC_RECOVERY_EPOCHS = 10  # 10 epochs - monotonic needs more training
    MONOTONIC_RECOVERY_LR = 1e-5
    MONOTONIC_RECOVERY_WARMUP_RATIO = 0.15  # More warmup for softplus stability
    MONOTONIC_RECOVERY_WEIGHT_DECAY = 0.01
    
    # Training batch sizes
    BATCH_SIZE = 8  # Per-device batch size
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32
    MAX_GRAD_NORM = 1.0
    
    # Sequence length
    MAX_SEQ_LENGTH = 2048  # Pythia's context window
    
    # ======================================================================
    # EVALUATION CONFIGURATION
    # ======================================================================
    
    # Evaluation batch size (can be larger than training)
    EVAL_BATCH_SIZE = 16
    
    # Evaluation benchmarks
    EVAL_BENCHMARKS = [
        "pile_test",      # Perplexity on Pile test set (primary metric)
        "lambada",        # Language modeling (next-word prediction)
        "hellaswag",      # Commonsense reasoning
        "winogrande",     # Coreference resolution
        "truthfulqa",     # Factuality
    ]
    
    # Evaluation dataset sizes
    USE_FULL_EVAL_SETS = True  # Set to False for quick testing
    
    # Quick testing sizes (when USE_FULL_EVAL_SETS=False)
    QUICK_EVAL_SIZE = 500
    QUICK_PILE_TEST_SIZE = 1000
    
    # Full evaluation sizes
    FULL_PILE_TEST_SIZE = 10000  # 10K examples from Pile test
    
    # ======================================================================
    # TRAINING DATA CONFIGURATION
    # ======================================================================
    
    # Training dataset (for recovery phase)
    # Note: Using pile-uncopyrighted (parquet-based) instead of old pile (deprecated script-based)
    TRAINING_DATASET = "monology/pile-uncopyrighted"
    TRAINING_SUBSET = "train"  # Use train split (validation also available)
    TRAINING_SAMPLES = None  # None = full dataset, set number for quick tests
    
    # For quick testing (set TRAINING_SAMPLES to this value for quick mode)
    QUICK_TRAINING_SAMPLES = 100000  # 100K samples for production runs
    
    # CRITICAL FIX: Use full test sets for publication-quality results
    # Set to False for quick testing (reduces runtime from 60h to 5h per seed)
    USE_FULL_EVAL_SETS = True  # Production mode: use full eval sets
    
    # ======================================================================
    # ATTACK CONFIGURATION
    # ======================================================================
    
    # Universal Adversarial Triggers
    ATTACK_TRIGGER_LENGTH = 10  # Longer for general LLM (vs 5 for summarization)
    ATTACK_NUM_CANDIDATES = 200  # Larger vocabulary search
    ATTACK_NUM_RESTARTS = 5
    ATTACK_NUM_ITERATIONS = 100
    
    # HotFlip attacks
    HOTFLIP_NUM_FLIPS = 10  # More flips for longer sequences
    HOTFLIP_NUM_SAMPLES = 200  # Test on diverse prompts
    
    # Attack evaluation
    ATTACK_LOSS_BATCH_SIZE = 8
    ATTACK_SUCCESS_THRESHOLD = 0.15  # 15% perplexity increase (vs 10% for ROUGE)
    
    # ======================================================================
    # HPC-SPECIFIC SETTINGS
    # ======================================================================
    
    SLURM_PARTITION = "aa100"
    SLURM_QOS = "normal"
    SLURM_NODES = 1
    SLURM_TASKS_PER_NODE = 1
    SLURM_GPUS_PER_NODE = 1
    SLURM_MEM_GB = 80  # A100 40GB + overhead
    
    # Time limits - MAX OUT for automatic resubmission strategy
    # SLURM typically limits to 24 hours, so we max out what we can
    # Jobs will checkpoint and auto-resubmit if they timeout
    TIME_SETUP = "01:00:00"          # 1 hour (download model)
    TIME_APPLY_MONOTONICITY = "00:30:00"  # 30 min (apply constraints)
    TIME_TRAIN_BASELINE = "23:50:00"  # MAX: 24 hours (5 epochs will need 5+ resubmissions)
    TIME_TRAIN_MONOTONIC = "23:50:00"  # MAX: 24 hours (10 epochs will need 10+ resubmissions)
    TIME_EVALUATE = "08:00:00"        # 8 hours (multiple benchmarks)
    TIME_UAT = "06:00:00"             # 6 hours (UAT optimization)
    TIME_HOTFLIP = "04:00:00"         # 4 hours (HotFlip attacks)
    TIME_AGGREGATE = "00:30:00"       # 30 min (aggregate results)
    
    # ======================================================================
    # LOGGING & CHECKPOINTING (CRITICAL FOR RESUME)
    # ======================================================================
    
    # Checkpoint frequently for resume capability
    SAVE_CHECKPOINT_EVERY_N_STEPS = 500  # Every 500 steps (~every 30-60 min)
    SAVE_CHECKPOINT_EVERY_N_MINUTES = 30  # Also save every 30 minutes (time-based backup)
    
    # Cleanup old checkpoints to save space (keep last N)
    KEEP_LAST_N_CHECKPOINTS = 3  # Keep last 3 checkpoints only
    
    LOG_EVERY_N_STEPS = 100
    VERBOSE_LOGGING = True
    
    # Auto-resubmission settings
    MAX_RESUBMISSIONS_PER_STAGE = 5  # Max times to resubmit a timed-out job
    RESUBMISSION_DELAY_SECONDS = 60  # Wait before resubmitting
    
    # ======================================================================
    # HELPER METHODS
    # ======================================================================
    
    @classmethod
    def to_dict(cls):
        """Export configuration as dictionary"""
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith('_') and not callable(v) and k.isupper()
        }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        dirs = [
            cls.WORK_DIR,
            cls.RESULTS_DIR,
            cls.CHECKPOINT_DIR,
            cls.DATA_CACHE_DIR,
            cls.FINAL_RESULTS_DIR,
            os.path.join(cls.CHECKPOINT_DIR, 'baseline_checkpoints'),
            os.path.join(cls.CHECKPOINT_DIR, 'monotonic_checkpoints'),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"✓ Created all directories under {cls.WORK_DIR}")
    
    @classmethod
    def get_device(cls):
        """Get compute device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print("⚠️  Using CPU (no GPU available)")
        return device
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        issues = []
        
        if not os.path.exists(cls.SCRATCH_DIR):
            issues.append(f"SCRATCH_DIR does not exist: {cls.SCRATCH_DIR}")
        if not os.path.exists(cls.PROJECT_DIR):
            issues.append(f"PROJECT_DIR does not exist: {cls.PROJECT_DIR}")
        
        if not torch.cuda.is_available():
            issues.append("No GPU available - training will be very slow!")
        
        if issues:
            print("\n⚠️  Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✓ Configuration validated successfully")
            return True
    
    @classmethod
    def estimate_training_time(cls):
        """Estimate total training time"""
        # Pile has ~300B tokens, 1 epoch with batch_size=8, seq_len=2048, grad_accum=4
        # tokens_per_step = 8 * 2048 * 4 = 65,536
        # steps_per_epoch = 300B / 65,536 ≈ 4.5M steps
        # At ~1 step/sec on A100: 4.5M sec ≈ 1250 hours ≈ 52 days
        # 
        # In practice, we'll use a subset or distributed training
        # For single A100, realistic: ~24 hours for baseline, ~32 for monotonic
        
        baseline_hours = 24
        monotonic_hours = 32
        eval_hours = 8
        attack_hours = 10
        total_hours = baseline_hours + monotonic_hours + eval_hours + attack_hours
        
        print("\nEstimated Training Time:")
        print(f"  Baseline Recovery: ~{baseline_hours} hours")
        print(f"  Monotonic Recovery: ~{monotonic_hours} hours")
        print(f"  Evaluation: ~{eval_hours} hours")
        print(f"  Attacks: ~{attack_hours} hours")
        print(f"  Total per seed: ~{total_hours} hours ({total_hours/24:.1f} days)")
        print(f"  Total 5 seeds: ~{total_hours*5} hours ({total_hours*5/24:.1f} days)")


if __name__ == "__main__":
    print("="*80)
    print("FOUNDATION LLM EXPERIMENT CONFIGURATION")
    print("="*80)
    
    config = FoundationExperimentConfig
    print(f"\nModel: {config.MODEL_NAME}")
    print(f"Seed: {config.CURRENT_SEED}")
    print(f"Work dir: {config.WORK_DIR}")
    
    print("\nValidating...")
    config.validate_config()
    
    config.estimate_training_time()
    
    print("\n✓ Configuration ready for experiments")
