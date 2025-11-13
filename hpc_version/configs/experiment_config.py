"""
Centralized Configuration for HPC Mono-S2S Experiments

This file contains all hyperparameters, paths, and settings for reproducible
fair comparison experiments on HPC clusters.

Edit this file to customize your experiment before submitting jobs.
"""

import os
import torch

class ExperimentConfig:
    """Centralized configuration for reproducibility"""
    
    # ======================================================================
    # PATHS (CUSTOMIZE FOR YOUR HPC ENVIRONMENT)
    # ======================================================================
    
    # HPC paths - EDIT THESE FOR YOUR CLUSTER
    # Default to environment variables, fall back to placeholders
    # On CURC clusters, $SCRATCH and $PROJECT are automatically set
    SCRATCH_DIR = os.environ.get("SCRATCH", "/scratch/summit/your_username")
    PROJECT_DIR = os.environ.get("PROJECT", "/projects/your_project")
    
    # CURC Cluster-Specific Examples:
    # Summit:  SCRATCH=/scratch/summit/$USER, PROJECT=/projects/$USER
    # Alpine:  SCRATCH=/scratch/alpine/$USER, PROJECT=/pl/active/$USER
    # Blanca:  SCRATCH=/rc_scratch/$USER, PROJECT=/projects/your_group
    
    # Experiment directories (created automatically)
    WORK_DIR = os.path.join(SCRATCH_DIR, "mono_s2s_work")
    RESULTS_DIR = os.path.join(SCRATCH_DIR, "mono_s2s_results")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    DATA_CACHE_DIR = os.path.join(WORK_DIR, "data_cache")
    
    # Final results (copy to project for persistence)
    FINAL_RESULTS_DIR = os.path.join(PROJECT_DIR, "mono_s2s_final_results")
    
    # ======================================================================
    # RANDOM SEEDS (Multi-seed Experiments)
    # ======================================================================
    
    RANDOM_SEEDS = [42, 1337, 2024, 8888, 12345]
    CURRENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))  # Can set via SLURM
    
    # ======================================================================
    # MODEL CONFIGURATION
    # ======================================================================
    
    MODEL_NAME = "t5-small"  # Guaranteed T5 checkpoint (t5-base, t5-large also work)
    
    # ======================================================================
    # TRAINING HYPERPARAMETERS (IDENTICAL for baseline and monotonic)
    # ======================================================================
    
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.1
    
    # ======================================================================
    # TOKENIZATION PARAMETERS (IDENTICAL for all)
    # ======================================================================
    
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128
    
    # ======================================================================
    # DECODING PARAMETERS (IDENTICAL for all models)
    # ======================================================================
    
    DECODE_NUM_BEAMS = 4
    DECODE_LENGTH_PENALTY = 1.0
    DECODE_MIN_NEW_TOKENS = 10
    DECODE_MAX_NEW_TOKENS = 128
    DECODE_NO_REPEAT_NGRAM_SIZE = 3
    DECODE_EARLY_STOPPING = True
    
    # ======================================================================
    # ROUGE CONFIGURATION (Frozen)
    # ======================================================================
    
    ROUGE_METRICS = ["rouge1", "rouge2", "rougeLsum"]
    ROUGE_USE_STEMMER = True
    ROUGE_BOOTSTRAP_SAMPLES = 1000
    
    # ======================================================================
    # ATTACK CONFIGURATION
    # ======================================================================
    
    ATTACK_TRIGGER_LENGTH = 5
    ATTACK_NUM_CANDIDATES = 100
    ATTACK_NUM_GRAD_STEPS = 50
    ATTACK_NUM_RESTARTS = 3
    ATTACK_NUM_ITERATIONS = 50
    
    # ======================================================================
    # EVALUATION CONFIGURATION
    # ======================================================================
    
    USE_FULL_TEST_SETS = True  # Set to False for quick testing (200 samples)
    EVAL_BATCH_SIZE = 8
    
    # Quick testing sizes (when USE_FULL_TEST_SETS=False)
    QUICK_TEST_SIZE = 200
    TRIGGER_OPT_SIZE_QUICK = 80
    TRIGGER_EVAL_SIZE_QUICK = 120
    
    # Full evaluation sizes (when USE_FULL_TEST_SETS=True)
    TRIGGER_OPT_SIZE_FULL = 500
    TRIGGER_EVAL_SIZE_FULL = 1000
    
    # ======================================================================
    # DATASET CONFIGURATION
    # ======================================================================
    
    # Training datasets
    TRAIN_DATASETS = [
        ("knkarthick/dialogsum", "dialogue", "summary", "DialogSum"),
        ("samsum", "dialogue", "summary", "SAMSum"),
        ("EdinburghNLP/xsum", "document", "summary", "XSUM"),
        ("knkarthick/AMI", "dialogue", "summary", "AMI"),
        ("knkarthick/highlightsum", "dialogue", "summary", "HighlightSum"),
        ("ccdv/arxiv-summarization", "article", "abstract", "arXiv"),
        ("knkarthick/MEETING_SUMMARY", "dialogue", "summary", "MEETING_SUMMARY"),
    ]
    
    # Test datasets
    TEST_DATASETS = [
        ("cnn_dailymail", "3.0.0", "article", "highlights", "CNN/DM"),
        ("EdinburghNLP/xsum", None, "document", "summary", "XSUM"),
        ("samsum", None, "dialogue", "summary", "SAMSum"),
    ]
    
    # ======================================================================
    # HPC-SPECIFIC SETTINGS
    # ======================================================================
    
    # SLURM partition (edit for your cluster)
    # Summit: "shas" (Haswell GPU nodes) or "sgpu" (general GPU)
    # Alpine: "aa100" (A100 GPUs) or "ami100" (MI100 GPUs)
    # Blanca: "blanca-ics" or your specific condo partition
    SLURM_PARTITION = "shas"  # Change this for your cluster
    SLURM_QOS = "normal"  # Or "blanca-ics" for Blanca
    
    # Resource requests
    SLURM_NODES = 1
    SLURM_TASKS_PER_NODE = 1
    SLURM_GPUS_PER_NODE = 1
    SLURM_MEM_GB = 64  # Total memory
    
    # Time limits (format: HH:MM:SS)
    TIME_SETUP = "00:30:00"      # 30 minutes
    TIME_DATA = "02:00:00"       # 2 hours
    TIME_TRAIN = "12:00:00"      # 12 hours (per model)
    TIME_EVALUATE = "04:00:00"   # 4 hours
    TIME_UAT = "03:00:00"        # 3 hours
    TIME_HOTFLIP = "02:00:00"    # 2 hours
    TIME_AGGREGATE = "00:15:00"  # 15 minutes
    
    # ======================================================================
    # LOGGING & CHECKPOINTING
    # ======================================================================
    
    SAVE_CHECKPOINT_EVERY_N_EPOCHS = 1
    LOG_EVERY_N_STEPS = 100
    VERBOSE_LOGGING = True
    
    # ======================================================================
    # HELPER METHODS
    # ======================================================================
    
    @classmethod
    def to_dict(cls):
        """Export configuration as dictionary for logging"""
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
        """Validate configuration before running"""
        issues = []
        
        # Check paths exist
        if not os.path.exists(cls.SCRATCH_DIR):
            issues.append(f"SCRATCH_DIR does not exist: {cls.SCRATCH_DIR}")
        if not os.path.exists(cls.PROJECT_DIR):
            issues.append(f"PROJECT_DIR does not exist: {cls.PROJECT_DIR}")
        
        # Check GPU for training stages
        if not torch.cuda.is_available():
            issues.append("No GPU available - training will be very slow!")
        
        # Warn about configuration
        if cls.USE_FULL_TEST_SETS and cls.BATCH_SIZE > 4:
            issues.append(f"Large batch size ({cls.BATCH_SIZE}) with full test sets may cause OOM")
        
        if issues:
            print("\n⚠️  Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✓ Configuration validated successfully")
            return True


# Module-level exports for convenience
def get_config():
    """Get configuration instance"""
    return ExperimentConfig

def get_paths():
    """Get all paths as dictionary"""
    config = ExperimentConfig
    return {
        'work_dir': config.WORK_DIR,
        'results_dir': config.RESULTS_DIR,
        'checkpoint_dir': config.CHECKPOINT_DIR,
        'data_cache_dir': config.DATA_CACHE_DIR,
        'final_results_dir': config.FINAL_RESULTS_DIR,
        'baseline_checkpoint_dir': os.path.join(config.CHECKPOINT_DIR, 'baseline_checkpoints'),
        'monotonic_checkpoint_dir': os.path.join(config.CHECKPOINT_DIR, 'monotonic_checkpoints'),
    }


if __name__ == "__main__":
    # Test configuration
    print("="*80)
    print("EXPERIMENT CONFIGURATION TEST")
    print("="*80)
    
    config = ExperimentConfig
    print(f"\nPaths:")
    print(f"  Work dir: {config.WORK_DIR}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print(f"  Checkpoint dir: {config.CHECKPOINT_DIR}")
    
    print(f"\nModel: {config.MODEL_NAME}")
    print(f"Seed: {config.CURRENT_SEED}")
    print(f"Full test sets: {config.USE_FULL_TEST_SETS}")
    
    print(f"\nValidating...")
    config.validate_config()
    
    print("\n✓ Configuration ready for HPC execution")

