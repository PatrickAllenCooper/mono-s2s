#!/usr/bin/env python3
"""
Stage 0: Setup and Environment Verification

This stage:
1. Sets up environment variables for determinism
2. Verifies Python/PyTorch installation
3. Downloads and verifies T5 model
4. Creates directory structure
5. Logs environment details

Outputs:
- setup_complete.json (environment info + verification)
- stage_0_complete.flag
"""

# Set environment variables BEFORE importing torch
import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, log_environment, create_completion_flag,
    save_json, StageLogger
)

def main():
    """Run setup stage"""
    logger = StageLogger("stage_0_setup")
    
    try:
        # Set seeds
        logger.log("Setting random seeds...")
        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        
        # Create directories
        logger.log("Creating directory structure...")
        ExperimentConfig.create_directories()
        
        # Log environment
        logger.log("Logging environment details...")
        env_info = log_environment()
        
        # Validate configuration
        logger.log("Validating configuration...")
        if not ExperimentConfig.validate_config():
            logger.log("⚠️  Configuration validation failed (non-fatal)")
        
        # Test PyTorch + CUDA
        logger.log("Testing PyTorch and CUDA...")
        import torch
        
        if torch.cuda.is_available():
            logger.log(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.log(f"  CUDA version: {torch.version.cuda}")
            logger.log(f"  cuDNN version: {torch.backends.cudnn.version()}")
            
            # Test GPU with small tensor
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x.t())
            logger.log(f"✓ GPU tensor operations working")
        else:
            logger.log("⚠️  No CUDA available - training will be VERY slow!")
        
        # Download and verify T5 model
        logger.log(f"Downloading T5 model: {ExperimentConfig.MODEL_NAME}...")
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        tokenizer = T5Tokenizer.from_pretrained(ExperimentConfig.MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(ExperimentConfig.MODEL_NAME)
        
        # Verify T5 architecture
        assert model.config.model_type == "t5", \
            f"ERROR: Expected T5, got {model.config.model_type}"
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.log(f"✓ Model verified: {ExperimentConfig.MODEL_NAME}")
        logger.log(f"  Architecture: {model.config.model_type}")
        logger.log(f"  Parameters: {num_params:,}")
        logger.log(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Save setup results
        setup_results = {
            "stage": "setup",
            "status": "success",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "seed": ExperimentConfig.CURRENT_SEED,
            "environment": env_info,
            "model": {
                "name": ExperimentConfig.MODEL_NAME,
                "type": model.config.model_type,
                "parameters": num_params,
                "vocab_size": tokenizer.vocab_size
            },
            "configuration": ExperimentConfig.to_dict()
        }
        
        output_file = os.path.join(ExperimentConfig.RESULTS_DIR, "setup_complete.json")
        save_json(setup_results, output_file)
        
        logger.complete(success=True)
        return 0
        
    except Exception as e:
        logger.log(f"\n❌ ERROR in setup stage: {e}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

