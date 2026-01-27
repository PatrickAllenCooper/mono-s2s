#!/usr/bin/env python3
"""
Stage 0: Setup for Foundation LLM Experiments

Downloads Pythia-1.4B model and prepares environment.

Outputs:
- Downloaded model in cache
- setup_complete.json with environment info
- stage_0_setup_complete.flag
"""

import os
import sys
import torch
import platform

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, create_completion_flag, save_json, StageLogger
)

def main():
    """Run setup stage"""
    logger = StageLogger("stage_0_setup")
    
    try:
        # Set seeds
        logger.log("Setting random seeds...")
        set_all_seeds(Config.CURRENT_SEED)
        
        # Create directories
        logger.log("Creating experiment directories...")
        Config.create_directories()
        
        # Validate configuration
        logger.log("Validating configuration...")
        if not Config.validate_config():
            logger.complete(success=False)
            return 1
        
        # Get device info
        device = Config.get_device()
        
        # Download model
        logger.log(f"Downloading model: {Config.MODEL_NAME}")
        logger.log("This may take several minutes...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Download tokenizer
        logger.log("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            revision=Config.MODEL_REVISION,
            cache_dir=Config.DATA_CACHE_DIR
        )
        logger.log(f"✓ Tokenizer loaded: vocab size = {len(tokenizer)}")
        
        # Download model
        logger.log("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            revision=Config.MODEL_REVISION,
            cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.log(f"✓ Model loaded: {Config.MODEL_NAME}")
        logger.log(f"  Total parameters: {num_params:,}")
        logger.log(f"  Trainable parameters: {num_trainable:,}")
        logger.log(f"  Model dtype: {model.dtype}")
        
        # Count FFN parameters
        ffn_params = 0
        for name, param in model.named_parameters():
            if any(x in name.lower() for x in ['mlp', 'dense_h_to_4h', 'dense_4h_to_h']):
                ffn_params += param.numel()
        
        logger.log(f"  FFN parameters: {ffn_params:,} ({100*ffn_params/num_params:.1f}%)")
        
        # Test model
        logger.log("\nTesting model forward pass...")
        test_input = tokenizer("Hello, world!", return_tensors="pt")
        if torch.cuda.is_available():
            model = model.to(device)
            test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            outputs = model(**test_input)
            logger.log(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")
        
        # Save setup info
        setup_info = {
            'timestamp': torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else None,
            'model_name': Config.MODEL_NAME,
            'model_revision': Config.MODEL_REVISION,
            'seed': Config.CURRENT_SEED,
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable,
            'ffn_parameters': ffn_params,
            'ffn_percentage': 100 * ffn_params / num_params,
            'vocab_size': len(tokenizer),
            'max_seq_length': Config.MAX_SEQ_LENGTH,
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'pytorch_version': torch.__version__,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
        }
        
        if torch.cuda.is_available():
            setup_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        
        save_json(
            setup_info,
            os.path.join(Config.RESULTS_DIR, 'setup_complete.json')
        )
        
        logger.log("\n" + "="*80)
        logger.log("SETUP SUMMARY")
        logger.log("="*80)
        logger.log(f"Model: {Config.MODEL_NAME}")
        logger.log(f"Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
        logger.log(f"FFN Parameters: {ffn_params:,} ({100*ffn_params/num_params:.1f}%)")
        logger.log(f"Device: {device}")
        logger.log(f"Seed: {Config.CURRENT_SEED}")
        logger.log("="*80)
        
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
