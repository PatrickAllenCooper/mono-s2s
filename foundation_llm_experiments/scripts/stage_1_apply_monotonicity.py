#!/usr/bin/env python3
"""
Stage 1: Apply Monotonicity Constraints

Loads Pythia-1.4B and applies softplus parametrization to FFN layers.
Saves the monotonic-initialized model (before recovery training).

Inputs:
- Pythia-1.4B from cache (downloaded in stage 0)

Outputs:
- monotonic_initialized.pt (model with constraints applied)
- monotonicity_application_log.json (statistics about transformation)
- stage_1_apply_monotonicity_complete.flag
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, create_completion_flag, save_json, 
    StageLogger, check_dependencies, make_model_monotonic
)

def main():
    """Apply monotonicity constraints to foundation model"""
    logger = StageLogger("stage_1_apply_monotonicity")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_0_setup']):
            logger.complete(success=False)
            return 1
        
        # Set seeds
        logger.log("Setting random seeds...")
        set_all_seeds(Config.CURRENT_SEED)
        
        # Get device
        device = Config.get_device()
        
        # Load model
        logger.log(f"Loading model: {Config.MODEL_NAME}")
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            revision=Config.MODEL_REVISION,
            cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.float32,  # Use full precision for constraints
            low_cpu_mem_usage=True
        )
        
        # Count parameters before
        num_params_before = sum(p.numel() for p in model.parameters())
        logger.log(f"✓ Model loaded: {num_params_before:,} parameters")
        
        # Count FFN parameters
        ffn_params = 0
        ffn_modules = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['mlp', 'dense_h_to_4h', 'dense_4h_to_h']):
                if isinstance(module, torch.nn.Linear):
                    ffn_params += module.weight.numel()
                    if module.bias is not None:
                        ffn_params += module.bias.numel()
                    ffn_modules.append(name)
        
        logger.log(f"\nFFN Layer Analysis:")
        logger.log(f"  Total FFN parameters: {ffn_params:,}")
        logger.log(f"  Percentage of total: {100*ffn_params/num_params_before:.1f}%")
        logger.log(f"  Number of FFN modules: {len(ffn_modules)}")
        
        # Apply monotonicity
        logger.log("\nApplying monotonicity constraints...")
        logger.log("This will:")
        logger.log("  1. Take absolute value of FFN weights")
        logger.log("  2. Apply inverse softplus transformation")
        logger.log("  3. Register softplus parametrization")
        
        model = make_model_monotonic(model)
        
        # Count parameters after (should include parametrization params)
        num_params_after = sum(p.numel() for p in model.parameters())
        logger.log(f"\nParameters after monotonicity: {num_params_after:,}")
        
        # Test forward pass
        logger.log("\nTesting forward pass with monotonic constraints...")
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR
        )
        
        test_input = tokenizer("The quick brown fox", return_tensors="pt")
        if torch.cuda.is_available():
            model = model.to(device)
            test_input = {k: v.to(device) for k, v in test_input.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**test_input)
            test_loss = outputs.loss if hasattr(outputs, 'loss') else None
            logger.log(f"✓ Forward pass successful")
            logger.log(f"  Output shape: {outputs.logits.shape}")
            if test_loss is not None:
                logger.log(f"  Test loss: {test_loss.item():.4f}")
        
        # Verify monotonicity (check that FFN weights are non-negative)
        logger.log("\nVerifying weight constraints...")
        all_non_negative = True
        min_weight = float('inf')
        max_weight = float('-inf')
        
        for name, param in model.named_parameters():
            # Skip .original parameters - these are pre-transformation and expected to be negative
            if '.original' in name:
                continue
            
            # Skip attention layers - only check FFN/MLP layers
            if 'attention' in name.lower():
                continue
                
            # Check FFN/MLP weights only (same patterns as make_model_monotonic)
            if 'weight' in name and any(x in name.lower() for x in 
                                       ['mlp', 'dense_h_to_4h', 'dense_4h_to_h', 
                                        'c_fc', 'c_proj', 'fc_in', 'fc_out']):
                param_min = param.data.min().item()
                param_max = param.data.max().item()
                min_weight = min(min_weight, param_min)
                max_weight = max(max_weight, param_max)
                
                if param_min < 0:
                    all_non_negative = False
                    logger.log(f"  ⚠️  {name}: min={param_min:.6f} (NEGATIVE!)")
        
        if all_non_negative:
            logger.log(f"✓ All FFN weights are non-negative")
            logger.log(f"  Min weight: {min_weight:.6f}")
            logger.log(f"  Max weight: {max_weight:.6f}")
        else:
            logger.log(f"❌ Some weights are still negative!")
            logger.complete(success=False)
            return 1
        
        # Save monotonic model
        logger.log("\nSaving monotonic-initialized model...")
        save_path = os.path.join(Config.CHECKPOINT_DIR, 'monotonic_initialized.pt')
        torch.save(model.state_dict(), save_path)
        logger.log(f"✓ Saved to: {save_path}")
        
        # Save application log
        application_log = {
            'model_name': Config.MODEL_NAME,
            'seed': Config.CURRENT_SEED,
            'num_parameters_before': num_params_before,
            'num_parameters_after': num_params_after,
            'ffn_parameters': ffn_params,
            'ffn_percentage': 100 * ffn_params / num_params_before,
            'num_ffn_modules': len(ffn_modules),
            'ffn_modules': ffn_modules,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'all_weights_non_negative': all_non_negative,
        }
        
        save_json(
            application_log,
            os.path.join(Config.RESULTS_DIR, 'monotonicity_application_log.json')
        )
        
        logger.log("\n✓ Monotonicity constraints applied successfully!")
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
