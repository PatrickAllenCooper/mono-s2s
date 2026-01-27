#!/usr/bin/env python3
"""
Local Verification Script

Runs quick checks to verify the pipeline works before HPC deployment.
Uses tiny models and minimal data for fast validation.

Usage:
    python verify_local.py
    python verify_local.py --verbose
    python verify_local.py --stage 0  # Test specific stage
"""

import argparse
import sys
import os
import tempfile
import torch
import shutil

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds,
    make_model_monotonic,
    compute_perplexity,
    save_json,
    create_completion_flag,
    check_dependencies,
    StageLogger,
    LanguageModelingDataset,
)


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def verify_config():
    """Verify configuration is valid"""
    print_section("VERIFYING CONFIGURATION")
    
    checks = []
    
    # Check model name
    if Config.MODEL_NAME and len(Config.MODEL_NAME) > 0:
        checks.append(("✓", f"Model name: {Config.MODEL_NAME}"))
    else:
        checks.append(("✗", "Model name not set"))
    
    # Check batch size
    if Config.BATCH_SIZE > 0:
        checks.append(("✓", f"Batch size: {Config.BATCH_SIZE}"))
    else:
        checks.append(("✗", f"Invalid batch size: {Config.BATCH_SIZE}"))
    
    # Check learning rates
    if 1e-6 <= Config.RECOVERY_LR <= 1e-3:
        checks.append(("✓", f"Learning rate: {Config.RECOVERY_LR}"))
    else:
        checks.append(("✗", f"Invalid learning rate: {Config.RECOVERY_LR}"))
    
    # Check seeds
    if len(Config.RANDOM_SEEDS) == 5:
        checks.append(("✓", f"Random seeds: {Config.RANDOM_SEEDS}"))
    else:
        checks.append(("✗", f"Wrong number of seeds: {len(Config.RANDOM_SEEDS)}"))
    
    # Check device
    device = Config.get_device()
    checks.append(("✓", f"Device: {device}"))
    
    for symbol, message in checks:
        print(f"  {symbol} {message}")
    
    failures = [msg for sym, msg in checks if sym == "✗"]
    return len(failures) == 0


def verify_imports():
    """Verify all required packages can be imported"""
    print_section("VERIFYING IMPORTS")
    
    imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
    ]
    
    checks = []
    for module_name, display_name in imports:
        try:
            __import__(module_name)
            checks.append(("✓", f"{display_name}"))
        except ImportError:
            checks.append(("✗", f"{display_name} - NOT INSTALLED"))
    
    for symbol, message in checks:
        print(f"  {symbol} {message}")
    
    failures = [msg for sym, msg in checks if sym == "✗"]
    return len(failures) == 0


def verify_monotonicity():
    """Verify monotonicity application works"""
    print_section("VERIFYING MONOTONICITY APPLICATION")
    
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        
        # Create tiny model
        print("  Creating tiny GPT-2 model for testing...")
        config = GPT2Config(
            vocab_size=100,
            n_positions=64,
            n_embd=64,
            n_layer=2,
            n_head=2,
            n_inner=256,
        )
        model = GPT2LMHeadModel(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"    Model created: {num_params:,} parameters")
        
        # Apply monotonicity
        print("  Applying monotonicity constraints...")
        monotonic_model = make_model_monotonic(model)
        print("    ✓ Monotonicity applied")
        
        # Verify weights
        print("  Verifying weight constraints...")
        min_weight = float('inf')
        max_weight = float('-inf')
        num_constrained = 0
        
        for name, param in monotonic_model.named_parameters():
            if 'weight' in name and any(x in name.lower() for x in ['mlp', 'fc', 'dense']):
                min_weight = min(min_weight, param.data.min().item())
                max_weight = max(max_weight, param.data.max().item())
                num_constrained += 1
        
        if min_weight >= -1e-6:
            print(f"    ✓ All FFN weights non-negative")
            print(f"      Min weight: {min_weight:.6f}")
            print(f"      Max weight: {max_weight:.6f}")
            print(f"      Constrained layers: {num_constrained}")
        else:
            print(f"    ✗ Found negative weights: min = {min_weight:.6f}")
            return False
        
        # Test forward pass
        print("  Testing forward pass...")
        x = torch.randint(0, 100, (2, 32))
        with torch.no_grad():
            outputs = monotonic_model(input_ids=x)
        print(f"    ✓ Forward pass successful, output shape: {outputs.logits.shape}")
        
        # Test gradient flow
        print("  Testing gradient flow...")
        outputs = monotonic_model(input_ids=x)
        loss = outputs.logits.sum()
        loss.backward()
        
        has_grads = any(p.grad is not None for p in monotonic_model.parameters() if p.requires_grad)
        if has_grads:
            print("    ✓ Gradients computed successfully")
        else:
            print("    ✗ No gradients computed")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_training():
    """Verify training loop works"""
    print_section("VERIFYING TRAINING LOOP")
    
    try:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW
        
        # Create tiny model
        print("  Creating tiny model...")
        config = GPT2Config(vocab_size=100, n_positions=64, n_embd=64, 
                           n_layer=2, n_head=2, n_inner=256)
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create dummy data
        print("  Creating dummy training data...")
        texts = ["This is a test sentence."] * 10
        dataset = LanguageModelingDataset(texts, tokenizer, max_length=64)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Setup training
        print("  Setting up optimizer...")
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, 2, 10)
        
        # Training loop
        print("  Running 3 training steps...")
        model.train()
        device = torch.device('cpu')
        
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            print(f"    Step {i+1}: loss = {loss.item():.4f}")
        
        print("    ✓ Training loop completed successfully")
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_perplexity():
    """Verify perplexity computation works"""
    print_section("VERIFYING PERPLEXITY COMPUTATION")
    
    try:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        from torch.utils.data import DataLoader
        
        # Create model
        print("  Creating model...")
        config = GPT2Config(vocab_size=100, n_positions=64, n_embd=64,
                           n_layer=2, n_head=2, n_inner=256)
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create data
        print("  Creating evaluation data...")
        texts = ["Test sentence for evaluation."] * 5
        dataset = LanguageModelingDataset(texts, tokenizer, max_length=64)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Compute perplexity
        print("  Computing perplexity...")
        device = torch.device('cpu')
        model.eval()
        result = compute_perplexity(model, dataloader, device)
        
        print(f"    Perplexity: {result['perplexity']:.2f}")
        print(f"    Loss: {result['loss']:.4f}")
        print(f"    Tokens: {result['total_tokens']}")
        
        # Verify exp(loss) = perplexity
        import numpy as np
        expected_ppl = np.exp(result['loss'])
        if abs(result['perplexity'] - expected_ppl) < 0.01:
            print(f"    ✓ Perplexity = exp(loss) verified")
        else:
            print(f"    ✗ Perplexity mismatch: {result['perplexity']} vs {expected_ppl}")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_file_operations():
    """Verify file I/O works"""
    print_section("VERIFYING FILE OPERATIONS")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON save/load
            print("  Testing JSON operations...")
            test_data = {'test': 123, 'list': [1, 2, 3]}
            json_path = os.path.join(tmpdir, 'test.json')
            save_json(test_data, json_path)
            
            from utils.common_utils import load_json
            loaded = load_json(json_path)
            
            if loaded == test_data:
                print("    ✓ JSON save/load works")
            else:
                print("    ✗ JSON roundtrip failed")
                return False
            
            # Test completion flags
            print("  Testing completion flags...")
            create_completion_flag('test_stage', work_dir=tmpdir)
            
            from utils.common_utils import check_completion_flag
            if check_completion_flag('test_stage', work_dir=tmpdir):
                print("    ✓ Completion flags work")
            else:
                print("    ✗ Completion flag check failed")
                return False
            
            # Test stage logger
            print("  Testing stage logger...")
            logger = StageLogger('test_stage', log_dir=tmpdir)
            logger.log("Test message")
            
            if os.path.exists(logger.log_file):
                print("    ✓ Stage logger works")
            else:
                print("    ✗ Stage logger failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def verify_determinism():
    """Verify reproducibility"""
    print_section("VERIFYING DETERMINISM")
    
    try:
        # Test set_all_seeds
        print("  Testing random seed setting...")
        set_all_seeds(42)
        vals1 = [torch.rand(1).item() for _ in range(10)]
        
        set_all_seeds(42)
        vals2 = [torch.rand(1).item() for _ in range(10)]
        
        if vals1 == vals2:
            print("    ✓ Random seeds produce reproducible results")
        else:
            print("    ✗ Random seeds not reproducible")
            return False
        
        # Test generator
        print("  Testing generator reproducibility...")
        from utils.common_utils import get_generator
        
        gen1 = get_generator(device='cpu', seed=42)
        vals1 = [torch.rand(1, generator=gen1).item() for _ in range(10)]
        
        gen2 = get_generator(device='cpu', seed=42)
        vals2 = [torch.rand(1, generator=gen2).item() for _ in range(10)]
        
        if vals1 == vals2:
            print("    ✓ Generator produces reproducible results")
        else:
            print("    ✗ Generator not reproducible")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def run_all_verifications(verbose=False):
    """Run all verification checks"""
    print("\n" + "="*80)
    print("  FOUNDATION LLM PIPELINE - LOCAL VERIFICATION")
    print("="*80)
    print()
    print("This script verifies the pipeline works locally before HPC deployment.")
    print("Using tiny models and minimal data for fast testing.")
    print()
    
    results = {}
    
    # Run checks
    results['config'] = verify_config()
    results['imports'] = verify_imports()
    results['determinism'] = verify_determinism()
    results['file_ops'] = verify_file_operations()
    results['monotonicity'] = verify_monotonicity()
    results['perplexity'] = verify_perplexity()
    results['training'] = verify_training()
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    print()
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"  Total: {passed_checks}/{total_checks} checks passed")
    print()
    
    if all(results.values()):
        print("="*80)
        print("  ✓ ALL VERIFICATIONS PASSED")
        print("  Pipeline is ready for HPC deployment!")
        print("="*80)
        print()
        print("Next steps:")
        print("  1. Review any warnings above")
        print("  2. Run pytest for comprehensive tests: pytest tests/ -v")
        print("  3. Submit to HPC: bash run_all.sh")
        print()
        return 0
    else:
        print("="*80)
        print("  ✗ SOME VERIFICATIONS FAILED")
        print("  Please fix issues before HPC deployment")
        print("="*80)
        print()
        print("Debugging steps:")
        print("  1. Check failed verifications above")
        print("  2. Verify dependencies installed: pip install -r requirements.txt")
        print("  3. Run tests with verbose: python verify_local.py --verbose")
        print("  4. Run pytest for detailed errors: pytest tests/ -v")
        print()
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Local verification for foundation LLM pipeline")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--stage', type=int, help="Test specific stage only (0-7)")
    
    args = parser.parse_args()
    
    if args.stage is not None:
        print(f"Testing stage {args.stage} only...")
        # TODO: Implement stage-specific testing
        print("Stage-specific testing not yet implemented.")
        print("Run without --stage for full verification.")
        return 1
    
    return run_all_verifications(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
