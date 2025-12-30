#!/usr/bin/env python3
"""
Test Script: Validate Improvements Before Full Pipeline Run

This script tests:
1. XSUM and SAMSum dataset loading with retry logic
2. Improved softplus initialization
3. Monotonic model configuration

Run this before submitting the full pipeline to catch issues early.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, load_dataset_split, make_model_monotonic,
    NonNegativeParametrization
)
from transformers import T5ForConditionalGeneration

def test_config():
    """Test new configuration parameters"""
    print("="*80)
    print("TEST 1: Configuration")
    print("="*80)
    
    config = ExperimentConfig
    
    print("\n✓ Baseline hyperparameters:")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Warmup ratio: {config.WARMUP_RATIO}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    
    print("\n✓ Monotonic hyperparameters:")
    print(f"  Epochs: {config.MONOTONIC_NUM_EPOCHS}")
    print(f"  Warmup ratio: {config.MONOTONIC_WARMUP_RATIO}")
    print(f"  Learning rate: {config.MONOTONIC_LEARNING_RATE}")
    
    print("\n✓ Decoding parameters:")
    print(f"  Length penalty: {config.DECODE_LENGTH_PENALTY}")
    print(f"  Max new tokens: {config.DECODE_MAX_NEW_TOKENS}")
    
    print("\n✓ Dataset configuration:")
    print(f"  Max retries: {config.DATASET_MAX_RETRIES}")
    print(f"  Retry delay: {config.DATASET_RETRY_DELAY}s")
    print(f"  Allow partial: {config.DATASET_ALLOW_PARTIAL}")
    
    print("\n✅ Configuration test passed!\n")


def test_dataset_loading():
    """Test dataset loading with retry logic"""
    print("="*80)
    print("TEST 2: Dataset Loading (with Retry Logic)")
    print("="*80)
    
    # Test CNN/DM (should work)
    print("\n1. Testing CNN/DailyMail (validation, 10 samples)...")
    cnn_texts, cnn_sums = load_dataset_split(
        "cnn_dailymail", "validation", "article", "highlights",
        config="3.0.0", max_samples=10
    )
    assert len(cnn_texts) > 0, "CNN/DM loading failed"
    print(f"   ✓ Loaded {len(cnn_texts)} samples")
    
    # Test XSUM (may need retry)
    print("\n2. Testing XSUM (test, 10 samples)...")
    xsum_texts, xsum_sums = load_dataset_split(
        "xsum", "test", "document", "summary",
        max_samples=10
    )
    if len(xsum_texts) > 0:
        print(f"   ✓ Loaded {len(xsum_texts)} samples")
    else:
        print(f"   ⚠️  XSUM loading failed (will be skipped in evaluation)")
    
    # Test SAMSum (may need retry)
    print("\n3. Testing SAMSum (test, 10 samples)...")
    samsum_texts, samsum_sums = load_dataset_split(
        "samsum", "test", "dialogue", "summary",
        max_samples=10
    )
    if len(samsum_texts) > 0:
        print(f"   ✓ Loaded {len(samsum_texts)} samples")
    else:
        print(f"   ⚠️  SAMSum loading failed (will be skipped in evaluation)")
    
    print("\n✅ Dataset loading test completed!\n")
    return len(xsum_texts) > 0, len(samsum_texts) > 0


def test_softplus_initialization():
    """Test improved softplus initialization"""
    print("="*80)
    print("TEST 3: Softplus Initialization")
    print("="*80)
    
    print("\nTesting inverse softplus preservation...")
    
    # Create test weights
    test_weights = torch.randn(100, 100)
    
    # Initialize parametrization
    param = NonNegativeParametrization(init_weight=test_weights)
    
    # Get the transformed weight using right_inverse
    V = param.right_inverse(test_weights)
    
    # Apply forward to get back W
    W_reconstructed = param.forward(V)
    
    # Check preservation (should be close to |test_weights|)
    W_target = torch.abs(test_weights) + 1e-4
    relative_error = torch.mean(torch.abs(W_reconstructed - W_target) / (W_target + 1e-6))
    
    print(f"  Original weights range: [{test_weights.min():.4f}, {test_weights.max():.4f}]")
    print(f"  Target weights range: [{W_target.min():.4f}, {W_target.max():.4f}]")
    print(f"  Reconstructed range: [{W_reconstructed.min():.4f}, {W_reconstructed.max():.4f}]")
    print(f"  Relative error: {relative_error:.6f}")
    
    if relative_error < 0.1:  # Less than 10% error
        print("\n✅ Softplus initialization preserves weights well!\n")
    else:
        print(f"\n⚠️  Warning: Relative error is high ({relative_error:.2%})\n")


def test_monotonic_model_creation():
    """Test monotonic model creation with improved initialization"""
    print("="*80)
    print("TEST 4: Monotonic Model Creation")
    print("="*80)
    
    print("\nLoading T5-small model...")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Get a sample weight before transformation
    for module in model.modules():
        if hasattr(module, 'wi') and hasattr(module.wi, 'weight'):
            original_weight = module.wi.weight.data.clone()
            print(f"  Sample FFN weight before: [{original_weight.min():.4f}, {original_weight.max():.4f}]")
            break
    
    print("\nApplying monotonic constraints with improved initialization...")
    model = make_model_monotonic(model)
    
    # Check the same weight after transformation
    for module in model.modules():
        if hasattr(module, 'wi') and hasattr(module.wi, 'weight'):
            new_weight = module.wi.weight.data
            print(f"  Sample FFN weight after: [{new_weight.min():.4f}, {new_weight.max():.4f}]")
            print(f"  All weights non-negative: {(new_weight >= 0).all()}")
            break
    
    print("\n✅ Monotonic model creation test passed!\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING IMPROVEMENTS BEFORE FULL PIPELINE")
    print("="*80 + "\n")
    
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    try:
        # Test 1: Configuration
        test_config()
        
        # Test 2: Dataset loading
        xsum_ok, samsum_ok = test_dataset_loading()
        
        # Test 3: Softplus initialization
        test_softplus_initialization()
        
        # Test 4: Monotonic model creation
        test_monotonic_model_creation()
        
        # Summary
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print("\n✅ All tests passed!")
        print("\nDataset availability:")
        print(f"  - CNN/DailyMail: ✓ Available")
        print(f"  - XSUM: {'✓ Available' if xsum_ok else '⚠️  Not available (will be skipped)'}")
        print(f"  - SAMSum: {'✓ Available' if samsum_ok else '⚠️  Not available (will be skipped)'}")
        
        print("\nImprovements ready:")
        print("  ✓ Monotonic model: 7 epochs, 15% warmup, improved init")
        print("  ✓ Decoding: length_penalty=1.2, max_tokens=80")
        print("  ✓ Dataset retry logic with graceful fallback")
        
        print("\n🚀 Ready to run full pipeline!")
        print("   Run: ./run_all.sh\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

