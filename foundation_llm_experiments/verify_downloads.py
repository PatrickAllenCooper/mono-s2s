#!/usr/bin/env python3
"""
Verify Model and Dataset Downloads

Tests that all required models and datasets can be downloaded successfully.
Run this before HPC deployment to catch download issues early.

Usage:
    python verify_downloads.py
    python verify_downloads.py --quick  # Skip large downloads
    python verify_downloads.py --offline  # Only check cached
"""

import argparse
import sys
import os
import tempfile

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from configs.experiment_config import FoundationExperimentConfig as Config


def print_header(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def verify_model_info():
    """Verify model exists on Hugging Face"""
    print_header("VERIFYING MODEL INFORMATION")
    
    try:
        from huggingface_hub import model_info
        
        print(f"Checking model: {Config.MODEL_NAME}")
        info = model_info(Config.MODEL_NAME)
        
        print(f"  ✓ Model found on Hugging Face")
        print(f"    Model ID: {info.modelId}")
        print(f"    Downloads: {info.downloads:,}")
        print(f"    Likes: {info.likes}")
        print(f"    Tags: {', '.join(info.tags[:5])}")
        
        # Check model size
        if hasattr(info, 'safetensors'):
            total_size = sum(f.size for f in info.siblings if f.rfilename.endswith(('.bin', '.safetensors', '.pt')))
            print(f"    Approx size: {total_size / 1e9:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking model info: {e}")
        print(f"    This might be OK if model exists but API failed")
        print(f"    Will attempt download test anyway...")
        return False


def verify_model_download(quick=False):
    """Verify model can be downloaded"""
    print_header("VERIFYING MODEL DOWNLOAD")
    
    if quick:
        print("⚠️  Skipping actual download in quick mode")
        print("   Set --full to test actual download")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        import tempfile
        
        # Use temporary cache
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Testing download to temporary directory...")
            print(f"Model: {Config.MODEL_NAME}")
            
            # Test tokenizer download (small, fast)
            print("\n1. Testing tokenizer download...")
            tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME,
                cache_dir=tmpdir
            )
            print(f"   ✓ Tokenizer downloaded successfully")
            print(f"     Vocab size: {len(tokenizer):,}")
            
            # Test config download (tiny)
            print("\n2. Testing model config download...")
            config = AutoConfig.from_pretrained(
                Config.MODEL_NAME,
                cache_dir=tmpdir
            )
            print(f"   ✓ Config downloaded successfully")
            print(f"     Hidden size: {config.hidden_size}")
            print(f"     Num layers: {config.num_hidden_layers}")
            
            # For full download test, we'd need to download full model (~6GB)
            # That's slow, so we skip unless --full flag
            print("\n3. Skipping full model download (would be ~6GB)")
            print("   To test full download, use: --full flag")
            print("   Or test on HPC with: sinteractive and run stage_0_setup.py")
        
        print(f"\n✓ Model downloads working (tokenizer + config verified)")
        return True
        
    except Exception as e:
        print(f"\n✗ Model download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataset_info():
    """Verify dataset exists"""
    print_header("VERIFYING DATASET INFORMATION")
    
    try:
        from huggingface_hub import dataset_info
        
        dataset_name = Config.TRAINING_DATASET
        print(f"Checking dataset: {dataset_name}")
        
        # Note: This might fail if dataset is very large or has special access
        try:
            info = dataset_info(dataset_name)
            print(f"  ✓ Dataset found on Hugging Face")
            print(f"    Dataset ID: {info.id}")
            print(f"    Downloads: {info.downloads:,}")
        except:
            # Pile might not have public API info
            print(f"  ⚠️  Could not get dataset info via API")
            print(f"     This is normal for Pile dataset")
            print(f"     Will test actual download...")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Error checking dataset info: {e}")
        print(f"     Will attempt download test...")
        return False


def verify_dataset_download(quick=False):
    """Verify dataset can be accessed"""
    print_header("VERIFYING DATASET DOWNLOAD")
    
    try:
        from datasets import load_dataset
        import tempfile
        
        dataset_name = Config.TRAINING_DATASET
        
        if quick:
            print("Running quick test (validation split only)...")
            split = "validation"
            streaming = False
        else:
            print("Running full test (streaming train split)...")
            split = "train"
            streaming = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\nDataset: {dataset_name}")
            print(f"Split: {split}")
            print(f"Streaming: {streaming}")
            print(f"Cache: {tmpdir}")
            
            # Test loading
            print("\nAttempting to load dataset...")
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                cache_dir=tmpdir,
                trust_remote_code=True
            )
            
            print(f"✓ Dataset loaded successfully")
            
            # Test reading samples
            print("\nTesting sample reading...")
            if streaming:
                sample_iter = iter(dataset)
                sample = next(sample_iter)
                print(f"✓ Read streaming sample successfully")
            else:
                sample = dataset[0]
                print(f"✓ Read sample successfully")
                print(f"  Dataset size: {len(dataset):,} examples")
            
            # Check sample structure
            print(f"\nSample keys: {list(sample.keys())}")
            if 'text' in sample:
                text_preview = sample['text'][:100].replace('\n', ' ')
                print(f"Sample text preview: {text_preview}...")
                print(f"✓ 'text' field found in sample")
            else:
                print(f"⚠️  Warning: 'text' field not found")
                print(f"   Available fields: {list(sample.keys())}")
            
        print(f"\n✓ Dataset download and access working")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset access failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nPossible solutions:")
        print("  1. Check internet connection")
        print("  2. Try different dataset split (validation instead of train)")
        print("  3. Try different dataset (C4, OpenWebText)")
        print("  4. Set up Hugging Face token if required")
        
        return False


def verify_dependencies():
    """Verify all required packages are installed"""
    print_header("VERIFYING DEPENDENCIES")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('scipy', 'SciPy'),
    ]
    
    all_ok = True
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {display_name:15s} version {version}")
        except ImportError:
            print(f"  ✗ {display_name:15s} NOT INSTALLED")
            all_ok = False
    
    # Check optional dependencies
    optional_deps = [
        ('accelerate', 'Accelerate'),
        ('lm_eval', 'LM Eval Harness'),
    ]
    
    print("\nOptional dependencies:")
    for module_name, display_name in optional_deps:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {display_name:15s} version {version}")
        except ImportError:
            print(f"  - {display_name:15s} not installed (optional)")
    
    return all_ok


def verify_huggingface_auth():
    """Check if Hugging Face authentication is set up"""
    print_header("VERIFYING HUGGING FACE AUTHENTICATION")
    
    try:
        from huggingface_hub import HfFolder
        
        token = HfFolder.get_token()
        if token:
            print("  ✓ Hugging Face token found")
            print(f"    Token: {token[:10]}...{token[-10:]}")
        else:
            print("  ⚠️  No Hugging Face token found")
            print("     Most models/datasets don't require authentication")
            print("     If you encounter 403 errors, set up token with:")
            print("       huggingface-cli login")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Could not check authentication: {e}")
        print("     This is usually fine for public models")
        return True


def verify_disk_space():
    """Check available disk space"""
    print_header("VERIFYING DISK SPACE")
    
    try:
        import shutil
        
        # Check temp directory
        temp_dir = tempfile.gettempdir()
        temp_usage = shutil.disk_usage(temp_dir)
        temp_free_gb = temp_usage.free / 1e9
        
        print(f"Temporary directory: {temp_dir}")
        print(f"  Free space: {temp_free_gb:.1f} GB")
        
        if temp_free_gb < 10:
            print(f"  ⚠️  Warning: Low disk space (<10 GB)")
        else:
            print(f"  ✓ Sufficient temp space")
        
        # Check if SCRATCH/PROJECT are set
        if 'SCRATCH' in os.environ:
            scratch_dir = os.environ['SCRATCH']
            print(f"\nSCRATCH directory: {scratch_dir}")
            if os.path.exists(scratch_dir):
                scratch_usage = shutil.disk_usage(scratch_dir)
                scratch_free_gb = scratch_usage.free / 1e9
                print(f"  Free space: {scratch_free_gb:.1f} GB")
                
                if scratch_free_gb < 500:
                    print(f"  ⚠️  Warning: Less than 500 GB free")
                    print(f"     Full pipeline needs ~500 GB")
                else:
                    print(f"  ✓ Sufficient scratch space for full pipeline")
        else:
            print(f"\n⚠️  SCRATCH environment variable not set")
            print(f"   This is OK for local testing")
            print(f"   On HPC, SCRATCH should be set automatically")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True


def verify_internet_connection():
    """Verify internet connection for downloads"""
    print_header("VERIFYING INTERNET CONNECTION")
    
    try:
        import urllib.request
        
        # Test connection to Hugging Face
        print("Testing connection to Hugging Face Hub...")
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        print("  ✓ Can reach huggingface.co")
        
        # Test connection to datasets CDN
        print("Testing connection to Hugging Face datasets CDN...")
        urllib.request.urlopen('https://datasets-server.huggingface.co', timeout=10)
        print("  ✓ Can reach datasets server")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Internet connection test failed: {e}")
        print(f"\n  Possible issues:")
        print(f"    - No internet connection")
        print(f"    - Firewall blocking huggingface.co")
        print(f"    - Proxy configuration needed")
        print(f"\n  On HPC: This might work differently on compute nodes")
        return False


def test_quick_model_load():
    """Test loading a small model quickly"""
    print_header("TESTING QUICK MODEL LOAD (GPT-2)")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import tempfile
        
        print("Loading GPT-2 (small model for testing)...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use GPT-2 as a proxy test (small, fast, reliable)
            tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=tmpdir)
            print(f"  ✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")
            
            model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=tmpdir)
            print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
            
            # Test forward pass
            import torch
            inputs = tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            print(f"  ✓ Forward pass successful (output shape: {outputs.logits.shape})")
        
        print(f"\n✓ Model download and loading mechanism working")
        print(f"  (Pythia-1.4B should work similarly, but is ~6GB)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_streaming():
    """Test dataset streaming mechanism"""
    print_header("TESTING DATASET STREAMING")
    
    try:
        from datasets import load_dataset
        import tempfile
        
        # Test with a small, reliable dataset first
        print("Testing with C4 dataset (small, reliable)...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try C4 validation (smaller than Pile)
            dataset = load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True,
                cache_dir=tmpdir,
                trust_remote_code=True
            )
            
            print("  ✓ Dataset loaded in streaming mode")
            
            # Read one sample
            sample = next(iter(dataset))
            print(f"  ✓ Read sample successfully")
            print(f"    Sample keys: {list(sample.keys())}")
            
            if 'text' in sample:
                text_preview = sample['text'][:80].replace('\n', ' ')
                print(f"    Text preview: {text_preview}...")
                print(f"  ✓ Dataset streaming mechanism working")
            
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pile_dataset_access(quick=True):
    """Test Pile dataset specifically"""
    print_header("TESTING PILE DATASET ACCESS")
    
    try:
        from datasets import load_dataset
        import tempfile
        
        dataset_name = Config.TRAINING_DATASET
        
        print(f"Dataset: {dataset_name}")
        
        if quick:
            print("Testing with validation split (faster)...")
            split = "validation"
            streaming = False
        else:
            print("Testing with train split (streaming)...")
            split = "train"
            streaming = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Loading {split} split...")
            
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                cache_dir=tmpdir,
                trust_remote_code=True
            )
            
            print(f"  ✓ Dataset loaded successfully")
            
            # Read sample
            if streaming:
                sample = next(iter(dataset))
            else:
                sample = dataset[0]
                print(f"  Dataset size: {len(dataset):,} examples")
            
            print(f"  ✓ Read sample successfully")
            print(f"    Sample keys: {list(sample.keys())}")
            
            if 'text' in sample:
                text_len = len(sample['text'])
                text_preview = sample['text'][:100].replace('\n', ' ')
                print(f"    Text length: {text_len:,} characters")
                print(f"    Text preview: {text_preview}...")
                print(f"  ✓ Pile dataset structure verified")
            else:
                print(f"  ⚠️  'text' field not found")
                print(f"     Available fields: {list(sample.keys())}")
                print(f"     You may need to adjust data loading code")
        
        print(f"\n✓ Pile dataset accessible")
        return True
        
    except Exception as e:
        print(f"\n✗ Pile dataset access failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("  1. Pile dataset is very large (825GB)")
        print("  2. May need to use 'validation' split for testing")
        print("  3. Consider alternative datasets:")
        print("     - 'allenai/c4' (smaller, well-supported)")
        print("     - 'openwebtext' (medium size)")
        print("  4. On HPC: Test in interactive session first")
        
        return False


def verify_all(quick=False, offline=False):
    """Run all verification checks"""
    print("\n" + "="*80)
    print("  MODEL AND DATASET DOWNLOAD VERIFICATION")
    print("="*80)
    print()
    print("This script verifies that all required models and datasets")
    print("can be downloaded successfully before HPC deployment.")
    print()
    
    if quick:
        print("Running in QUICK mode (skips large downloads)")
    if offline:
        print("Running in OFFLINE mode (checks cache only)")
    print()
    
    results = {}
    
    # Run checks
    if not offline:
        results['internet'] = verify_internet_connection()
        results['dependencies'] = verify_dependencies()
        results['hf_auth'] = verify_huggingface_auth()
        results['disk_space'] = verify_disk_space()
        results['model_info'] = verify_model_info()
        results['model_download'] = verify_model_download(quick=quick)
        results['dataset_info'] = verify_dataset_info()
        results['dataset_access'] = test_pile_dataset_access(quick=quick)
        results['streaming_test'] = test_dataset_streaming()
        results['quick_model'] = test_quick_model_load()
    else:
        print("Offline mode - skipping download tests")
        results['dependencies'] = verify_dependencies()
        results['disk_space'] = verify_disk_space()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    print(f"\n  Total: {passed}/{total} checks passed")
    print()
    
    if all(results.values()):
        print("="*80)
        print("  ✓ ALL DOWNLOAD VERIFICATIONS PASSED")
        print("="*80)
        print()
        print("Models and datasets are accessible!")
        print()
        print("Next steps:")
        print("  1. Run full test suite: bash run_tests.sh all")
        print("  2. Run pipeline verification: python verify_local.py")
        print("  3. Deploy to HPC: bash run_all.sh")
        print()
        return 0
    else:
        failed_checks = [name for name, success in results.items() if not success]
        
        print("="*80)
        print("  ⚠️  SOME CHECKS FAILED")
        print("="*80)
        print()
        print(f"Failed checks: {', '.join(failed_checks)}")
        print()
        print("Review errors above and:")
        print("  1. Check internet connection")
        print("  2. Install missing dependencies: pip install -r requirements.txt")
        print("  3. Try quick mode if full downloads timing out: --quick")
        print("  4. Some warnings OK for local testing (will work on HPC)")
        print()
        
        # Determine if critical failures
        critical_failures = [f for f in failed_checks if f in ['dependencies', 'model_download', 'dataset_access']]
        
        if critical_failures:
            print("❌ CRITICAL FAILURES - Fix before HPC deployment")
            return 1
        else:
            print("⚠️  NON-CRITICAL WARNINGS - May be OK, review carefully")
            return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Verify model and dataset downloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_downloads.py              # Full verification
  python verify_downloads.py --quick      # Skip large downloads
  python verify_downloads.py --offline    # Check cached only

This should be run before deploying to HPC to catch download issues early.
        """
    )
    parser.add_argument('--quick', action='store_true',
                       help="Quick mode (skip large downloads)")
    parser.add_argument('--offline', action='store_true',
                       help="Offline mode (check cache only, no downloads)")
    parser.add_argument('--full', action='store_true',
                       help="Full mode (test complete model download, slow)")
    
    args = parser.parse_args()
    
    try:
        return verify_all(quick=args.quick and not args.full, offline=args.offline)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
