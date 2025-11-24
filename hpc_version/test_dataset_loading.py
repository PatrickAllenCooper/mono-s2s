#!/usr/bin/env python3
"""
Test Dataset Loading - Diagnostic Script

This script tests loading all test datasets to identify any issues
before running the full pipeline.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from configs.experiment_config import ExperimentConfig

print("="*80)
print("DATASET LOADING DIAGNOSTIC")
print("="*80)
print(f"Configuration:")
print(f"  USE_FULL_TEST_SETS: {ExperimentConfig.USE_FULL_TEST_SETS}")
print(f"  QUICK_TEST_SIZE: {ExperimentConfig.QUICK_TEST_SIZE}")
print("")

# Test importing datasets library
print("Testing HuggingFace datasets library...")
try:
    from datasets import load_dataset
    print("  ✓ datasets library imported successfully")
except ImportError as e:
    print(f"  ❌ Failed to import datasets: {e}")
    sys.exit(1)

print("")
print("="*80)
print("TESTING INDIVIDUAL DATASETS")
print("="*80)

# Test datasets configuration
test_configs = [
    {
        'name': 'CNN/DailyMail',
        'dataset_name': 'cnn_dailymail',
        'config': '3.0.0',
        'split': 'test',
        'text_field': 'article',
        'summary_field': 'highlights'
    },
    {
        'name': 'XSUM',
        'dataset_name': 'EdinburghNLP/xsum',
        'config': None,
        'split': 'test',
        'text_field': 'document',
        'summary_field': 'summary'
    },
    {
        'name': 'SAMSum',
        'dataset_name': 'samsum',
        'config': None,
        'split': 'test',
        'text_field': 'dialogue',
        'summary_field': 'summary'
    }
]

results = []

for test_config in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {test_config['name']}")
    print(f"{'='*80}")
    print(f"  Dataset: {test_config['dataset_name']}")
    print(f"  Config: {test_config['config']}")
    print(f"  Split: {test_config['split']}")
    print(f"  Fields: {test_config['text_field']} → {test_config['summary_field']}")
    print("")
    
    try:
        # Load dataset
        print("  [1/4] Loading dataset from HuggingFace...")
        if test_config['config']:
            dataset = load_dataset(
                test_config['dataset_name'], 
                test_config['config'], 
                split=test_config['split'],
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                test_config['dataset_name'], 
                split=test_config['split'],
                trust_remote_code=True
            )
        print(f"        ✓ Loaded {len(dataset)} total samples")
        
        # Check fields exist
        print(f"  [2/4] Checking fields...")
        if len(dataset) > 0:
            example = dataset[0]
            print(f"        Available fields: {list(example.keys())}")
            
            if test_config['text_field'] in example:
                print(f"        ✓ Text field '{test_config['text_field']}' found")
            else:
                print(f"        ❌ Text field '{test_config['text_field']}' NOT FOUND")
                results.append({'name': test_config['name'], 'status': 'FAIL', 'reason': f"Text field missing"})
                continue
                
            if test_config['summary_field'] in example:
                print(f"        ✓ Summary field '{test_config['summary_field']}' found")
            else:
                print(f"        ❌ Summary field '{test_config['summary_field']}' NOT FOUND")
                results.append({'name': test_config['name'], 'status': 'FAIL', 'reason': f"Summary field missing"})
                continue
        
        # Extract samples
        print(f"  [3/4] Extracting samples...")
        max_samples = ExperimentConfig.QUICK_TEST_SIZE if not ExperimentConfig.USE_FULL_TEST_SETS else None
        
        texts = []
        summaries = []
        
        for i, example in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            text = example.get(test_config['text_field'], "")
            summary = example.get(test_config['summary_field'], "")
            
            if text and summary:
                texts.append(text.strip())
                summaries.append(summary.strip())
        
        print(f"        ✓ Extracted {len(texts)} valid samples")
        
        # Show example
        print(f"  [4/4] Showing first example...")
        if len(texts) > 0:
            print(f"        Text (first 100 chars): {texts[0][:100]}...")
            print(f"        Summary (first 100 chars): {summaries[0][:100]}...")
        
        # Success
        print("")
        print(f"  ✅ {test_config['name']}: SUCCESS ({len(texts)} samples)")
        results.append({
            'name': test_config['name'], 
            'status': 'SUCCESS', 
            'samples': len(texts),
            'avg_text_len': sum(len(t.split()) for t in texts) / len(texts) if texts else 0,
            'avg_summary_len': sum(len(s.split()) for s in summaries) / len(summaries) if summaries else 0
        })
        
    except Exception as e:
        print(f"\n  ❌ {test_config['name']}: FAILED")
        print(f"     Error: {type(e).__name__}: {str(e)}")
        results.append({
            'name': test_config['name'], 
            'status': 'FAIL', 
            'reason': f"{type(e).__name__}: {str(e)}"
        })

# Summary
print("")
print("="*80)
print("SUMMARY")
print("="*80)
print("")

for result in results:
    if result['status'] == 'SUCCESS':
        print(f"✅ {result['name']:20s} SUCCESS - {result['samples']} samples")
        print(f"   Avg text length: {result['avg_text_len']:.1f} words")
        print(f"   Avg summary length: {result['avg_summary_len']:.1f} words")
    else:
        print(f"❌ {result['name']:20s} FAILED")
        print(f"   Reason: {result['reason']}")
    print("")

# Overall status
success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
total_count = len(results)

print("="*80)
if success_count == total_count:
    print(f"✅ ALL {total_count} DATASETS LOADED SUCCESSFULLY")
    print("")
    print("You can proceed with Stage 1 data preparation.")
    sys.exit(0)
else:
    print(f"❌ {total_count - success_count}/{total_count} DATASETS FAILED")
    print("")
    print("Fix the issues above before running Stage 1.")
    sys.exit(1)

