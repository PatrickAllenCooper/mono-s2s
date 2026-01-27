# Implementation Guide for Remaining Stages

This guide explains how to complete the implementation of stages 4-7.

## Current Status

**Completed**:
- ✅ Stage 0: Setup (fully implemented)
- ✅ Stage 1: Apply monotonicity (fully implemented)
- ✅ Stage 2: Baseline training (skeleton implemented)
- ✅ Stage 3: Monotonic training (skeleton implemented)
- ✅ Stage 4: Evaluation (skeleton implemented)

**TODO**:
- ⏳ Stage 5: UAT attacks (needs implementation)
- ⏳ Stage 6: HotFlip attacks (needs implementation)
- ⏳ Stage 7: Aggregation (needs implementation)

## Implementation Strategy

### Option A: Adapt from Main Project (Recommended)

The easiest approach is to adapt the working scripts from `../hpc_version/scripts/`:

1. **Copy base structure**:
   ```bash
   cp ../hpc_version/scripts/stage_5_uat_attacks.py scripts/stage_5_uat_attacks.py
   ```

2. **Modify for decoder-only architecture**:
   - Change imports from T5 to AutoModelForCausalLM
   - Replace ROUGE metrics with perplexity
   - Adjust attack targets (sequence generation vs summarization)

3. **Key Changes Required**:

   | Aspect | T5 (Main Project) | Pythia (This Project) |
   |---|---|---|
   | Model class | `T5ForConditionalGeneration` | `AutoModelForCausalLM` |
   | Loss function | `model(input_ids, labels=labels).loss` | Same (but labels = input_ids shifted) |
   | Evaluation metric | ROUGE scores | Perplexity |
   | Attack target | Summary quality | Generation quality/perplexity |
   | Trigger position | Prepend to input | Prepend to prompt |

### Option B: Use LM Evaluation Harness

Leverage EleutherAI's evaluation library:

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Wrap your model
hf_model = HFLM(pretrained=model, tokenizer=tokenizer)

# Run evaluation
results = evaluator.simple_evaluate(
    model=hf_model,
    tasks=["lambada", "hellaswag", "winogrande"],
    num_fewshot=0,
    batch_size=8
)
```

**Pros**: Battle-tested, standard benchmarks
**Cons**: Requires integration with your checkpoint loading

### Option C: Minimal Implementation

For quick validation, create minimal versions:

**Stage 4: Evaluation**
```python
# Compute perplexity only, skip other benchmarks
from utils.common_utils import compute_perplexity
baseline_ppl = compute_perplexity(baseline_model, test_loader, device)
monotonic_ppl = compute_perplexity(monotonic_model, test_loader, device)
```

**Stage 5: UAT**
```python
# Copy ../hpc_version/scripts/stage_5_uat_attacks.py
# Change line ~200: Replace ROUGE computation with perplexity
```

**Stage 6: HotFlip**
```python
# Copy ../hpc_version/scripts/stage_6_hotflip_attacks.py
# Change line ~150: Replace ROUGE degradation with perplexity increase
```

**Stage 7: Aggregate**
```python
# Simple aggregation
import json
results = {
    'evaluation': json.load(open('evaluation_results.json')),
    'uat': json.load(open('uat_results.json')),
    'hotflip': json.load(open('hotflip_results.json'))
}
json.dump(results, open('final_results.json', 'w'))
```

## Detailed Implementation Instructions

### Stage 5: UAT Attacks

**Base File**: `../hpc_version/scripts/stage_5_uat_attacks.py`

**Key Modifications**:

1. **Change line 50-55** (model loading):
   ```python
   # OLD (T5):
   from transformers import T5ForConditionalGeneration
   model = T5ForConditionalGeneration.from_pretrained(...)
   
   # NEW (Pythia):
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(...)
   ```

2. **Change line 150-170** (loss computation):
   ```python
   # OLD (T5 - encoder-decoder):
   outputs = model(input_ids=input_ids, labels=labels)
   
   # NEW (Pythia - decoder-only):
   # Labels are input_ids shifted by 1
   labels = input_ids.clone()
   labels[attention_mask == 0] = -100
   outputs = model(input_ids=input_ids, labels=labels)
   ```

3. **Change line 300-320** (evaluation metric):
   ```python
   # OLD (T5 - ROUGE):
   from rouge_score import rouge_scorer
   rouge_deltas = compute_rouge_delta(...)
   
   # NEW (Pythia - Perplexity):
   baseline_ppl = compute_perplexity(model, clean_loader, device)
   attack_ppl = compute_perplexity(model, attacked_loader, device)
   ppl_increase = (attack_ppl - baseline_ppl) / baseline_ppl
   ```

### Stage 6: HotFlip Attacks

**Base File**: `../hpc_version/scripts/stage_6_hotflip_attacks.py`

**Key Modifications**:

1. **Change line 100-120** (gradient computation):
   ```python
   # Similar to main project, but for decoder-only:
   # Compute gradients w.r.t. input embeddings
   embeddings = model.get_input_embeddings()(input_ids)
   embeddings.retain_grad()
   
   outputs = model(inputs_embeds=embeddings, labels=labels)
   loss = outputs.loss
   loss.backward()
   
   token_gradients = embeddings.grad
   ```

2. **Change line 200-210** (success metric):
   ```python
   # OLD (T5 - ROUGE degradation > 10%):
   success = rouge_delta > 0.10
   
   # NEW (Pythia - Perplexity increase > 15%):
   success = (attack_ppl - clean_ppl) / clean_ppl > 0.15
   ```

### Stage 7: Aggregation

**Create from scratch** - relatively simple:

```python
#!/usr/bin/env python3
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import create_completion_flag, save_json, StageLogger

def main():
    logger = StageLogger("stage_7_aggregate")
    
    # Load all results
    results_dir = Config.RESULTS_DIR
    
    baseline_history = json.load(open(os.path.join(results_dir, 'baseline_training_history.json')))
    monotonic_history = json.load(open(os.path.join(results_dir, 'monotonic_training_history.json')))
    evaluation = json.load(open(os.path.join(results_dir, 'evaluation_results.json')))
    uat = json.load(open(os.path.join(results_dir, 'uat_results.json')))
    hotflip = json.load(open(os.path.join(results_dir, 'hotflip_results.json')))
    
    # Aggregate
    final_results = {
        'experiment_info': {
            'seed': Config.CURRENT_SEED,
            'model_name': Config.MODEL_NAME,
        },
        'training_summary': {
            'baseline': baseline_history,
            'monotonic': monotonic_history
        },
        'evaluation_summary': evaluation,
        'attack_summary': {
            'uat': uat,
            'hotflip': hotflip
        }
    }
    
    # Save
    save_json(final_results, os.path.join(Config.FINAL_RESULTS_DIR, 'final_results.json'))
    
    logger.complete(success=True)
    return 0

if __name__ == "__main__":
    exit(main())
```

## Data Loading Implementation

### Loading Pile Dataset

```python
from datasets import load_dataset

# Full dataset (streaming for large data)
pile_train = load_dataset(
    "EleutherAI/pile",
    split="train",
    streaming=True,
    cache_dir=Config.DATA_CACHE_DIR
)

# Take first N samples
train_texts = []
for i, example in enumerate(pile_train):
    if i >= Config.TRAINING_SAMPLES:
        break
    train_texts.append(example['text'])

# For validation
pile_val = load_dataset(
    "EleutherAI/pile",
    split="validation",
    streaming=False,
    cache_dir=Config.DATA_CACHE_DIR
)
val_texts = [ex['text'] for ex in pile_val]
```

### Loading Evaluation Benchmarks

```python
# LAMBADA
lambada = load_dataset("lambada", split="test")

# HellaSwag
hellaswag = load_dataset("hellaswag", split="validation")

# Winogrande
winogrande = load_dataset("winogrande", "winogrande_xl", split="validation")

# TruthfulQA
truthfulqa = load_dataset("truthful_qa", "multiple_choice", split="validation")
```

## Testing the Pipeline

### Quick Test (Without Full Training)

1. **Edit config**:
   ```python
   # configs/experiment_config.py
   USE_FULL_EVAL_SETS = False
   TRAINING_SAMPLES = 1000  # Tiny training set
   QUICK_PILE_TEST_SIZE = 100
   RECOVERY_EPOCHS = 1
   ```

2. **Run setup only**:
   ```bash
   sbatch jobs/job_0_setup.sh
   sbatch jobs/job_1_monotonicity.sh
   ```

3. **Verify outputs**:
   ```bash
   ls $SCRATCH/foundation_llm_work/checkpoints/monotonic_initialized.pt
   cat $SCRATCH/foundation_llm_results/monotonicity_application_log.json
   ```

### Integration Testing

After implementing stages 2-7:

```bash
# Test stage 2
sbatch jobs/job_2_baseline.sh

# After it completes (~1 hour with dummy data), check:
ls $SCRATCH/foundation_llm_work/checkpoints/baseline_checkpoints/best_model.pt
```

## Common Implementation Issues

### Issue 1: Pile Dataset Too Large

**Problem**: Pile is 825GB, takes hours to download

**Solution**: Use subset or validation split
```python
# Use validation split (smaller)
pile = load_dataset("EleutherAI/pile", split="validation")

# Or limit streaming
pile_stream = load_dataset("EleutherAI/pile", split="train", streaming=True)
train_texts = [ex['text'] for i, ex in enumerate(pile_stream) if i < 10000]
```

### Issue 2: OOM During Training

**Problem**: Pythia-1.4B + batch_size=8 exceeds 40GB

**Solution**: Use gradient checkpointing
```python
model.gradient_checkpointing_enable()
# And/or reduce batch size:
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8  # Keep effective batch size
```

### Issue 3: Tokenizer Padding Issues

**Problem**: Pythia tokenizer doesn't have pad_token

**Solution**: Set it manually
```python
tokenizer = AutoTokenizer.from_pretrained(...)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

## Next Steps

1. **Implement Stage 5** (UAT):
   - Copy from `../hpc_version/scripts/stage_5_uat_attacks.py`
   - Apply modifications listed above
   - Test on dummy data

2. **Implement Stage 6** (HotFlip):
   - Copy from `../hpc_version/scripts/stage_6_hotflip_attacks.py`
   - Apply modifications listed above
   - Test on dummy data

3. **Implement Stage 7** (Aggregate):
   - Use template above
   - Test with dummy JSON files

4. **Create Job Scripts**:
   - Copy `jobs/job_0_setup.sh` as template
   - Adjust time limits for each stage
   - Update script paths

5. **End-to-End Test**:
   - Run full pipeline with dummy data
   - Verify all stages complete
   - Check output files

6. **Full Run**:
   - Set `USE_FULL_EVAL_SETS=True`
   - Set `TRAINING_SAMPLES=None` (use full Pile)
   - Submit with `bash run_all.sh`

## Questions?

- Check main project: `../hpc_version/scripts/` for working examples
- Review config: `configs/experiment_config.py` for all parameters
- Test utilities: `python utils/common_utils.py`

**Estimated Implementation Time**: 4-6 hours to adapt all scripts
**Estimated Testing Time**: 2-3 hours for integration testing
**Estimated Full Run Time**: 60-70 hours per seed on A100
