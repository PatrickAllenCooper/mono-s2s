# Foundation LLM Monotonicity Experiments

This directory contains experiments applying monotonicity constraints to general-purpose foundation language models, extending the work in the main `mono-s2s` project from summarization-specific models to broader LLM capabilities.

## Overview

**Goal:** Test whether monotonicity constraints can improve robustness of general-purpose LLMs while maintaining task performance across diverse evaluation benchmarks.

**Key Differences from Main Project:**
- Uses general LLM instead of T5 summarization model
- Evaluates on diverse language tasks (not just summarization)
- Includes post-monotonicity recovery training on standard LLM data
- Tests UAT/HotFlip attacks in general text generation context

## Model Selection

**Selected Model:** Pythia-1.4B (EleutherAI)

**Rationale:**
- Size: 1.4B parameters fits comfortably on single A100 (40GB)
- Training: Fully reproducible with published training data
- Architecture: Standard decoder-only transformer (generalizes beyond T5)
- Performance: Strong baseline on diverse benchmarks
- FFN Parameters: ~560M parameters (40% of total) - substantial constraint surface

**Alternative Models Considered:**
- GPT-2 Large (774M): Older architecture, less capable
- OPT-1.3B: Similar size but less documentation
- FLAN-T5-Large (780M): Too similar to main project
- Phi-2 (2.7B): Might be tight on single A100 with full finetuning

## Experimental Pipeline - All Stages Complete ✅

### Stage 0: Setup ✅
- Downloads Pythia-1.4B checkpoint (~6GB)
- Verifies GPU/environment setup
- Creates all necessary directories
- **Status**: Fully implemented, tested, ready

### Stage 1: Apply Monotonicity Constraints ✅
- Converts FFN layers to softplus-parameterized weights (W = softplus(V))
- Initializes using inverse softplus of pretrained weights
- Verifies all weights are non-negative
- Saves monotonic-initialized model
- **Status**: Fully implemented, tested, ready

### Stage 2: Baseline Training ✅
- Finetunes standard Pythia-1.4B on Pile data
- Uses validation split (quick) or streaming train split (full)
- 1 epoch recovery training with AdamW, LR 1e-5
- Saves checkpoints every epoch
- **Status**: Fully implemented with Pile loading, ready

### Stage 3: Monotonic Recovery Training ✅
- Finetunes monotonic-initialized model on Pile data
- Same data as baseline for fair comparison
- Extended warmup (15% vs 10%) for softplus stability
- Verifies weights stay non-negative throughout training
- **Status**: Fully implemented with Pile loading, ready

### Stage 4: Evaluation ✅
- Computes perplexity on Pile test set (primary metric)
- Optional: LAMBADA, HellaSwag, Winogrande benchmarks
- Compares baseline vs monotonic performance
- **Status**: Fully implemented with Pile loading, ready

### Stage 5: UAT Attacks ✅
- Learns universal triggers via coordinate ascent
- 5 restarts, 100 iterations for robust optimization
- Tests on held-out Pile test data
- Measures perplexity increase (NLL degradation)
- **Status**: Fully implemented, adapted from T5 version, ready

### Stage 6: HotFlip Attacks ✅
- Gradient-based token flipping attacks
- Uses embedding gradients to find vulnerable positions
- Up to 10 flips per example (configurable)
- Measures attack success rate at 15% threshold
- **Status**: Fully implemented, adapted from T5 version, ready

### Stage 7: Aggregate Results ✅
- Combines all results into final_results.json
- Generates human-readable experiment_summary.txt
- Creates formatted output for paper Table 7
- Computes key findings and statistics
- **Status**: Fully implemented, tested, ready

## Directory Structure

```
foundation_llm_experiments/
├── README.md                    # This file
├── configs/
│   └── experiment_config.py     # Centralized configuration
├── scripts/
│   ├── stage_0_setup.py
│   ├── stage_1_apply_monotonicity.py
│   ├── stage_2_train_baseline.py
│   ├── stage_3_train_monotonic.py
│   ├── stage_4_evaluate.py
│   ├── stage_5_uat_attacks.py
│   ├── stage_6_hotflip_attacks.py
│   └── stage_7_aggregate.py
├── jobs/
│   ├── job_0_setup.sh
│   ├── job_1_monotonicity.sh
│   ├── job_2_baseline.sh
│   ├── job_3_monotonic.sh
│   ├── job_4_evaluate.sh
│   ├── job_5_uat.sh
│   ├── job_6_hotflip.sh
│   └── job_7_aggregate.sh
├── utils/
│   └── common_utils.py          # Shared utilities (from main project)
└── run_all.sh                   # Master submission script
```

## Key Differences in Methodology

### 1. Recovery Training
Unlike T5-small which was already fine-tuned for summarization, Pythia-1.4B is a general foundation model. We include a "recovery training" phase:
- **Purpose:** Restore perplexity lost from monotonicity initialization
- **Data:** Standard Pile training data (same distribution as pretraining)
- **Duration:** 1 epoch (~25B tokens)
- **Goal:** Close perplexity gap while maintaining monotonicity

### 2. Evaluation Benchmarks
Instead of ROUGE on summarization:
- **Perplexity:** Pile test set (primary metric)
- **LAMBADA:** Next-word prediction accuracy
- **HellaSwag:** Commonsense reasoning (sentence completion)
- **Winogrande:** Coreference resolution
- **TruthfulQA:** Factual accuracy

### 3. Attack Context
Instead of attacking summaries:
- **UAT:** Triggers that increase perplexity on arbitrary text
- **HotFlip:** Perturb prompts to degrade generation quality
- **Metrics:** Perplexity increase, answer correctness degradation

## Resource Requirements

### Compute
- **GPU:** 1x A100 (40GB) per job
- **Training Time:** 
  - Baseline: ~20 hours (1 epoch on Pile)
  - Monotonic: ~28 hours (extended warmup)
  - Total per seed: ~60 hours
- **Storage:** ~500GB (model checkpoints, evaluation data)

### Data
- **Pile:** 825GB deduped (~300B tokens)
- **Evaluation Sets:** ~10GB total
- **Checkpoints:** ~50GB per seed

## Usage

### Quick Start
```bash
# Submit all stages with dependencies
cd foundation_llm_experiments
bash run_all.sh
```

### Individual Stages
```bash
# Run specific stage
sbatch jobs/job_2_baseline.sh
```

### Monitor Progress
```bash
# Check job status
squeue -u $USER | grep foundation

# Check results
cat $SCRATCH/foundation_llm_work/experiment_summary.txt
```

## Expected Results

Based on extrapolation from T5-small findings:

**Perplexity (Pile Test Set):**
- Baseline: ~10.2
- Monotonic (pre-recovery): ~15.8 (+54% degradation)
- Monotonic (post-recovery): ~10.9 (+6.8% degradation)

**HotFlip Robustness:**
- Baseline attack success: ~55%
- Monotonic attack success: ~18% (67% reduction)

**UAT Robustness:**
- Minimal effect across all models (consistent with T5 findings)

## Limitations

1. **Single Model Family:** Only tests Pythia architecture
2. **Compute Budget:** 1 epoch recovery may be insufficient
3. **Evaluation Scope:** Limited to standard benchmarks
4. **Attack Sophistication:** UAT/HotFlip may not capture all threats

## Future Directions

1. Scale to Pythia-2.8B, 6.9B
2. Test on instruction-tuned variants (e.g., Dolly, OpenAssistant)
3. Combine with other robustness techniques (adversarial training)
4. Develop certified defenses leveraging monotonicity

## Citation

If you use this pipeline, please cite:

```bibtex
@article{monotone-llms-2025,
  title={Monotonicity as an Architectural Bias for Robust Language Models},
  author={[Authors]},
  journal={ICML},
  year={2025}
}
```

## Contact

For questions or issues, see main project README or contact [authors].
