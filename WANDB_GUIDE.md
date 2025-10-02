# Weights & Biases Integration Guide

## 📊 Overview

The MONO-S2S notebook now includes **comprehensive Weights & Biases (wandb) integration** for experiment tracking, visualization, and analysis. Every aspect of training and adversarial experiments is automatically logged.

---

## 🚀 Quick Start

### 1. Installation (Automatic)

Wandb is automatically installed by the setup scripts:
```bash
./setup_conda_env.sh  # Linux/Mac
# or
setup_conda_env.bat   # Windows
```

### 2. Authentication

**First Time Setup:**
```python
# Option 1: Interactive (recommended)
import wandb
wandb.login()

# Option 2: API Key
import os
os.environ['WANDB_API_KEY'] = 'your-api-key-here'

# Option 3: Anonymous (Colab)
# Automatically handled - runs anonymously in Colab
```

Get your API key from: https://wandb.ai/authorize

### 3. Run the Notebook

Just run the notebook normally! All logging happens automatically.

---

## 📈 What Gets Logged

### **Training Runs (Both Models)**

#### Batch-Level Metrics (Every Batch)
- `train/batch_loss` - Loss for current batch
- `train/learning_rate` - Current learning rate
- `train/teacher_forcing_ratio` - Teacher forcing probability
- `train/used_teacher_forcing` - Whether TF was used (0/1)
- `train/used_noise` - Whether input noise was applied (0/1)
- `train/batch_time` - Time to process batch (seconds)
- `train/grad_norm` - Gradient norm BEFORE clipping ⭐
- `train/grad_norm_clipped` - Gradient norm after clipping
- `train/epoch` - Current epoch
- `train/batch` - Current batch index

#### Epoch-Level Statistics
- `train/epoch_loss` - Average training loss
- `train/epoch_perplexity` - Training perplexity
- `train/epoch_time` - Total epoch time
- `train/batches_per_second` - Throughput
- `train/avg_grad_norm` - Average gradient norm
- `train/max_grad_norm` - Maximum gradient norm
- `train/min_batch_loss` - Minimum batch loss
- `train/max_batch_loss` - Maximum batch loss
- `train/std_batch_loss` - Batch loss standard deviation

#### Validation Metrics
- `val/loss` - Validation loss
- `val/perplexity` - Validation perplexity
- `val/eval_time` - Evaluation time
- `val/min_loss`, `val/max_loss`, `val/std_loss` - Distribution stats
- `val/samples` - Sample prediction table (every 2 epochs)

#### Model & Experiment Info
- `epoch` - Current epoch
- `best_val_loss` - Best validation loss so far
- `patience_counter` - Early stopping patience
- `is_best_epoch` - Whether this is the best epoch
- `training_samples` - Sample generations table

---

### **Adversarial Attack Experiments**

#### HotFlip Attack (Gradient-Based) ⭐⭐⭐

**Per-Iteration Logging (every 5 iterations):**
- `hotflip/{model}/iteration_loss` - Attack loss at iteration
- `hotflip/{model}/iteration_grad_norm` - Gradient norm ⭐
- `hotflip/{model}/iteration_tokens_changed` - Tokens modified
- `hotflip/{model}/iteration` - Iteration number

**Per-Sample Comparative Logging:**
- `hotflip/sample_idx` - Sample index
- `hotflip/nonmono_degradation` - Non-monotonic degradation
- `hotflip/mono_degradation` - Monotonic degradation
- `hotflip/robustness_delta` - Difference in degradation
- `hotflip/nonmono_avg_grad_norm` - Non-mono gradient norm ⭐
- `hotflip/mono_avg_grad_norm` - Mono gradient norm ⭐
- `hotflip/grad_norm_ratio` - Gradient norm ratio (NM/M) ⭐
- `hotflip/nonmono_loss_improvement` - Loss change
- `hotflip/mono_loss_improvement` - Loss change

**Final Summary:**
- `hotflip/summary_*` - Average metrics across all samples
- `hotflip/detailed_comparison` - Full comparison table
- Gradient statistics per position
- Token evolution tracking

#### Universal Trigger, NES, Injection, OOD Attacks

Similar comprehensive logging for each attack type with:
- Attack-specific metrics
- Comparative analysis
- Success rates
- Statistical summaries

---

### **Final Comprehensive Analysis**

#### Overall Metrics (in run summary)
- `overall_robustness_improvement_pct` - Overall improvement
- `overall_nonmono_degradation` - Average non-mono degradation
- `overall_mono_degradation` - Average mono degradation
- `overall_p_value` - Statistical significance
- `overall_statistically_significant` - Boolean flag

#### Per-Attack Final Metrics
- `final/{attack}_nonmono_mean` - Non-mono mean
- `final/{attack}_mono_mean` - Mono mean
- `final/{attack}_improvement_pct` - Improvement percentage
- `final/{attack}_p_value` - P-value
- `final/{attack}_significant` - Significance flag

#### Summary Tables
- `final/comprehensive_summary` - All attacks comparison table
- `final/attack_comparison_with_gradients` - With gradient norms
- `key_findings` - Textual summary of results

---

## 🎯 Experiment Organization

### Project Structure

All experiments are logged to:
- **Project:** `mono-s2s-adversarial-robustness`
- **Entity:** Your wandb username (or default)

### Run Types

The notebook creates **3 separate wandb runs**:

1. **Non-Monotonic Training**
   - Name: `{env}_nonmono_training_{timestamp}`
   - Tags: `["training", "non-monotonic", "seq2seq", "summarization"]`
   
2. **Monotonic Training**
   - Name: `{env}_mono_training_{timestamp}`
   - Tags: `["training", "monotonic", "seq2seq", "summarization", "robustness"]`
   
3. **Adversarial Experiments**
   - Name: `{env}_adversarial_experiments_{timestamp}`
   - Tags: `["adversarial", "robustness", "gradient-analysis", "attacks"]`

Where `{env}` is either `colab` or `local`.

---

## 📊 Key Visualizations

### What You Can Plot in Wandb

#### Training Dynamics
- **Loss curves**: `train/epoch_loss` vs `val/loss`
- **Gradient norms**: `train/avg_grad_norm` over time
- **Learning rate schedule**: `train/learning_rate`
- **Perplexity**: `train/epoch_perplexity` and `val/perplexity`

#### Adversarial Analysis
- **Attack success**: Compare degradation across attack types
- **Gradient behavior**: Compare `hotflip/*_grad_norm` between models ⭐
- **Robustness delta**: `hotflip/robustness_delta` over samples
- **Convergence**: Attack loss over iterations

#### Model Comparison
- **Side-by-side training**: Compare non-mono vs mono runs
- **Attack resistance**: Bar charts of degradation per attack
- **Statistical significance**: P-values and effect sizes

---

## 🔍 Advanced Analysis

### Gradient Analysis (HotFlip) ⭐

The most detailed logging is for gradient-based attacks:

```python
# What gets tracked per attack iteration:
{
    'avg_grad_norm': 2.4531,           # Average L2 norm
    'max_grad_norm': 3.2145,           # Maximum norm
    'min_grad_norm': 1.8234,           # Minimum norm  
    'std_grad_norm': 0.4321,           # Standard deviation
    'avg_tokens_changed': 2.3,         # Avg tokens modified per iter
    'loss_improvement': 1.234,         # Total loss change
    'grad_norms_per_iteration': [...], # Full history
    'loss_per_iteration': [...],       # Loss trajectory
    'avg_grad_mag_per_position': [...] # Per-position gradients
}
```

### Custom Queries

You can query wandb runs programmatically:

```python
import wandb

api = wandb.Api()
runs = api.runs("your-username/mono-s2s-adversarial-robustness")

# Get all training runs
training_runs = [r for r in runs if "training" in r.tags]

# Compare gradient norms
for run in training_runs:
    if "monotonic" in run.tags:
        grad_norms = run.history(keys=["train/avg_grad_norm"])
        print(f"{run.name}: avg grad = {grad_norms['train/avg_grad_norm'].mean()}")
```

---

## 💡 Best Practices

### 1. Run Names

Runs are automatically named with:
- Environment (colab/local)
- Model type (nonmono/mono)
- Timestamp

You can add custom notes:
```python
wandb.run.notes = "Testing new hyperparameters"
```

### 2. Comparing Runs

In Wandb UI:
1. Select multiple runs (checkbox)
2. Click "Compare"
3. Choose metrics to plot
4. Export charts/tables

### 3. Hyperparameter Sweeps

To run hyperparameter sweeps:

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform', 'min': -5, 'max': -3},
        'dropout': {'values': [0.1, 0.2, 0.3]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="mono-s2s-adversarial-robustness")
wandb.agent(sweep_id, function=train_function, count=10)
```

### 4. Artifacts

Save models as artifacts:

```python
# Already done automatically via checkpointing, but you can also:
artifact = wandb.Artifact('model-checkpoint', type='model')
artifact.add_file('best_model_mono.pt')
wandb.log_artifact(artifact)
```

---

## 🐛 Troubleshooting

### "Not logged in to wandb"

**Solution:**
```bash
wandb login
# Or set environment variable
export WANDB_API_KEY=your_key
```

### "Wandb run is None"

This means wandb initialization failed. Check:
1. Are you logged in?
2. Is wandb installed? (`pip install wandb`)
3. Network connectivity?

The notebook will continue without wandb if initialization fails.

### Too Much Logging

If you want to reduce logging frequency:

```python
# Modify in the training function
if log_wandb and wandb.run and (batch_idx % 100 == 0):  # Log every 100 batches instead of every batch
    wandb.log({...})
```

### Anonymous Mode (Colab)

If in Colab and don't want to log in:
```python
wandb.login(anonymous="allow")
```

Results will be saved anonymously and link provided.

---

## 📚 Key Metrics Reference

### Most Important Metrics to Watch

1. **Training Health**
   - `train/avg_grad_norm` - Should be stable, not exploding
   - `train/epoch_loss` - Should decrease
   - `val/loss` - Should decrease without overfitting

2. **Adversarial Robustness** ⭐⭐⭐
   - `hotflip/grad_norm_ratio` - Lower = more robust
   - `hotflip/robustness_delta` - Positive = monotonic is better
   - `final/overall_robustness_improvement_pct` - Overall result

3. **Model Comparison**
   - Compare `train/epoch_loss` between mono/non-mono runs
   - Compare gradient norms during attacks
   - Compare final degradation metrics

---

## 🎓 Example Analyses

### Analysis 1: Gradient Behavior During Attacks

**Question:** Do monotonic models have smaller gradients during attacks?

**Metrics to Compare:**
- `hotflip/nonmono_avg_grad_norm` vs `hotflip/mono_avg_grad_norm`
- `hotflip/grad_norm_ratio`

**Expected:** Monotonic models should show smaller gradient norms, indicating more stable optimization landscape.

### Analysis 2: Training Stability

**Question:** Are gradient norms more stable during training?

**Metrics to Plot:**
- `train/avg_grad_norm` over time
- `train/std_batch_loss`

**Expected:** Monotonic models should show more consistent gradient norms.

### Analysis 3: Attack Resistance

**Question:** Which attack type benefits most from monotonicity?

**Metrics to Compare:**
- `final/{attack}_improvement_pct` for each attack
- Gradient-based vs non-gradient-based attacks

**Expected:** Gradient-based attacks (HotFlip) should show highest improvement.

---

## 🔗 Resources

- **Wandb Documentation:** https://docs.wandb.ai/
- **Wandb Examples:** https://github.com/wandb/examples
- **Dashboard:** https://wandb.ai/home
- **API Reference:** https://docs.wandb.ai/ref/python

---

## ✅ Quick Checklist

Before running experiments:
- [ ] Wandb installed (`pip install wandb`)
- [ ] Logged in (`wandb login`)
- [ ] Project name is appropriate
- [ ] Run tags make sense

After experiments:
- [ ] Check wandb.run.url for results
- [ ] Verify key metrics logged correctly
- [ ] Export important charts/tables
- [ ] Add notes to runs for future reference

---

**Happy Experimenting! 🚀**

All your experiments are now tracked, reproducible, and shareable with comprehensive gradient analysis during adversarial attacks.

