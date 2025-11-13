# Monotonic Feed-Forward Networks for Adversarial Robustness in Seq2Seq Models

**A methodologically rigorous evaluation of non-negative weight constraints in T5 summarization models**

---

## What We're Testing

This experiment rigorously evaluates whether **local monotonic constraints in feed-forward network (FFN) sublayers** improve adversarial robustness in sequence-to-sequence models.

### Hypothesis
Constraining FFN weights to be non-negative (W ≥ 0) via softplus parametrization creates locally monotonic transformations that are more resistant to adversarial perturbations, analogous to improvements observed in CNNs.

### Three-Way Fair Comparison

| Model | Description | Purpose |
|-------|-------------|---------|
| **Standard T5** | Pre-trained `t5-small`, not fine-tuned | Reference baseline |
| **Baseline T5** | Fine-tuned with **identical** hyperparameters, **no constraints** | Fair comparison control |
| **Monotonic T5** | Fine-tuned with **identical** hyperparameters, **W≥0 FFN constraints** | Treatment condition |

### What Makes This Fair

✅ **Identical Training:** Same data, optimizer, learning rate, batch size, epochs, warmup  
✅ **Identical Evaluation:** Fixed decoding parameters, same test sets, same metrics  
✅ **Statistical Rigor:** Bootstrap 95% confidence intervals on full test sets  
✅ **Proper Attack Evaluation:** Held-out optimization/test splits, transfer matrix  
✅ **Complete Reproducibility:** All seeds controlled, deterministic algorithms  

### Critical Scope Limitation

⚠️ **This is NOT a globally monotonic model.** Constraints apply to **FFN sublayers only**.  
The full encoder/decoder remains non-monotonic due to LayerNorm, residual connections, and attention mechanisms.

---

## Quick Start: CURC Alpine (aa100)

### Prerequisites

```bash
# SSH to Alpine
ssh your_username@login.rc.colorado.edu

# Check system Python (Alpine has python3 by default)
which python3
python3 --version  # Should show Python 3.x

# No modules needed - Alpine uses system Python
# CUDA drivers are available on compute nodes automatically
```

### 1. Setup (5 minutes)

```bash
# Clone/transfer repository to /projects (larger disk space)
cd /projects/$USER
git clone https://github.com/PatrickAllenCooper/mono-s2s.git
cd mono-s2s/hpc_version

# Install PyTorch with CUDA support (installs to ~/.local)
python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
python3 -m pip install --user transformers datasets rouge-score scipy pandas tqdm

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
# Note: CUDA will show False on login nodes, True on GPU compute nodes

# Make scripts executable
chmod +x run_all.sh jobs/*.sh scripts/*.py

# Validate setup
./validate_setup.sh
```

### 2. Configure for Alpine (2 minutes)

Edit `configs/experiment_config.py`:

```python
# Lines 23-24: Verify paths (environment variables auto-populate on Alpine)
# SCRATCH_DIR = os.environ.get("SCRATCH", "/scratch/alpine/your_username")
# PROJECT_DIR = os.environ.get("PROJECT", "/pl/active/your_project")
# These should work automatically - $SCRATCH and $PROJECT are set by SLURM

# Line 147: Verify GPU partition
SLURM_PARTITION = "aa100"  # A100 GPUs on Alpine
```

**Storage Layout:**
- **Code:** `/projects/$USER/mono-s2s/` (where you cloned - persistent, backed up)
- **Scratch:** `$SCRATCH/mono_s2s_work/` (temporary files, fast I/O, auto-cleaned after 90 days)
- **Results:** `$PROJECT/mono_s2s_final_results/` (final results, persistent, backed up)

### 3. Deploy (1 command)

```bash
# Submit full pipeline
./run_all.sh

# OR specify a custom seed
./run_all.sh 42
```

### 4. Monitor

```bash
# Check job queue
squeue -u $USER

# View logs
tail -f logs/job_*.out

# Check progress
ls $SCRATCH/mono_s2s_work/*.flag
```

**Expected completion:** 18-22 hours (fully automated)

---

## Resource Requirements (Alpine aa100)

| Stage | GPU | Memory | Time | Notes |
|-------|-----|--------|------|-------|
| Setup | ❌ | 16 GB | 30 min | Model download |
| Data Prep | ❌ | 32 GB | 2 hr | Dataset caching |
| Train Baseline | ✅ A100 | 64 GB | 10 hr | Parallel with Monotonic |
| Train Monotonic | ✅ A100 | 64 GB | 10 hr | Parallel with Baseline |
| Evaluation | ✅ A100 | 32 GB | 3 hr | All models × all tests |
| UAT Attacks | ✅ A100 | 16 GB | 2 hr | Parallel with HotFlip |
| HotFlip Attacks | ✅ A100 | 16 GB | 2 hr | Parallel with UAT |
| Aggregation | ❌ | 8 GB | 15 min | Final analysis |

**Total Wall-Clock:** ~18 hours (with parallelization)  
**Total GPU Hours:** ~27 hours on A100  
**Total SUs:** ~350-450 on Alpine

---

## Datasets & Evaluation

### Training Data (7 datasets, ~500K examples)
DialogSum • SAMSum • XSUM • AMI • HighlightSum • arXiv • MEETING_SUMMARY

### Test Data (3 datasets, held-out splits)
**CNN/DailyMail v3.0.0** • **XSUM** • **SAMSum**

### Metrics
- **ROUGE-1/2/L** with bootstrap 95% CIs (1000 resamples)
- **Adversarial robustness:** UAT (universal triggers), HotFlip (gradient-based)
- **Transfer matrix:** Cross-model attack transferability
- **Length statistics:** Brevity penalty, token ratios

---

## Key Results Files

After completion, find results in `$PROJECT/mono_s2s_final_results/`:

```
evaluation_results.json       ← ROUGE scores with 95% CIs
uat_results.json             ← UAT attacks + transfer matrix
hotflip_results.json         ← Gradient attack results
final_results.json           ← Comprehensive aggregation
experiment_summary.txt       ← Human-readable tables
```

---

## Architecture & Implementation

### Pipeline Structure

```
Stage 0 → Stage 1 → ┬ Stage 2 (Baseline)     ┐
                    └ Stage 3 (Monotonic)    ┘ → Stage 4 → ┬ Stage 5 (UAT)      ┐
                                                             └ Stage 6 (HotFlip)  ┘ → Stage 7
```

### Monotonic Constraint Implementation

```python
# Softplus parametrization: W = softplus(V) ≥ 0
class NonNegativeParametrization(nn.Module):
    def forward(self, V):
        return F.softplus(V)

# Applied to FFN sublayers: wi, wi_0, wi_1, wo
for module in model.modules():
    if isinstance(module, T5DenseReluDense):
        P.register_parametrization(sub_module, "weight", 
                                  NonNegativeParametrization())
```

**Scope:** Feed-forward sublayers only  
**Not constrained:** Attention, LayerNorm, embeddings, residual connections

---

## Advanced Usage

### Multi-Seed Experiments

```bash
# Run with all 5 seeds for robust statistics
for seed in 42 1337 2024 8888 12345; do
    ./run_all.sh $seed
    mv $SCRATCH/mono_s2s_results $SCRATCH/results_seed_$seed
done
```

### Quick Testing Mode

```python
# In experiment_config.py
USE_FULL_TEST_SETS = False  # Use 200-sample subsets
NUM_EPOCHS = 1              # Faster training
```

### Interactive Debugging

```bash
sinteractive --partition=aa100 --gres=gpu:1 --time=2:00:00 --mem=32G
module load python/3.10.0 cuda/11.8
cd hpc_version/scripts
python stage_4_evaluate.py  # Run stage manually
```

---

## Troubleshooting

### Check Partition Access
```bash
sacctmgr show assoc where user=$USER format=partition,qos%50
```

### Out of Memory
```python
# Reduce in experiment_config.py
BATCH_SIZE = 2  # Was 4
```

### Module Not Found
```bash
module avail python  # Check available versions
module avail cuda
```

### Job Fails
```bash
cat logs/job_*_<jobid>.err  # Check error logs
scontrol show job <jobid>   # Check job details
```

---

## Citation & Acknowledgment

When using CURC Alpine resources, please acknowledge in publications:

> This work utilized the Alpine high performance computing resource at the University of Colorado Boulder. Alpine is jointly funded by the University of Colorado Boulder, the University of Colorado Anschutz, Colorado State University, and the National Science Foundation (award 2201538).

See: [CURC Acknowledging Resources](https://curc.readthedocs.io/en/latest/additional-resources/acknowledgements.html)

---

## Support & Documentation

- **CURC Support:** rc-help@colorado.edu
- **CURC Documentation:** https://curc.readthedocs.io/
- **Alpine Quick Start:** https://curc.readthedocs.io/en/latest/clusters/alpine/quick-start.html
- **Detailed Guide:** See `QUICKSTART.md` in this directory
- **Setup Guide:** See `README_SETUP.md` in this directory

---

## Project Structure

```
hpc_version/
├── configs/
│   └── experiment_config.py    # ← Edit paths & partition here
├── scripts/
│   ├── stage_0_setup.py        # Environment setup
│   ├── stage_1_prepare_data.py # Dataset caching
│   ├── stage_2_train_baseline.py
│   ├── stage_3_train_monotonic.py
│   ├── stage_4_evaluate.py
│   ├── stage_5_uat_attacks.py
│   ├── stage_6_hotflip_attacks.py
│   └── stage_7_aggregate.py
├── jobs/
│   └── job_*.sh                # SLURM job wrappers
├── utils/
│   └── common_utils.py         # Shared functions
├── run_all.sh                  # ← Run this to start
├── validate_setup.sh           # Pre-flight checks
├── QUICKSTART.md              # Detailed guide
└── README_SETUP.md            # Complete setup docs
```

---

## Technical Details

**Model:** T5-small (60M params)  
**Framework:** PyTorch + HuggingFace Transformers  
**Determinism:** Full seed control + CUBLAS workspace config  
**Constraint:** W = softplus(V) ≥ 0 on FFN weights  
**Optimization:** AdamW + linear warmup  
**Evaluation:** Fixed beam search, bootstrap CIs  

---

**Version:** HPC Edition v1.7  
**Status:** Production Ready  
**Last Updated:** 2025-11-13  
**Platform:** CURC Alpine (A100 GPUs)

---

*For questions about methodology or implementation, please refer to the comprehensive documentation in `QUICKSTART.md` and `README_SETUP.md`.*

