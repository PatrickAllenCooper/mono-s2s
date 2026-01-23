# Mono-S2S (Monotonic Seq2Seq): HPC Fair-Comparison Pipeline

**Status:** Active research project for ICML 2025 submission  
**Latest Update:** 2026-01-21  
**Test Coverage:** 98.01% ✅  
**Current Configuration:** Fair comparison (7 epochs both models), full test sets

This repo contains a **fully automated SLURM/HPC pipeline** to test whether **local monotonic constraints in T5 feed-forward sublayers (FFNs)** improve adversarial robustness in seq2seq summarization, under a **methodologically fair three-way comparison**:

| Model | Description | Purpose |
|------|-------------|---------|
| **Standard T5** | Pre-trained `t5-small`, not fine-tuned | Reference baseline |
| **Baseline T5** | Fine-tuned for 7 epochs, unconstrained | Fair control |
| **Monotonic T5** | Fine-tuned for 7 epochs, FFN weights constrained \(W \ge 0\) | Treatment |

**Critical:** Both fine-tuned models use **identical** epochs (7), learning rate (5e-5), batch size (4), and datasets for fair comparison.

**Important scope limitation:** Constraints apply to **FFN sublayers only**. The full encoder/decoder is not globally monotonic due to attention, LayerNorm, residuals, etc.

---

## Repo layout

- **`hpc_version/`**: the production SLURM pipeline (configs, stage scripts, job wrappers, orchestration).
- **`mono_s2s_v1_7.py`**: legacy monolithic script (kept for reference; the pipeline lives in `hpc_version/`).

---

## Quick start (CURC Alpine, `aa100`)

### Prerequisites (Alpine disk quota gotcha)

Alpine `$HOME` quota is tiny (~2–5GB). **Install conda + envs under `/projects/$USER/`**, not `~`.

```bash
ssh your_username@login.rc.colorado.edu

# If you previously installed conda in $HOME, remove it (optional but recommended)
rm -rf ~/miniconda3 ~/anaconda3
sed -i '/>>> conda initialize >>>/,/<<< conda initialize <<</d' ~/.bashrc
source ~/.bashrc

# Install Miniconda to /projects
cd /projects/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /projects/$USER/miniconda3
/projects/$USER/miniconda3/bin/conda init bash
source ~/.bashrc

which conda  # must be /projects/$USER/miniconda3/bin/conda
```

### Setup (clone + env + deps)

```bash
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s

cd /projects/$USER
git clone https://github.com/PatrickAllenCooper/mono-s2s.git
cd mono-s2s/hpc_version

# PyTorch: prefer pip wheels on Alpine (avoids common conda/MKL conflicts)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other deps
pip install transformers datasets rouge-score scipy pandas tqdm sentencepiece protobuf

# One-time: make scripts executable
chmod +x run_all.sh jobs/*.sh scripts/*.py

# Validate setup
./validate_setup.sh
```

### Configure (paths + partition)

Edit `hpc_version/configs/experiment_config.py`:

- Set/verify `SCRATCH_DIR` and `PROJECT_DIR` (CURC often provides `$SCRATCH` and `$PROJECT`).
- Set `SLURM_PARTITION = "aa100"` for A100s on Alpine.

### Run (one command)

```bash
cd /projects/$USER/mono-s2s/hpc_version
conda activate mono_s2s

./run_all.sh          # default seed
./run_all.sh 42       # custom seed
```

### Monitor

```bash
squeue -u $USER
tail -f logs/job_*.out
ls $SCRATCH/mono_s2s_work/*.flag
```

---

## Quick start (other SLURM clusters)

You need Python 3.10+, CUDA-enabled PyTorch, and the Python deps above.

```bash
cd hpc_version

# Example (modules vary by cluster):
module load anaconda cuda
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s

conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers datasets rouge-score scipy pandas tqdm sentencepiece protobuf

chmod +x run_all.sh jobs/*.sh scripts/*.py
./validate_setup.sh
```

Then edit `configs/experiment_config.py` for your cluster’s paths + partition and run `./run_all.sh`.

---

## What the pipeline runs

Stages are orchestrated via `hpc_version/run_all.sh` and SLURM job wrappers in `hpc_version/jobs/`:

```
Stage 0 → Stage 1 → ┬ Stage 2 (Baseline)     ┐
                    └ Stage 3 (Monotonic)    ┘ → Stage 4 → ┬ Stage 5 (UAT)      ┐
                                                             └ Stage 6 (HotFlip)  ┘ → Stage 7
```

### Resource requirements (Alpine `aa100` ballpark)

| Stage | GPU | Memory | Time |
|------:|:---:|:------:|:----:|
| 0 Setup | No | 16 GB | 30 min |
| 1 Data prep | No | 32 GB | 2 hr |
| 2 Train baseline | Yes | 64 GB | 10–12 hr |
| 3 Train monotonic | Yes | 64 GB | 10–12 hr |
| 4 Evaluate | Yes | 32 GB | 2–4 hr |
| 5 UAT | Yes | 16 GB | 2–3 hr |
| 6 HotFlip | Yes | 16 GB | 1–2 hr |
| 7 Aggregate | No | 8 GB | 15 min |

---

## Outputs

After completion you’ll find results under:

- **Scratch (working + temporary results)**: `$SCRATCH/mono_s2s_work/` and `$SCRATCH/mono_s2s_results/`
- **Permanent copy**: `$PROJECT/mono_s2s_final_results/`

Key files:

- `evaluation_results.json` (ROUGE + bootstrap 95% CIs)
- `uat_results.json` (UAT attacks + transfer matrix)
- `hotflip_results.json` (HotFlip attack results)
- `final_results.json` + `experiment_summary.txt` (aggregation + human-readable summary)

---

## Datasets & metrics

### Training Data
- DialogSum, HighlightSum, arXiv abstracts (~237K examples total)
- Uses train splits only - validation splits for early stopping
- **Critical:** CNN/DailyMail held-out entirely (NOT in training data)

### Evaluation Data
- **CNN/DailyMail v3.0.0** test split (11,490 examples) - primary evaluation
- **XSUM** test split (11,334 examples) - generalization check
- **SAMSum** test split (819 examples) - dialogue domain

### Metrics
- **ROUGE-1/2/L** with bootstrap 95% CIs (1,000 resamples)
- **Adversarial robustness:** UAT + HotFlip attacks with degradation metrics
- **Transfer attacks:** Cross-model attack evaluation matrix
- **All results timestamped** with date, time, and seed for tracking

---

## Monotonic constraint implementation (FFN-only)

Weights in the FFN sublayers are constrained non-negative using softplus parametrization:

\[
W = \mathrm{softplus}(V) \ge 0
\]

This is applied to the FFN linear layers (e.g., `wi`, `wo`, and gated variants) while leaving attention/LayerNorm/residuals unconstrained.

---

## Current Configuration (ICML 2025)

**Fair Comparison Settings** (both models identical):
- **Epochs:** 7 (both baseline and monotonic)
- **Learning Rate:** 5e-5
- **Batch Size:** 4
- **Gradient Clip:** 1.0
- **Optimizer:** AdamW with weight decay 0.01

**Evaluation Settings:**
- **USE_FULL_TEST_SETS:** True (11,490 examples for CNN/DM)
- **Bootstrap Samples:** 1,000 resamples for CIs
- **Attack Budget:** 5 tokens for triggers/flips

**Only Difference:**
- Warmup: Baseline 10% vs Monotonic 15% (for softplus stability)

### Diagnostic Tools

If you see dataset-loading issues:
```bash
cd hpc_version
python test_dataset_loading.py  # Test dataset access
python test_improvements.py     # Test improvements before full run
```

---

## Troubleshooting

- **No space left on device (Alpine)**: ensure conda/envs are in `/projects/$USER/`, not `$HOME`.
- **CUDA OOM**: reduce `BATCH_SIZE` (and/or eval batch size) in `configs/experiment_config.py`.
- **Dataset download failures**: pre-download on nodes with internet, set `HF_DATASETS_CACHE` to a project directory, and retry.
- **Job fails**: inspect `hpc_version/logs/job_*.err` and `$SCRATCH/mono_s2s_work/stage_logs/`.
- **Start fresh**: use `hpc_version/clean_all.sh` (supports `--force` and `--keep-cache`).

---

## Citation / acknowledgment (CURC Alpine)

If you use CURC Alpine resources, acknowledge:

> This work utilized the Alpine high performance computing resource at the University of Colorado Boulder. Alpine is jointly funded by the University of Colorado Boulder, the University of Colorado Anschutz, Colorado State University, and the National Science Foundation (award 2201538).

---

## Documentation

### Core Documentation
- **README.md** (this file) - Setup and pipeline overview
- **TESTING.md** - Test suite guide (98% coverage achieved)
- **PAPER_STATUS.md** - ICML submission status and roadmap

### Paper Development
- **documentation/monotone_llms_paper.tex** - ICML paper draft
- **documentation/QUICK_ICML_RECOMMENDATIONS.md** - Priority fixes
- **documentation/paper_methods_critique.md** - Detailed methods review

### Testing Documentation  
- **TEST_COVERAGE_FINAL.md** - Coverage achievement report (98.01%)
- **tests/README.md** - Test suite reference

### HPC Documentation
- **hpc_version/IMPROVEMENTS_SUMMARY.md** - Pipeline improvements
- **hpc_version/CHANGES_AT_A_GLANCE.md** - Quick reference

### Configuration
All experimental settings in `hpc_version/configs/experiment_config.py`

---

## Recent Updates (2026-01-21)

1. **Fair Comparison:** Both models now train 7 epochs (was 5 vs 7)
2. **Full Evaluation:** USE_FULL_TEST_SETS=True (11,490 examples vs 200)
3. **Test Coverage:** Achieved 98.01% coverage (target 90%)
4. **Timestamped Results:** All outputs include run metadata
5. **Analysis Tracking:** Added flags for gradient norms, computational cost


