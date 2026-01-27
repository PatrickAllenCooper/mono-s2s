# Models and Datasets Used

Complete reference for all models and datasets required by this pipeline.

## Models

### Primary Model: Pythia-1.4B

**Identifier**: `EleutherAI/pythia-1.4b`

**Specifications**:
- Parameters: 1.4 billion
- Architecture: GPTNeoX (decoder-only transformer)
- Hidden size: 2048
- Layers: 24
- Attention heads: 16
- FFN intermediate: 8192 (4x hidden size)
- FFN parameters: ~560M (40% of total)
- Context window: 2048 tokens
- Vocabulary: ~50k tokens

**Download Size**: ~6 GB

**Source**: Hugging Face Hub

**License**: Apache 2.0 (fully open)

**Requirements**:
- No authentication needed
- Public model
- Works with `transformers>=4.30.0`

**Verify Download**:
```bash
python verify_downloads.py --quick
```

### Test Models (for local validation)

**GPT-2** (`gpt2`):
- Used in local tests and verification
- Size: 124M parameters (~500 MB)
- Fast to download
- Used to validate pipeline logic

**GPT-2 Config** (tiny custom):
- Used in unit tests
- Created programmatically (no download)
- Vocab: 1000, Layers: 2, Hidden: 128
- For fast test execution

## Datasets

### Primary Dataset: The Pile

**Identifier**: `EleutherAI/pile`

**Specifications**:
- Size: 825 GB (deduped)
- Tokens: ~300 billion
- Documents: ~210 million
- Domains: 22 diverse sources
- Splits: train, validation, test

**Download Requirements**:
- **Train split**: ~800 GB (use streaming)
- **Validation split**: ~1 GB (recommended for testing)
- **Test split**: ~25 GB

**Structure**:
```python
{
    'text': str,      # Main text content
    'meta': dict,     # Metadata (source, etc.)
}
```

**Access Method**:
```python
from datasets import load_dataset

# Streaming (for train)
pile = load_dataset("EleutherAI/pile", split="train", streaming=True)

# Non-streaming (for validation/test)
pile = load_dataset("EleutherAI/pile", split="validation")
```

**Requirements**:
- No authentication needed
- Public dataset
- Requires `datasets>=2.14.0`
- Recommended: use streaming for train split

**Verify Download**:
```bash
python verify_downloads.py
```

### Alternative Datasets (if Pile unavailable)

#### C4 (Colossal Clean Crawled Corpus)

**Identifier**: `allenai/c4`

**Specifications**:
- Size: ~750 GB (en variant)
- Cleaner than Pile (filtered Common Crawl)
- Well-supported by Hugging Face

**Usage**:
```python
load_dataset("allenai/c4", "en", split="validation")
```

#### OpenWebText

**Identifier**: `openwebtext`

**Specifications**:
- Size: ~40 GB
- Reddit-sourced (like GPT-2 training data)
- Smaller, faster to download

**Usage**:
```python
load_dataset("openwebtext", split="train")
```

### Evaluation Datasets (Optional)

These are **optional** - pipeline works without them:

#### LAMBADA
- **ID**: `lambada`
- **Size**: ~5 MB
- **Purpose**: Next-word prediction accuracy

#### HellaSwag
- **ID**: `hellaswag`
- **Size**: ~10 MB
- **Purpose**: Commonsense reasoning

#### Winogrande
- **ID**: `winogrande`
- **Size**: ~5 MB
- **Purpose**: Coreference resolution

#### TruthfulQA
- **ID**: `truthful_qa`
- **Size**: ~2 MB
- **Purpose**: Factuality testing

## Download Strategy

### For Local Testing

**Recommended**:
1. Test with GPT-2 (small, fast)
2. Test with C4 validation split (medium)
3. Skip Pythia download (test on HPC)

**Command**:
```bash
python verify_downloads.py --quick
```

**Time**: ~2-5 minutes

### For HPC Deployment

**Recommended**:
1. Test Pythia tokenizer/config (fast)
2. Test Pile validation split (medium)
3. Full model download happens in stage 0 on HPC

**Command**:
```bash
# On HPC login node
python verify_downloads.py
```

**Time**: ~10-15 minutes (without full model)

### For Full Validation

**Only if needed** (slow):
```bash
python verify_downloads.py --full
```

**Downloads**:
- Full Pythia-1.4B model (~6 GB)
- Pile validation split (~1 GB)

**Time**: ~20-30 minutes (depending on connection)

## Troubleshooting Downloads

### Model Download Fails

**Error**: `HTTPError: 403 Client Error`

**Solutions**:
1. Check internet connection
2. Try different model:
   ```python
   MODEL_NAME = "EleutherAI/pythia-410m"  # Smaller
   ```
3. Set up HF token (if gated):
   ```bash
   huggingface-cli login
   ```

### Dataset Download Fails

**Error**: `ConnectionError` or timeout

**Solutions**:
1. Use validation split instead of train:
   ```python
   pile = load_dataset("EleutherAI/pile", split="validation")
   ```
2. Use alternative dataset:
   ```python
   TRAINING_DATASET = "allenai/c4"
   ```
3. Enable streaming:
   ```python
   load_dataset(..., streaming=True)
   ```

### Download Extremely Slow

**Symptoms**: Hours to download

**Solutions**:
1. On HPC: Download to $SCRATCH (fast filesystem)
2. Set cache directory:
   ```python
   cache_dir = os.environ.get("SCRATCH", "/tmp") + "/hf_cache"
   ```
3. Use validation split for testing first

### Firewall/Proxy Issues

**Symptoms**: Cannot reach huggingface.co

**Solutions**:
1. Check firewall rules
2. Configure proxy:
   ```bash
   export HTTP_PROXY=http://proxy.example.com:8080
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```
3. On HPC: May need to request firewall exemption

## Storage Requirements

### Local Testing

- **Minimum**: 10 GB (for GPT-2 + C4 validation)
- **Recommended**: 20 GB (for Pythia tokenizer + Pile validation)

### HPC Full Run

- **Scratch** (temporary):
  - Model cache: ~10 GB (Pythia + tokenizer)
  - Dataset cache: ~50 GB (Pile samples)
  - Checkpoints: ~20 GB per seed
  - Working files: ~10 GB
  - **Total**: ~90 GB per seed

- **Project** (persistent):
  - Final results: ~100 MB per seed
  - Best checkpoints: ~5 GB per seed
  - **Total**: ~5 GB per seed

## Bandwidth Requirements

### Initial Downloads

- **Pythia-1.4B**: ~6 GB
- **Pile validation**: ~1 GB
- **Pile train** (streaming): Continuous (~100 MB/sec if fast)

### Subsequent Runs

- Models cached → No re-download
- Datasets cached → Only stream new samples
- Much faster after first run

## Cache Management

### Default Cache Locations

**Transformers**:
- Linux: `~/.cache/huggingface/`
- On HPC: `$SCRATCH/huggingface_cache/` (set via `HF_HOME`)

**Datasets**:
- Linux: `~/.cache/huggingface/datasets/`
- On HPC: `$SCRATCH/huggingface_cache/datasets/`

### Manual Cache Setup

```python
# In scripts
cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
```

### Clearing Cache

If needed:
```bash
# Clear all
rm -rf ~/.cache/huggingface/

# Clear only models
rm -rf ~/.cache/huggingface/transformers/

# Clear only datasets
rm -rf ~/.cache/huggingface/datasets/
```

## Pre-Download for HPC

To avoid download issues during job execution:

```bash
# On HPC login node (before submitting jobs)
cd foundation_llm_experiments

# Download model
python -c "
from transformers import AutoTokenizer, AutoConfig
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b')
config = AutoConfig.from_pretrained('EleutherAI/pythia-1.4b')
print('✓ Model cached')
"

# Download dataset
python -c "
from datasets import load_dataset
pile = load_dataset('EleutherAI/pile', split='validation')
print(f'✓ Dataset cached: {len(pile)} samples')
"
```

This ensures downloads complete before jobs start.

## Verification Commands

### Quick Check (2-5 min)

```bash
python verify_downloads.py --quick
```

Verifies:
- Dependencies installed
- Internet connection working
- Model info accessible
- Dataset structure correct
- No full downloads

### Full Check (10-15 min)

```bash
python verify_downloads.py
```

Verifies:
- All of quick check
- Pythia tokenizer/config download
- Pile validation split download
- Streaming mechanism works

### Complete Check (20-30 min)

```bash
python verify_downloads.py --full
```

Verifies:
- All of full check
- Complete Pythia-1.4B model download (~6 GB)
- Tests full forward pass

## Expected Outputs

### Successful Verification

```
====================================================================
  MODEL AND DATASET DOWNLOAD VERIFICATION
====================================================================

====================================================================
  VERIFYING INTERNET CONNECTION
====================================================================
  ✓ Can reach huggingface.co
  ✓ Can reach datasets server

====================================================================
  VERIFYING DEPENDENCIES
====================================================================
  ✓ PyTorch         version 2.0.1
  ✓ Transformers    version 4.30.2
  ✓ Datasets        version 2.14.0
  ...

====================================================================
  VERIFYING MODEL INFORMATION
====================================================================
  ✓ Model found on Hugging Face
    Model ID: EleutherAI/pythia-1.4b
    Downloads: 50,000+
    Approx size: 5.8 GB

====================================================================
  TESTING PILE DATASET ACCESS
====================================================================
  ✓ Dataset loaded successfully
  ✓ Read sample successfully
  ✓ Pile dataset structure verified

====================================================================
  VERIFICATION SUMMARY
====================================================================
  ✓ PASS: internet
  ✓ PASS: dependencies
  ✓ PASS: model_info
  ✓ PASS: model_download
  ✓ PASS: dataset_access
  ✓ PASS: streaming_test

  Total: 10/10 checks passed

====================================================================
  ✓ ALL DOWNLOAD VERIFICATIONS PASSED
====================================================================

Models and datasets are accessible!
```

### If Issues Found

```
====================================================================
  ⚠️  SOME CHECKS FAILED
====================================================================

Failed checks: dataset_access, internet

Review errors above and:
  1. Check internet connection
  2. Install missing dependencies
  3. Try quick mode if timing out
```

## FAQ

**Q: Do I need to download Pile locally?**
A: No. Test with `--quick` mode locally. Download happens on HPC.

**Q: Can I use a different model?**
A: Yes. Edit `MODEL_NAME` in `configs/experiment_config.py`.

**Q: What if Pile is too large?**
A: Use validation split (1GB) or alternative dataset (C4, OpenWebText).

**Q: Do downloads need authentication?**
A: No. All models/datasets are public.

**Q: What if my institution blocks Hugging Face?**
A: Contact IT for firewall exemption or download elsewhere and transfer.

---

**Verification Tool**: `python verify_downloads.py`

**Quick Test**: `python verify_downloads.py --quick` (2-5 min)

**Before HPC**: Run verification to catch issues early
