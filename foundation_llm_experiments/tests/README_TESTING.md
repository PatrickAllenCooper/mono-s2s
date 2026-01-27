# Testing Guide for Foundation LLM Pipeline

Comprehensive guide to testing the experimental pipeline locally before HPC deployment.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
bash run_tests.sh

# Run verification script
python verify_local.py
```

## Test Organization

### Test Files

| File | Purpose | Speed | Coverage |
|---|---|---|---|
| `test_config.py` | Configuration validation | Fast | Config module |
| `test_common_utils.py` | Utility functions | Fast | Utils module |
| `test_stage_scripts.py` | Stage script interfaces | Medium | Scripts |
| `test_integration.py` | End-to-end workflows | Slow | Full pipeline |
| `conftest.py` | Shared fixtures | N/A | Test infrastructure |

### Test Categories

**Unit Tests** (~150 tests, ~30 sec):
- Configuration validation
- Utility function correctness
- Monotonicity application
- File I/O operations

**Integration Tests** (~30 tests, ~2 min):
- Stage-to-stage workflows
- Checkpoint save/load
- Dependency checking
- Full pipeline simulation

**Verification Tests** (7 checks, ~2 min):
- Configuration
- Imports
- Monotonicity
- Training
- Perplexity
- File operations
- Determinism

## Running Tests

### Quick Tests (30 seconds)

```bash
bash run_tests.sh quick
```

Runs only fast unit tests for configuration and utilities.

### Full Test Suite (5 minutes)

```bash
bash run_tests.sh all
```

Runs all tests including integration tests.

### Coverage Report (3 minutes)

```bash
bash run_tests.sh coverage
```

Generates HTML coverage report in `htmlcov/index.html`.

**View coverage**:
```bash
# On local machine
open htmlcov/index.html

# On HPC (copy to local first)
scp user@hpc:path/to/htmlcov.tar.gz .
tar -xzf htmlcov.tar.gz
open htmlcov/index.html
```

### Individual Test Files

```bash
# Test config only
pytest tests/test_config.py -v

# Test utilities only
pytest tests/test_common_utils.py -v

# Test with specific pattern
pytest tests/ -k "monotonic" -v
```

### Verification Script

```bash
# Run all verifications
python verify_local.py

# Verbose output
python verify_local.py --verbose
```

## Test Coverage Goals

### Current Coverage (Estimated)

| Module | Coverage | Notes |
|---|---|---|
| `configs/experiment_config.py` | 85% | Core config well-tested |
| `utils/common_utils.py` | 80% | Core utils covered |
| `scripts/stage_0_setup.py` | 60% | Imports tested |
| `scripts/stage_1_apply_monotonicity.py` | 60% | Core logic tested |
| `scripts/stage_2_train_baseline.py` | 40% | Skeleton tested |
| `scripts/stage_3_train_monotonic.py` | 40% | Skeleton tested |
| `scripts/stage_4_evaluate.py` | 30% | Partial coverage |

### Target Coverage for HPC Deployment

**Minimum** (before first HPC run):
- Config: >80%
- Core utilities: >75%
- Monotonicity application: >85%
- Stage imports: 100%

**Recommended** (before production):
- All modules: >70%
- Critical paths: >90%
- Integration tests: All pass

## What Tests Verify

### 1. Configuration Tests (`test_config.py`)

**Validates**:
- All required config attributes exist
- Values are in reasonable ranges
- Monotonic settings > baseline settings
- SLURM settings are valid
- Path handling works
- Device detection works

**Example**:
```python
def test_warmup_ratios_valid():
    assert 0 < Config.RECOVERY_WARMUP_RATIO < 1
    assert Config.MONOTONIC_RECOVERY_WARMUP_RATIO >= Config.RECOVERY_WARMUP_RATIO
```

**Why Important**: Catches configuration errors before wasting GPU hours

### 2. Utility Tests (`test_common_utils.py`)

**Validates**:
- Softplus parametrization works correctly
- Weights stay non-negative after initialization
- Weights stay non-negative after training
- Perplexity computation is correct
- File I/O works
- Logging works
- Determinism works

**Example**:
```python
def test_forward_always_positive():
    param = NonNegativeParametrization()
    V = torch.randn(10, 10) - 5  # Negative bias
    W = param.forward(V)
    assert torch.all(W >= 0)
```

**Why Important**: Core monotonicity logic must be bulletproof

### 3. Integration Tests (`test_integration.py`)

**Validates**:
- Full pipeline can execute end-to-end
- Stages can depend on each other
- Checkpoints save and load correctly
- Results accumulate properly
- Monotonicity preserved through pipeline

**Example**:
```python
def test_end_to_end_minimal_pipeline():
    # Simulates: setup -> apply -> train -> evaluate
    # Verifies all completion flags created
    # Verifies results JSON files valid
```

**Why Important**: Catches integration bugs before HPC deployment

### 4. Verification Script (`verify_local.py`)

**Validates**:
- Environment is set up correctly
- All imports work
- Monotonicity application works on real models
- Training loop executes
- Perplexity computation works
- Determinism is enforced

**Why Important**: Final check before expensive HPC run

## Common Test Failures

### Import Errors

**Symptom**:
```
ImportError: cannot import name 'GPTNeoXConfig'
```

**Fix**:
```bash
pip install transformers --upgrade
```

### Missing Dependencies

**Symptom**:
```
ModuleNotFoundError: No module named 'pytest'
```

**Fix**:
```bash
pip install pytest pytest-cov
```

### CUDA Not Available Warnings

**Symptom**:
```
UserWarning: CUDA not available, using CPU
```

**Fix**: This is OK for local testing. GPU not needed for tests.

### Fixture Not Found

**Symptom**:
```
fixture 'mock_gpt_model' not found
```

**Fix**: Make sure running from project root:
```bash
cd foundation_llm_experiments
pytest tests/ -v
```

## Test Performance

### Speed Benchmarks

On modern laptop (CPU only):
- Config tests: ~2 seconds
- Utility tests: ~15 seconds
- Integration tests: ~90 seconds
- Full suite: ~120 seconds
- Verification: ~90 seconds

**Total**: ~5-7 minutes for complete validation

### GPU Tests

Some tests benefit from GPU but all work on CPU:
- Monotonicity application: 10x faster on GPU
- Training tests: 5x faster on GPU
- Perplexity tests: 3x faster on GPU

**Use GPU if available**: `CUDA_VISIBLE_DEVICES=0 pytest tests/`

## Continuous Testing

### Pre-Commit Checks

Before committing code:
```bash
# Quick tests
bash run_tests.sh quick

# Format code
black scripts/ utils/ tests/
isort scripts/ utils/ tests/

# Lint
flake8 scripts/ utils/ --max-line-length=100
```

### Pre-HPC Submission

Before `sbatch`:
```bash
# Full tests
bash run_tests.sh all

# Verification
python verify_local.py

# Check coverage
bash run_tests.sh coverage
```

## Debugging Failed Tests

### Get Detailed Error Output

```bash
# Run with full traceback
pytest tests/test_common_utils.py::TestMonotonicParametrization::test_forward_always_positive -vv --tb=long

# Run with print statements visible
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x
```

### Test Specific Function

```bash
# Test one function
pytest tests/test_config.py::TestExperimentConfig::test_model_name_valid -v

# Test one class
pytest tests/test_common_utils.py::TestMonotonicParametrization -v
```

### Debug with PDB

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Drop into debugger on error (not just assertion)
pytest tests/ --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb
```

## Coverage Goals

### Critical Functions (Must be >90%)

- `make_model_monotonic()` - Core monotonicity application
- `NonNegativeParametrization.forward()` - Constraint enforcement
- `NonNegativeParametrization.right_inverse()` - Initialization
- `compute_perplexity()` - Evaluation metric
- `create_completion_flag()` - Stage tracking
- `check_dependencies()` - Pipeline orchestration

### Important Functions (Should be >75%)

- Training loops
- Checkpoint save/load
- Data loading
- Logging

### Nice to Have (>50%)

- Error handling
- Edge cases
- Optional features

## Adding New Tests

### Template for Unit Test

```python
def test_new_function():
    """Test description"""
    # Setup
    input_data = ...
    
    # Execute
    result = function_to_test(input_data)
    
    # Verify
    assert result == expected_value
    assert isinstance(result, expected_type)
```

### Template for Integration Test

```python
def test_new_workflow(mock_model, tmp_path):
    """Test description"""
    # Setup complete workflow
    stage1_output = run_stage1(...)
    stage2_output = run_stage2(stage1_output, ...)
    
    # Verify end-to-end behavior
    assert stage2_output.is_valid()
```

### Template for Fixture

```python
@pytest.fixture
def new_mock_data():
    """Description"""
    data = create_test_data()
    return data
```

## Test Maintenance

### When to Update Tests

- **After changing config**: Update `test_config.py`
- **After modifying utilities**: Update `test_common_utils.py`
- **After changing stage logic**: Update corresponding `test_stage_scripts.py`
- **After adding features**: Add new integration tests

### Test Review Checklist

- [ ] All new functions have tests
- [ ] All edge cases covered
- [ ] Error conditions tested
- [ ] Integration tests pass
- [ ] Coverage >70% for new code
- [ ] Documentation updated

## Running Tests on HPC

### Before Job Submission

```bash
# On login node (no GPU needed)
cd foundation_llm_experiments
module load python
source activate mono_s2s

# Run quick tests
bash run_tests.sh quick
```

### In Interactive Session

```bash
# Request interactive session
sinteractive --partition=aa100 --gres=gpu:1 --mem=32G --time=01:00:00

# Activate environment
conda activate mono_s2s

# Run tests with GPU
cd foundation_llm_experiments
pytest tests/ -v
```

### As SLURM Job

```bash
#SBATCH --job-name=test_pipeline
#SBATCH --time=00:30:00
#SBATCH --mem=32G

conda activate mono_s2s
cd foundation_llm_experiments
pytest tests/ -v --cov=. --cov-report=json

# Save coverage report
cp coverage.json $PROJECT/foundation_llm_results/
```

## Interpreting Test Results

### All Pass (✓)

```
==================== 150 passed in 45.23s ====================
```

**Action**: Ready for HPC deployment

### Some Failures (✗)

```
==================== 145 passed, 5 failed in 47.31s ====================
```

**Action**: 
1. Review failure output
2. Fix issues
3. Rerun tests

### Import Errors

```
ImportError: cannot import name 'Config'
```

**Action**: Check Python path and module structure

### Warnings

```
100 passed, 25 warnings in 43.12s
```

**Action**: Review warnings, fix if critical

## Expected Test Output

### Successful Run Example

```
====================================================================
  FOUNDATION LLM PIPELINE - TEST SUITE
====================================================================

Running all tests...

tests/test_config.py::TestExperimentConfig::test_config_has_required_attributes PASSED
tests/test_config.py::TestExperimentConfig::test_model_name_valid PASSED
...
tests/test_integration.py::TestFullPipelineSimulation::test_end_to_end_minimal_pipeline PASSED

====================================================================
  ✓ ALL TESTS PASSED
====================================================================

Next steps:
  1. Review test output above
  2. Run verification: bash run_tests.sh verify
  3. Submit to HPC: bash run_all.sh
```

### Verification Script Output

```
====================================================================
  FOUNDATION LLM PIPELINE - LOCAL VERIFICATION
====================================================================

===========================================================================
  VERIFYING CONFIGURATION
====================================================================
  ✓ Model name: EleutherAI/pythia-1.4b
  ✓ Batch size: 8
  ✓ Learning rate: 1e-05
  ✓ Random seeds: [42, 1337, 2024, 8888, 12345]
  ✓ Device: cpu

====================================================================
  VERIFICATION SUMMARY
====================================================================
  ✓ PASS: config
  ✓ PASS: imports
  ✓ PASS: determinism
  ✓ PASS: file_ops
  ✓ PASS: monotonicity
  ✓ PASS: perplexity
  ✓ PASS: training

  Total: 7/7 checks passed

====================================================================
  ✓ ALL VERIFICATIONS PASSED
  Pipeline is ready for HPC deployment!
====================================================================
```

## Troubleshooting Tests

### Tests Hang

**Cause**: Infinite loop or network timeout

**Fix**:
```bash
# Add timeout
pytest tests/ --timeout=300  # 5 minute timeout per test
```

### Tests Fail on CI but Pass Locally

**Cause**: Different environments

**Check**:
- Python version
- Package versions
- CUDA availability
- Random seeds

### Memory Issues in Tests

**Cause**: Loading large models in tests

**Fix**: Use smaller mock models (already done in conftest.py)

## Test Coverage Commands

### Generate Coverage Report

```bash
pytest tests/ --cov=. --cov-report=html
```

### View Coverage for Specific Module

```bash
pytest tests/ --cov=utils.common_utils --cov-report=term-missing
```

### Find Untested Lines

```bash
pytest tests/ --cov=scripts --cov-report=term-missing | grep "^scripts"
```

## Pre-HPC Deployment Checklist

Run through this checklist before `sbatch`:

- [ ] All tests pass: `bash run_tests.sh all`
- [ ] Verification passes: `python verify_local.py`
- [ ] Coverage >70%: `bash run_tests.sh coverage`
- [ ] No critical warnings
- [ ] Config values reviewed
- [ ] Paths set correctly for HPC
- [ ] Time limits appropriate
- [ ] Model name correct
- [ ] Dependencies installed on HPC

## Questions?

- **"Why do tests take so long?"** → Integration tests use real (small) models
- **"Can I skip tests?"** → Not recommended, but use `bash run_tests.sh quick`
- **"What if a test fails?"** → Read error, fix code, rerun
- **"Do I need GPU for tests?"** → No, all tests work on CPU

---

**Test Philosophy**: Better to find bugs locally (seconds) than on HPC (hours).

**Recommendation**: Run full test suite before every HPC submission.

**Coverage Target**: >70% for deployment, >85% for publication.
