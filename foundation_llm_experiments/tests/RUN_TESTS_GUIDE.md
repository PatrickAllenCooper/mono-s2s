# How to Run Tests - Complete Guide

## TL;DR

```bash
cd foundation_llm_experiments

# Quick check (30 sec)
bash run_tests.sh quick && python verify_local.py

# Full validation (10 min)
bash run_tests.sh all && python test_pipeline_local.py
```

If both pass → Ready for HPC deployment.

## Detailed Testing Workflow

### Step 1: Install Dependencies (2 minutes)

```bash
# Install testing dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pytest; print(f'pytest {pytest.__version__}')"
```

### Step 2: Quick Smoke Test (30 seconds)

```bash
bash run_tests.sh quick
```

**Tests**: Configuration and core utilities (~40 tests)

**Expected**:
```
==================== 40 passed in 2.35s ====================
```

**If fails**: Fix configuration or utility issues before proceeding.

### Step 3: Full Test Suite (5 minutes)

```bash
bash run_tests.sh all
```

**Tests**: All 155+ tests including integration

**Expected**:
```
==================== 155 passed in 4.87s ====================
```

**If fails**: Review error messages, fix issues, rerun.

### Step 4: Verification Script (2 minutes)

```bash
python verify_local.py
```

**Checks**: 7 critical verifications

**Expected**:
```
✓ PASS: config
✓ PASS: imports
✓ PASS: determinism
✓ PASS: file_ops
✓ PASS: monotonicity
✓ PASS: perplexity
✓ PASS: training

Total: 7/7 checks passed
✓ ALL VERIFICATIONS PASSED
```

**If fails**: Check specific failure, fix, rerun.

### Step 5: Local Pipeline Test (3 minutes)

```bash
python test_pipeline_local.py --verbose
```

**Simulates**: Full 7-stage pipeline with tiny model

**Expected**:
```
STAGE 0: SETUP ✓
STAGE 1: APPLY MONOTONICITY ✓
STAGE 2: BASELINE TRAINING ✓
STAGE 3: MONOTONIC TRAINING ✓
STAGE 4: EVALUATION ✓
STAGE 5: UAT ATTACKS ✓ (simulated)
STAGE 6: HOTFLIP ATTACKS ✓ (simulated)
STAGE 7: AGGREGATE ✓

✓ ALL STAGES COMPLETED SUCCESSFULLY
```

**If fails**: Debug specific stage, fix, rerun.

### Step 6: Coverage Report (3 minutes)

```bash
bash run_tests.sh coverage
```

**Generates**: `htmlcov/index.html` with line-by-line coverage

**Expected**: Overall coverage >70%

**Review**:
- Open `htmlcov/index.html` in browser
- Check red (uncovered) lines
- Ensure critical functions >90% coverage

## Test Modes Explained

### `bash run_tests.sh quick`

**Runs**: Config + Core utilities (40 tests, 30 sec)
**When**: Quick sanity check during development
**Use**: Frequent checks while coding

### `bash run_tests.sh unit`

**Runs**: All unit tests (135 tests, 2 min)
**When**: After implementing new functions
**Use**: Verify new code works in isolation

### `bash run_tests.sh integration`

**Runs**: Integration tests only (20 tests, 90 sec)
**When**: After connecting components
**Use**: Verify stages work together

### `bash run_tests.sh coverage`

**Runs**: All tests + coverage report (155 tests, 3 min)
**When**: Before HPC deployment
**Use**: Ensure sufficient code coverage

### `bash run_tests.sh all`

**Runs**: All tests (155 tests, 5 min)
**When**: Final check before HPC
**Use**: Complete validation

### `bash run_tests.sh verify`

**Runs**: `verify_local.py` (7 checks, 2 min)
**When**: Pre-deployment final check
**Use**: Critical path verification

## Understanding Test Output

### Successful Test

```
tests/test_config.py::TestExperimentConfig::test_model_name_valid PASSED [12%]
```

- `PASSED` = Test succeeded
- `[12%]` = Overall progress

### Failed Test

```
tests/test_common_utils.py::TestMonotonicParametrization::test_forward_always_positive FAILED [45%]

=================================== FAILURES ===================================
___________ TestMonotonicParametrization.test_forward_always_positive __________

    def test_forward_always_positive(self):
        param = NonNegativeParametrization()
        V = torch.randn(10, 10)
>       W = param.forward(V)
E       AttributeError: 'NoneType' object has no attribute 'forward'

tests/test_common_utils.py:125: AttributeError
```

**Read**:
- Function name: `test_forward_always_positive`
- Error type: `AttributeError`
- Line: `tests/test_common_utils.py:125`
- Issue: `param` is None

**Fix**: Check why `NonNegativeParametrization()` returns None

### Skipped Test

```
tests/test_stage_scripts.py::TestStage0Setup::test_stage0_execution SKIPPED [23%]
```

- `SKIPPED` = Test intentionally skipped (e.g., requires GPU/download)
- Usually OK, review reason in code

### Warning

```
tests/test_integration.py::test_full_pipeline 
  /path/to/file.py:42: UserWarning: CUDA not available
```

- Not fatal, but review
- Most warnings OK for local testing

## Debugging Failed Tests

### Get More Details

```bash
# Verbose output
pytest tests/test_config.py -v

# Very verbose (full tracebacks)
pytest tests/test_config.py -vv

# Show print statements
pytest tests/test_config.py -s

# Stop on first failure
pytest tests/ -x
```

### Test Specific Function

```bash
# Test one function
pytest tests/test_common_utils.py::TestMonotonicParametrization::test_forward_always_positive

# Test one class
pytest tests/test_config.py::TestExperimentConfig

# Test matching pattern
pytest tests/ -k "monotonic" -v
```

### With Debugger

```bash
# Drop into pdb on failure
pytest tests/ --pdb

# More interactive
pytest tests/ --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb
```

## Coverage Interpretation

### Good Coverage (>80%)

```
utils/common_utils.py    245    195    80%   12-15, 45, 89-91
```

- 245 total lines
- 195 executed
- 80% coverage
- Lines 12-15, 45, 89-91 not covered

**Action**: Review uncovered lines, add tests if critical.

### Acceptable Coverage (70-80%)

```
scripts/stage_2_train_baseline.py    180    135    75%   various
```

**Action**: OK for deployment, improve later.

### Low Coverage (<70%)

```
scripts/stage_5_uat_attacks.py    250    120    48%   many lines
```

**Action**: Add more tests or implement stage fully.

### Coverage Gaps

**Not Covered (Acceptable)**:
- Error handling edge cases
- Optional features
- Logging/printing code

**Not Covered (Fix Before HPC)**:
- Core monotonicity logic
- Training loops
- Checkpoint save/load
- Critical validation

## Test Performance

### Timing Breakdown

On modern laptop (CPU):
- Config tests: ~2 sec
- Utility tests: ~15 sec
- Stage tests: ~30 sec
- Integration tests: ~90 sec
- **Total**: ~2-5 min

On HPC login node (CPU):
- Similar performance
- May be slower if shared node

With GPU (optional):
- 2-3x faster for model tests
- Not necessary for passing

## Common Test Failures & Fixes

### Import Error: Cannot import module

```
ImportError: No module named 'transformers'
```

**Fix**:
```bash
pip install transformers
```

### Import Error: Cannot import from utils

```
ModuleNotFoundError: No module named 'utils'
```

**Fix**: Run from project root:
```bash
cd foundation_llm_experiments
pytest tests/ -v
```

### CUDA Warnings

```
UserWarning: CUDA not available, using CPU
```

**Fix**: This is OK! Tests work on CPU.

### Model Download in Tests

```
HTTPError: 403 Client Error
```

**Fix**: Tests use tiny local models, shouldn't download.
Check `conftest.py` uses `gpt2` (small, cached).

### Numerical Precision Errors

```
AssertionError: Tensors not close enough
```

**Fix**: Use `torch.allclose(a, b, atol=1e-5)` instead of `==`.

## Pre-HPC Deployment Command Sequence

### Complete Validation (10 minutes)

Run in order:

```bash
cd foundation_llm_experiments

# 1. Quick tests (30 sec)
bash run_tests.sh quick || exit 1

# 2. Full tests (5 min)
bash run_tests.sh all || exit 1

# 3. Verification (2 min)
python verify_local.py || exit 1

# 4. Local pipeline (3 min)
python test_pipeline_local.py || exit 1

# 5. Coverage check (3 min)
bash run_tests.sh coverage
# Review: htmlcov/index.html

echo ""
echo "✓ ALL VALIDATIONS PASSED"
echo "Ready for HPC deployment!"
```

**If ANY step fails**: Stop, fix, rerun from step 1.

## Troubleshooting Guide

### All Tests Fail

**Symptom**: Massive test failures

**Likely Cause**: Wrong Python version or missing dependencies

**Fix**:
```bash
python --version  # Should be 3.10+
pip install -r requirements.txt
```

### Some Tests Fail

**Symptom**: 10-20% failure rate

**Likely Cause**: Code bugs or test issues

**Fix**: Review failures one by one, fix code or tests.

### Tests Hang

**Symptom**: Tests don't complete

**Likely Cause**: Infinite loop or blocking operation

**Fix**:
```bash
# Add timeout
pip install pytest-timeout
pytest tests/ --timeout=60
```

### Verification Fails

**Symptom**: `verify_local.py` returns errors

**Likely Cause**: Environment issues

**Fix**: Check each verification individually:
```bash
python verify_local.py --verbose
# Review which check failed
# Fix that specific issue
```

## Success Criteria

Before HPC deployment, ensure:

- [ ] ✅ `bash run_tests.sh all` → 150+ passed
- [ ] ✅ `python verify_local.py` → 7/7 passed
- [ ] ✅ `python test_pipeline_local.py` → All stages complete
- [ ] ✅ Coverage >70% (check `htmlcov/index.html`)
- [ ] ✅ No critical errors in any test
- [ ] ✅ All scripts importable
- [ ] ✅ Monotonicity verified working

## Questions & Answers

**Q: Do I need GPU for tests?**
A: No, all tests run on CPU.

**Q: How long should tests take?**
A: 5-10 minutes total for complete validation.

**Q: What if a test fails?**
A: Fix the issue before HPC deployment. Tests catch bugs early.

**Q: Can I skip tests?**
A: Not recommended. Each test validates something important.

**Q: What coverage is needed?**
A: Minimum 70% overall, 90% for critical functions.

**Q: Do tests need internet?**
A: Only for downloading small test models (gpt2, cached after first run).

---

**Remember**: 10 minutes of local testing can save 60+ hours of GPU time.

**Current Status**: ✅ Comprehensive test suite ready

**Next Action**: Run `bash run_tests.sh all && python verify_local.py`
