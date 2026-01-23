# Test Coverage Status

## Current Coverage: 24.92%

### Target: 90%

## Module Breakdown

| Module | Statements | Miss | Coverage | Status |
|--------|-----------|------|----------|--------|
| `experiment_config.py` | 109 | 6 | **92.00%** | ✅ Excellent |
| `common_utils.py` | 337 | 103 | **65.99%** | ⚠️  Needs improvement |
| `stage_0_setup.py` | 54 | 42 | 20.69% | ❌ Poor |
| `stage_1_prepare_data.py` | 104 | 96 | 7.02% | ❌ Poor |
| `stage_2_train_baseline.py` | 196 | 181 | 7.01% | ❌ Poor |
| `stage_3_train_monotonic.py` | 226 | 210 | 6.40% | ❌ Poor |
| `stage_4_evaluate.py` | 138 | 123 | 9.26% | ❌ Poor |
| `stage_5_uat_attacks.py` | 277 | 249 | 8.72% | ❌ Poor |
| `stage_6_hotflip_attacks.py` | 193 | 171 | 10.53% | ❌ Poor |
| `stage_7_aggregate.py` | 231 | 213 | 6.59% | ❌ Poor |
| **TOTAL** | **1865** | **1394** | **24.92%** | ❌ Needs work |

## Path to 90% Coverage

### Completed
- ✅ Test infrastructure setup (pytest + coverage.py)
- ✅ Comprehensive configuration tests (92% coverage)
- ✅ Core utility function tests (66% coverage)
- ✅ Test fixtures and helpers
- ✅ CI/CD integration ready

### Remaining Work

#### 1. Stage Script Testing (Critical - 75% of codebase)

The stage scripts (`stage_0` through `stage_7`) make up **1,400 of 1,865 statements** (75% of codebase) but are currently only 6-10% covered.

**Challenges:**
- Heavy integration with transformers/T5 models
- TensorFlow import issues on macOS
- Require HPC/SLURM environment setup
- Complex training loops and model operations

**Solutions:**
1. **Extract testable logic**: Refactor stage scripts to separate:
   - Pure logic functions (easy to test)
   - I/O operations (mockable)
   - Model operations (can use smaller mock models)

2. **Mock external dependencies**:
   ```python
   @patch('transformers.T5ForConditionalGeneration')
   @patch('datasets.load_dataset')
   def test_stage_1_logic(mock_dataset, mock_model):
       # Test the logic without real models
   ```

3. **Integration tests with minimal models**:
   - Use tiny T5 configs for testing
   - Mock SLURM-specific operations
   - Test data flow through pipelines

#### 2. Improve `common_utils.py` (from 66% → 90%+)

**Missing coverage:**
- Model loading functions (lines 276-343)
- Generation functions (lines 397-417)
- Advanced checkpoint operations (lines 589-616)

**Action items:**
- Add tests for model loading with various checkpoint scenarios
- Test generation with edge cases (empty input, very long input)
- Test checkpoint saving/loading with corrupted files

#### 3. Add Integration Tests

Current tests are mostly unit tests. Need integration tests that:
- Run multiple stages in sequence
- Verify data flow between stages
- Test error recovery and resilience
- Validate end-to-end pipelines (with mocked models)

## Implementation Strategy

### Phase 1: Quick Wins (Target: 40% → 60% coverage)
1. Fix failing tests (monkeypatch issues, TensorFlow imports)
2. Add unit tests for pure functions in stage scripts
3. Complete `common_utils.py` coverage

### Phase 2: Stage Script Refactoring (Target: 60% → 80% coverage)
1. Extract testable logic from each stage script into separate functions
2. Create comprehensive mocks for transformers/datasets
3. Write unit tests for extracted logic

### Phase 3: Integration Testing (Target: 80% → 90% coverage)
1. Create end-to-end test scenarios
2. Test error handling and edge cases
3. Verify stage dependencies and data flow

## Technical Challenges & Solutions

### Challenge 1: TensorFlow Import Errors
**Problem:** Transformers imports TensorFlow which fails on macOS
**Solution:** 
- Use `pytest.mark.skip` for problematic imports
- Mock `transformers` module at import time
- Create minimal test doubles for T5 models

### Challenge 2: SLURM Dependencies
**Problem:** Stage scripts expect SLURM environment
**Solution:**
- Mock SLURM-specific operations
- Use environment variable overrides
- Create test harness that simulates SLURM

### Challenge 3: Large Model Operations
**Problem:** Testing with real T5 models is slow and resource-intensive
**Solution:**
- Create minimal T5Config for testing
- Use `unittest.mock.MagicMock` for model operations
- Test logic separately from model inference

## Next Steps

1. **Immediate** (1-2 hours):
   - Fix failing tests related to monkeypatch and imports
   - Skip or properly mock TensorFlow-dependent tests
   - Commit current progress to version control

2. **Short-term** (4-8 hours):
   - Refactor stage scripts to extract testable functions
   - Write comprehensive unit tests for extracted logic
   - Achieve 60%+ coverage

3. **Medium-term** (8-16 hours):
   - Create comprehensive mock framework for transformers
   - Write integration tests for stage pipelines
   - Achieve 80%+ coverage

4. **Long-term** (16-24 hours):
   - Fine-tune edge cases and error scenarios
   - Add property-based tests with Hypothesis
   - Achieve 90%+ coverage

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run without TensorFlow-dependent tests
pytest -m "not slow"

# View HTML coverage report
pytest && open coverage_html_report/index.html

# Run specific test file
pytest tests/test_experiment_config.py -v
```

## Coverage Goals by Module

- Configuration: ✅ 92% (Achieved!)
- Utilities: ⚠️  66% → Target: 90%
- Stage Scripts: ❌ 6-10% → Target: 85%
- Overall: ❌ 25% → Target: 90%

## Recommendations

Given the project structure (75% is HPC integration code), consider:

1. **Prioritize testing business logic**: Core algorithms and data transformations should have 95%+ coverage

2. **Accept lower coverage for integration code**: SLURM job wrappers and model loading may realistically achieve 70-80% coverage due to external dependencies

3. **Use CI/CD coverage trends**: Track coverage improvements over time rather than absolute thresholds

4. **Consider excluding some files**: HPC job scripts (`job_*.sh`) and infrastructure code might warrant exclusion from coverage metrics

## Last Updated
2026-01-21
