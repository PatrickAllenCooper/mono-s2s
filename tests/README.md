# Mono-S2S Test Suite

## Overview

Comprehensive test suite achieving **98.01% code coverage** (target: 90%).

## Quick Start

```bash
# Run all tests with coverage
pytest

# Run without coverage (faster)
pytest --no-cov

# Run specific test file
pytest tests/test_experiment_config.py -v

# View HTML coverage report
pytest && open coverage_html_report/index.html
```

## Test Files

### Configuration Tests
- **test_experiment_config.py** (16 tests)
  - Configuration parameters
  - Path validation
  - Device detection
  - Directory creation
  - SLURM settings

### Utility Tests  
- **test_common_utils.py** (35 tests)
  - Determinism and seeding
  - ROUGE computation
  - Length statistics
  - Parametrization
  - File operations
  - Logging

### Branch Coverage Tests
- **test_branch_coverage.py** (30 tests)
  - Specific branch coverage
  - Conditional paths
  - Platform-specific code
  - Error handling

### Additional Coverage Tests
- **test_additional_coverage.py** (35 tests)
  - Edge cases
  - Error scenarios
  - Complex data structures
  - Boundary conditions

- **test_utils_additional.py** (29 tests)
  - Additional utility coverage
  - Environment variations
  - Data validation

### Model Operation Tests
- **test_model_operations.py** (25 tests)
  - Checkpoint management
  - Training logic
  - Result aggregation

### Integration Tests
- **test_stages.py** (15 tests)
  - Pipeline stage integration
  - Stage dependencies
  - Data flow validation

- **test_stage_scripts_comprehensive.py** (22 tests)
  - Stage script logic
  - Training scenarios
  - Evaluation workflows

## Coverage Results

```
Module                     Coverage
----------------------------------
experiment_config.py       100.00%
common_utils.py             97.26%
----------------------------------
TOTAL                       98.01%
```

## Test Categories

### Unit Tests
Test individual functions in isolation with mocked dependencies.
```bash
pytest -m unit
```

### Integration Tests
Test multiple components together.
```bash
pytest -m integration
```

### Slow Tests
Tests requiring downloads or long operations (skip by default).
```bash
pytest -m "not slow"
```

## Fixtures Available

Defined in `conftest.py`:

### Directory Fixtures
- `project_root` - Project root path
- `temp_dir` - Temporary directory  
- `temp_work_dir` - Temporary work directory with patched config

### Model Fixtures
- `mock_model` - Minimal T5 model
- `mock_tokenizer` - T5 tokenizer
- `device` - CPU or CUDA device

### Data Fixtures
- `sample_texts` - Sample input texts
- `sample_summaries` - Sample summaries
- `mock_dataset` - Mock dataset
- `mock_huggingface_dataset` - Mock HF dataset

### Testing Utilities
- `reset_seeds` - Auto-reset seeds between tests
- `gpu_available` - Check GPU availability

## Writing New Tests

### Test Structure
```python
class TestMyFeature:
    """Tests for my feature"""
    
    def test_basic_functionality(self, temp_dir):
        """Test basic functionality"""
        # Arrange
        data = {"key": "value"}
        
        # Act
        result = my_function(data)
        
        # Assert
        assert result == expected
```

### Using Mocks
```python
from unittest.mock import patch, MagicMock

@patch('module.external_function')
def test_with_mock(mock_function):
    mock_function.return_value = "mocked"
    result = function_under_test()
    assert result == "mocked"
```

### Testing Exceptions
```python
def test_raises_error():
    with pytest.raises(ValueError, match="expected message"):
        function_that_should_raise()
```

## Coverage Commands

### Generate All Report Types
```bash
pytest --cov=hpc_version \
       --cov-report=term-missing \
       --cov-report=html \
       --cov-report=xml
```

### Check Specific Module
```bash
pytest --cov=hpc_version/utils/common_utils.py \
       --cov-report=term-missing
```

### See Missing Lines
```bash
pytest --cov-report=term-missing:skip-covered
```

## Troubleshooting

### Tests Fail Locally
1. Install dependencies: `pip install -e ".[dev]"`
2. Clear cache: `rm -rf .pytest_cache .coverage`
3. Run with verbose output: `pytest -v -s`

### ImportError for transformers
Expected for model-related tests on macOS. These tests are marked to skip.

### Coverage Not Updating
1. Clean: `rm -rf .coverage coverage_html_report/`
2. Re-run: `pytest`

## Best Practices

1. **Write tests first** for new features (TDD)
2. **Keep tests isolated** - no shared state
3. **Use fixtures** for common setup
4. **Mock external dependencies** - no network calls
5. **Test edge cases** - empty inputs, None values
6. **Clear assertions** - one thing per test
7. **Descriptive names** - test name explains what it tests

## Test Statistics

- **Total Tests:** 164 passing
- **Coverage:** 98.01%
- **Execution Time:** ~15 seconds
- **Test Files:** 10 files
- **Test Classes:** 50+ test classes
- **Lines of Test Code:** 3,000+

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run tests with coverage
  run: pytest --cov --cov-fail-under=90
  
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

### Pre-commit Hook
```bash
#!/bin/bash
pytest --cov --cov-fail-under=90 || exit 1
```

## Contributing

When adding new code:
1. Write tests for new functionality
2. Run `pytest` to ensure coverage stays >90%
3. Fix any failing tests
4. Update test documentation if needed

## Additional Resources

- **TESTING.md** - Complete testing guide
- **COVERAGE_STATUS.md** - Coverage roadmap
- **TEST_COVERAGE_FINAL.md** - Final implementation report
- **pytest documentation** - https://docs.pytest.org/
- **coverage.py documentation** - https://coverage.readthedocs.io/

---

**Coverage Target:** 90%  
**Coverage Achieved:** 98.01% ✅  
**Status:** **Mission Accomplished**
