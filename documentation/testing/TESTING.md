# Testing Documentation

## Overview

This project implements comprehensive test coverage using pytest with a target of 90% code coverage.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── test_experiment_config.py      # Configuration tests
├── test_common_utils.py           # Utility functions tests
└── test_stages.py                 # Pipeline stage tests
```

## Running Tests

### Run all tests with coverage
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_experiment_config.py
```

### Run with verbose output
```bash
pytest -v
```

### Run without coverage (faster)
```bash
pytest --no-cov
```

### Generate HTML coverage report
```bash
pytest
# Then open: coverage_html_report/index.html
```

### Run only fast tests (skip slow model-loading tests)
```bash
pytest -m "not slow"
```

## Test Categories

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Fast execution
- Tagged with `@pytest.mark.unit`

### Integration Tests  
- Test multiple components together
- May use real dependencies
- Slower execution
- Tagged with `@pytest.mark.integration`

### Slow Tests
- Require model downloads
- Long-running operations
- Skipped by default in CI
- Tagged with `@pytest.mark.slow`

## Coverage Requirements

- Target: 90% overall coverage
- Branch coverage enabled
- HTML and XML reports generated
- Coverage failure stops CI/CD

## Current Coverage

Run `pytest` to see current coverage report. Key modules:

- `hpc_version/configs/experiment_config.py`: Configuration management
- `hpc_version/utils/common_utils.py`: Core utility functions
- `hpc_version/scripts/stage_*.py`: Pipeline stages

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Using Fixtures
```python
def test_my_function(temp_dir, monkeypatch):
    # temp_dir provides a temporary directory
    # monkeypatch allows modifying objects/attributes
    pass
```

### Mocking External Dependencies
```python
from unittest.mock import patch, MagicMock

@patch('datasets.load_dataset')
def test_dataset_loading(mock_load_dataset):
    mock_load_dataset.return_value = MockDataset()
    # test code here
```

## Common Fixtures

Defined in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `temp_work_dir`: Temporary work directory with patched config
- `mock_model`: Minimal T5 model for testing
- `mock_tokenizer`: T5 tokenizer for testing
- `sample_texts`: Sample text data
- `sample_summaries`: Sample summary data
- `device`: Appropriate device (CPU/CUDA) for testing

## Continuous Integration

Tests run automatically on:
- Every commit
- Pull requests
- Pre-merge validation

CI will fail if:
- Any test fails
- Coverage drops below 90%
- New code is not tested

## Troubleshooting

### Test fails locally
1. Check dependencies: `pip install -e ".[dev]"`
2. Clear pytest cache: `rm -rf .pytest_cache`
3. Run with verbose output: `pytest -v -s`

### Coverage not updating
1. Clean coverage data: `rm -rf .coverage htmlcov/`
2. Re-run tests: `pytest`

### Import errors
1. Ensure project is installed: `pip install -e .`
2. Check PYTHONPATH includes project root

## Best Practices

1. **Isolate tests**: Each test should be independent
2. **Use fixtures**: Reuse common setup code
3. **Mock external calls**: Don't hit real APIs/networks
4. **Test edge cases**: Empty inputs, None values, errors
5. **Keep tests fast**: Use mocks for slow operations
6. **Clear assertions**: Test one thing at a time
7. **Descriptive names**: Test name should describe what it tests

## Coverage Goals by Module

- Configuration: 95%+ (critical, rarely changes)
- Utilities: 90%+ (shared code, high reuse)
- Stage scripts: 85%+ (complex, many paths)
- Integration: 80%+ (harder to test all paths)
