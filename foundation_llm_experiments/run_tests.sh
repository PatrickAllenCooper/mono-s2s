#!/bin/bash
################################################################################
# Test Runner for Foundation LLM Pipeline
#
# Runs all tests locally before HPC deployment.
################################################################################

set -euo pipefail

echo "======================================================================"
echo "  FOUNDATION LLM PIPELINE - TEST SUITE"
echo "======================================================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "ERROR: pytest not found"
    echo "Please install: pip install pytest pytest-cov"
    exit 1
fi

# Parse arguments
MODE=${1:-"all"}

case $MODE in
    "quick")
        echo "Running quick tests (unit tests only)..."
        pytest tests/test_config.py tests/test_common_utils.py -v --tb=short
        ;;
    
    "unit")
        echo "Running all unit tests..."
        pytest tests/test_config.py tests/test_common_utils.py tests/test_stage_scripts.py \
               tests/test_training_edge_cases.py tests/test_attack_mechanisms.py \
               tests/test_complete_coverage.py -v --tb=short
        ;;
    
    "integration")
        echo "Running integration tests..."
        pytest tests/test_integration.py -v
        ;;
    
    "coverage")
        echo "Running tests with coverage report..."
        pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --cov-report=json
        echo ""
        echo "Coverage report generated:"
        echo "  HTML: htmlcov/index.html"
        echo "  JSON: coverage.json"
        echo ""
        # Show coverage summary
        echo "Coverage Summary:"
        python -c "import json; data=json.load(open('coverage.json')); print(f\"  Total Coverage: {data['totals']['percent_covered']:.1f}%\")" 2>/dev/null || echo "  (Run pytest to generate coverage data)"
        ;;
    
    "all")
        echo "Running all tests..."
        pytest tests/ -v --tb=short
        ;;
    
    "verify")
        echo "Running local verification script..."
        python verify_local.py
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Usage: bash run_tests.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  quick       - Fast unit tests only (~30 sec)"
        echo "  unit        - All unit tests (~2 min)"
        echo "  integration - Integration tests (~5 min)"
        echo "  coverage    - Tests with coverage report (~3 min)"
        echo "  all         - All tests (~5 min)"
        echo "  verify      - Local verification script (~2 min)"
        echo ""
        echo "Default: all"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================================"
    echo "  ✓ ALL TESTS PASSED"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review test output above"
    echo "  2. Run verification: bash run_tests.sh verify"
    echo "  3. Submit to HPC: bash run_all.sh"
else
    echo "======================================================================"
    echo "  ✗ SOME TESTS FAILED"
    echo "======================================================================"
    echo ""
    echo "Please fix failing tests before HPC deployment."
    echo "Run with -v for more details: pytest tests/ -v"
fi

echo ""

exit $EXIT_CODE
