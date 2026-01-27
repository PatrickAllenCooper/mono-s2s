#!/bin/bash
################################################################################
# Verify Experimental Results Are Properly Organized
#
# Checks that all experimental results are properly tracked and version controlled.
# Run this before paper submission to ensure everything is in order.
################################################################################

echo "======================================================================"
echo "  VERIFYING EXPERIMENTAL RESULTS ORGANIZATION"
echo "======================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Experiment results directory exists
echo "1. Checking experiment_results/ directory..."
if [ -d "experiment_results" ]; then
    echo "  ✓ experiment_results/ exists"
else
    echo "  ✗ experiment_results/ missing"
    echo "     Create with: mkdir -p experiment_results/{t5_experiments,foundation_llm_experiments}"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Experiment index exists
echo ""
echo "2. Checking experiment index..."
if [ -f "experiment_results/experiment_index.json" ]; then
    echo "  ✓ experiment_index.json exists"
    
    # Validate JSON
    if jq empty experiment_results/experiment_index.json 2>/dev/null; then
        NUM_EXPERIMENTS=$(jq '.experiments | length' experiment_results/experiment_index.json)
        echo "    Experiments indexed: $NUM_EXPERIMENTS"
    else
        echo "  ⚠️  experiment_index.json is invalid JSON"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ⚠️  experiment_index.json missing (will be created on first commit)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 3: Paper evidence directory
echo ""
echo "3. Checking paper_evidence/ directory..."
if [ -d "paper_evidence" ]; then
    echo "  ✓ paper_evidence/ exists"
    
    if [ -f "paper_evidence/provenance.json" ]; then
        echo "  ✓ provenance.json exists"
        
        # Check provenance is valid
        if jq empty paper_evidence/provenance.json 2>/dev/null; then
            NUM_TABLES=$(jq '.tables | length' paper_evidence/provenance.json 2>/dev/null || echo "0")
            echo "    Paper tables linked: $NUM_TABLES"
        else
            echo "  ⚠️  provenance.json is invalid JSON"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "  ⚠️  provenance.json missing"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ⚠️  paper_evidence/ missing"
    echo "     Create with: mkdir -p paper_evidence"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 4: Git tracking
echo ""
echo "4. Checking git tracking..."

# Check .gitignore exists and has proper rules
if [ -f ".gitignore" ]; then
    echo "  ✓ .gitignore exists"
    
    # Verify JSON files are tracked
    if grep -q "experiment_results/\*\*/\*.json" .gitignore 2>/dev/null; then
        echo "  ⚠️  Warning: .gitignore might be ignoring result files"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  ✓ Result files should be tracked"
    fi
    
    # Verify checkpoints are ignored
    if grep -q "\*.pt" .gitignore; then
        echo "  ✓ Checkpoints properly ignored"
    else
        echo "  ⚠️  Checkpoints might be tracked (not recommended)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ .gitignore missing"
    ERRORS=$((ERRORS + 1))
fi

# Check 5: Experiment completeness
echo ""
echo "5. Checking experiment completeness..."

for seed_dir in experiment_results/*/seed_*; do
    if [ -d "$seed_dir" ]; then
        SEED_NAME=$(basename "$seed_dir")
        
        # Check required files
        MISSING_FILES=()
        for required_file in metadata.json final_results.json; do
            if [ ! -f "$seed_dir/$required_file" ]; then
                MISSING_FILES+=("$required_file")
            fi
        done
        
        if [ ${#MISSING_FILES[@]} -eq 0 ]; then
            echo "  ✓ $seed_dir complete"
        else
            echo "  ⚠️  $seed_dir missing: ${MISSING_FILES[*]}"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
done

# Check 6: Tracking scripts exist
echo ""
echo "6. Checking tracking scripts..."

REQUIRED_SCRIPTS=(
    "scripts/organize_results.py"
    "scripts/commit_experiment_results.sh"
    "scripts/update_experiment_index.py"
    "scripts/link_results_to_paper.py"
    "scripts/snapshot_running_experiments.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Summary
echo ""
echo "======================================================================"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "  ✓ ALL CHECKS PASSED"
    echo "======================================================================"
    echo ""
    echo "Experimental results are properly organized and tracked."
    echo ""
elif [ $ERRORS -eq 0 ]; then
    echo "  ✓ PASSED WITH WARNINGS"
    echo "======================================================================"
    echo ""
    echo "Warnings: $WARNINGS"
    echo "Review warnings above - these are not critical but should be addressed."
    echo ""
else
    echo "  ✗ VERIFICATION FAILED"
    echo "======================================================================"
    echo ""
    echo "Errors: $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Fix errors above before relying on results tracking."
    echo ""
fi

exit $ERRORS
