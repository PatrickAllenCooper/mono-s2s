#!/bin/bash
#
# Validation Script for Mono-S2S HPC Setup
#
# This script checks your configuration and environment before running experiments.
# Run this on a login node before submitting jobs.
#
# Usage: ./validate_setup.sh
#

echo "=========================================="
echo "Mono-S2S HPC Setup Validation"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ((ERRORS++))
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# Check 1: Python version
echo "Checking Python installation..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_status 0 "Python found: $PYTHON_VERSION"
    
    # Check if version is >= 3.8
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_status 0 "Python version is compatible (>= 3.8)"
    else
        print_status 1 "Python version should be >= 3.8, found $PYTHON_VERSION"
    fi
else
    print_status 1 "Python not found in PATH"
fi
echo ""

# Check 2: Required Python packages
echo "Checking Python packages..."
REQUIRED_PACKAGES=("torch" "transformers" "datasets" "rouge_score" "scipy" "pandas" "tqdm")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        print_status 0 "$package installed ($VERSION)"
    else
        print_status 1 "$package not installed"
    fi
done
echo ""

# Check 3: CUDA availability
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    print_status 0 "nvidia-smi found"
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        print_status 0 "GPUs detected: $GPU_COUNT"
    else
        print_warning "No GPUs detected (may be on login node)"
    fi
else
    print_warning "nvidia-smi not found (may be on login node)"
fi
echo ""

# Check 4: Environment variables
echo "Checking environment variables..."
if [ -n "$SCRATCH" ]; then
    print_status 0 "SCRATCH is set: $SCRATCH"
    if [ -d "$SCRATCH" ]; then
        print_status 0 "SCRATCH directory exists"
    else
        print_status 1 "SCRATCH directory does not exist: $SCRATCH"
    fi
else
    print_warning "SCRATCH not set (will use default from config)"
fi

if [ -n "$PROJECT" ]; then
    print_status 0 "PROJECT is set: $PROJECT"
    if [ -d "$PROJECT" ]; then
        print_status 0 "PROJECT directory exists"
    else
        print_status 1 "PROJECT directory does not exist: $PROJECT"
    fi
else
    print_warning "PROJECT not set (will use default from config)"
fi
echo ""

# Check 5: SLURM availability
echo "Checking SLURM..."
if command -v sbatch &> /dev/null; then
    print_status 0 "sbatch found"
    print_status 0 "squeue found" 
else
    print_status 1 "SLURM commands not found"
fi

# Check user's SLURM associations
if command -v sacctmgr &> /dev/null; then
    echo ""
    echo "Your SLURM associations:"
    sacctmgr show assoc where user=$USER format=account,partition,qos%50 -n | head -5
else
    print_warning "sacctmgr not available, cannot check associations"
fi
echo ""

# Check 6: File structure
echo "Checking file structure..."
REQUIRED_DIRS=("configs" "scripts" "jobs" "utils")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_status 0 "Directory exists: $dir/"
    else
        print_status 1 "Directory missing: $dir/"
    fi
done

REQUIRED_FILES=(
    "configs/experiment_config.py"
    "utils/common_utils.py"
    "scripts/stage_0_setup.py"
    "scripts/stage_1_prepare_data.py"
    "scripts/stage_2_train_baseline.py"
    "scripts/stage_3_train_monotonic.py"
    "scripts/stage_4_evaluate.py"
    "scripts/stage_5_uat_attacks.py"
    "scripts/stage_6_hotflip_attacks.py"
    "scripts/stage_7_aggregate.py"
    "jobs/job_0_setup.sh"
    "jobs/job_1_data.sh"
    "jobs/job_2_baseline.sh"
    "jobs/job_3_monotonic.sh"
    "jobs/job_4_evaluate.sh"
    "jobs/job_5_uat.sh"
    "jobs/job_6_hotflip.sh"
    "jobs/job_7_aggregate.sh"
    "run_all.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_status 0 "File exists: $file"
    else
        print_status 1 "File missing: $file"
    fi
done
echo ""

# Check 7: Configuration
echo "Checking experiment_config.py..."
CONFIG_FILE="configs/experiment_config.py"

# Check for placeholder values
if grep -q "your_username" "$CONFIG_FILE" 2>/dev/null; then
    print_warning "Configuration contains 'your_username' - please edit"
fi

if grep -q "your_project" "$CONFIG_FILE" 2>/dev/null; then
    print_warning "Configuration contains 'your_project' - please edit"
fi

# Check partition setting
PARTITION=$(grep "SLURM_PARTITION" "$CONFIG_FILE" | grep -v "#" | head -1 | cut -d'"' -f2)
if [ -n "$PARTITION" ]; then
    echo "  Configured partition: $PARTITION"
    if [ "$PARTITION" = "gpu" ] || [ "$PARTITION" = "shas" ]; then
        print_status 0 "Partition looks reasonable for CURC"
    else
        print_warning "Verify partition '$PARTITION' is correct for your cluster"
    fi
fi
echo ""

# Check 8: File permissions
echo "Checking file permissions..."
if [ -x "run_all.sh" ]; then
    print_status 0 "run_all.sh is executable"
else
    print_warning "run_all.sh is not executable (run: chmod +x run_all.sh)"
fi

EXECUTABLE_COUNT=$(find jobs/ -name "*.sh" -executable 2>/dev/null | wc -l)
TOTAL_JOB_COUNT=$(find jobs/ -name "*.sh" 2>/dev/null | wc -l)
if [ $EXECUTABLE_COUNT -eq $TOTAL_JOB_COUNT ]; then
    print_status 0 "All job scripts are executable"
else
    print_warning "Some job scripts are not executable (run: chmod +x jobs/*.sh)"
fi
echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo "Errors:   $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ No critical errors found!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Edit configs/experiment_config.py with your paths and partition"
    echo "  2. Make scripts executable: chmod +x run_all.sh jobs/*.sh scripts/*.py"
    echo "  3. Review QUICKSTART.md for detailed instructions"
    echo "  4. Submit jobs: ./run_all.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS critical error(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo "See QUICKSTART.md for help."
    echo ""
    exit 1
fi

