#!/bin/bash
################################################################################
# Make Scripts Executable
#
# Sets proper permissions on all scripts.
# Run once after downloading/cloning repository.
################################################################################

echo "Setting executable permissions..."

# Main scripts
chmod +x run_all.sh
chmod +x verify_local.py
chmod +x test_pipeline_local.py
chmod +x make_executable.sh

# Test runner
chmod +x run_tests.sh

# Python scripts
chmod +x scripts/*.py

# Job scripts
chmod +x jobs/*.sh

echo "✓ Done!"
echo ""
echo "Verify:"
ls -l run_all.sh scripts/*.py jobs/*.sh | head -5
