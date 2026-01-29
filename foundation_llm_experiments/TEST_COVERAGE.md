# Foundation LLM Experiments - Test Coverage

**Last Updated:** 2026-01-29  
**Python Test Coverage:** High (new checkpoint_manager.py: 100%)  
**Bash Script Coverage:** Manual testing required

---

## Python Code Testing

### Fully Tested Modules

#### ✓ `utils/checkpoint_manager.py` (22 tests, 100% coverage)

**CheckpointManager Class:**
- [x] Initialization with all parameters
- [x] Step-based checkpoint triggering (every N steps)
- [x] Time-based checkpoint triggering (every N minutes)
- [x] Checkpoint saving with metadata
- [x] Automatic cleanup (keeping last N checkpoints)
- [x] Loading latest checkpoint
- [x] Handling empty checkpoint directory
- [x] Saving best model separately
- [x] Best model exclusion from cleanup
- [x] Resume information extraction

**TrainingTimer Class:**
- [x] Initialization and configuration
- [x] Elapsed time calculation
- [x] Remaining time calculation (with/without limit)
- [x] Timeout detection and warnings
- [x] Status string generation

**Integration Tests:**
- [x] Full save-and-resume workflow
- [x] Multiple checkpoints with cleanup
- [x] Timeout scenario simulation (save → resubmit → resume)

### Existing Test Coverage

#### ✓ `utils/common_utils.py`
- Covered by `tests/test_common_utils.py`
- Includes: seeding, file I/O, logging, data utilities

#### ✓ Stage Scripts
- Covered by `tests/test_stage_scripts.py`
- Tests script imports and basic functionality

#### ✓ Configuration
- Covered by `tests/test_config.py`
- Validates all configuration parameters

#### ✓ Attack Mechanisms
- Covered by `tests/test_attack_mechanisms.py`
- UAT and HotFlip attack logic

#### ✓ Training Edge Cases
- Covered by `tests/test_training_edge_cases.py`
- Handles various training failure scenarios

---

## Bash Script Coverage

**Note:** Bash scripts are difficult to unit test with pytest. They require manual testing or specialized bash testing frameworks (like bats-core).

### Scripts Requiring Manual Testing

#### `bootstrap_curc.sh`
**Purpose:** Automatic conda installation and environment setup

**Manual Test Procedure:**
1. Test on fresh Alpine login node (no conda installed)
   ```bash
   bash bootstrap_curc.sh
   ```
   **Expected:** Conda installed to `/projects/$USER/miniconda3`, environment created, all dependencies installed

2. Test with existing conda
   ```bash
   bash bootstrap_curc.sh
   ```
   **Expected:** Detects existing installation, skips conda install, creates/updates environment

3. Test with existing environment
   ```bash
   bash bootstrap_curc.sh
   ```
   **Expected:** Detects existing environment, confirms PyTorch installed, skips setup

**Test Checklist:**
- [ ] Installs conda to correct location
- [ ] Updates .bashrc correctly
- [ ] Creates mono_s2s environment
- [ ] Installs PyTorch with CUDA
- [ ] Installs all requirements.txt dependencies
- [ ] Makes scripts executable
- [ ] Sets up HuggingFace cache locations
- [ ] Validates configuration
- [ ] Idempotent (safe to run multiple times)

#### `monitor_and_resubmit.sh`
**Purpose:** Monitors jobs and auto-resubmits on timeout

**Manual Test Procedure:**
1. Submit a test job with very short time limit
   ```bash
   # Create test job that times out quickly
   cat > test_timeout.sh <<'EOF'
   #!/bin/bash
   #SBATCH --time=00:02:00
   #SBATCH --job-name=test_timeout
   sleep 300  # Sleep longer than time limit
   EOF
   
   sbatch test_timeout.sh
   # Get job ID, then:
   ./monitor_and_resubmit.sh <JOB_ID>
   ```
   **Expected:** Detects timeout, logs it, resubmits job

2. Test with completed job
   ```bash
   # Submit job that completes normally
   ./monitor_and_resubmit.sh <COMPLETED_JOB_ID>
   ```
   **Expected:** Detects completion, stops monitoring

3. Test with failed job
   ```bash
   # Submit job that fails
   ./monitor_and_resubmit.sh <FAILED_JOB_ID>
   ```
   **Expected:** Detects failure, decides whether to resubmit based on failure type

**Test Checklist:**
- [ ] Detects TIMEOUT state correctly
- [ ] Detects COMPLETED state correctly
- [ ] Detects FAILED state correctly
- [ ] Detects PREEMPTED state correctly
- [ ] Resubmits timed-out jobs
- [ ] Respects MAX_RESUBMISSIONS limit
- [ ] Checks for checkpoints before resubmit
- [ ] Logs all actions to log file
- [ ] Updates job tracking correctly
- [ ] Handles multiple jobs simultaneously

#### `run_all.sh`
**Purpose:** Main pipeline submission with automatic setup

**Manual Test Procedure:**
1. Test first-time setup
   ```bash
   # On system with no conda/environment
   ./run_all.sh
   ```
   **Expected:** Prompts for setup, runs bootstrap, submits all jobs

2. Test with existing environment
   ```bash
   # With environment already set up
   ./run_all.sh
   ```
   **Expected:** Skips setup, directly submits jobs

3. Test monitoring option
   ```bash
   ./run_all.sh
   # Answer 'y' to monitoring
   ```
   **Expected:** Starts monitor_and_resubmit.sh in background, saves PID

**Test Checklist:**
- [ ] Detects conda installation
- [ ] Detects conda environment
- [ ] Runs bootstrap if needed
- [ ] Submits all 7 stages
- [ ] Sets up job dependencies correctly
- [ ] Saves job IDs to .job_ids
- [ ] Optionally starts monitoring
- [ ] Provides correct monitoring commands
- [ ] Handles user cancellation gracefully

---

## SLURM Job Scripts

**Note:** Job scripts can be tested by submitting to SLURM queue.

### Testing Job Scripts

#### Quick Test (No GPU Required)
```bash
# Test job script syntax and environment
cd foundation_llm_experiments

# Dry run - check for syntax errors
bash -n jobs/job_0_setup.sh
bash -n jobs/job_2_baseline.sh
bash -n jobs/job_3_monotonic.sh

# Test conda activation (on login node)
source /projects/$USER/miniconda3/etc/profile.d/conda.sh
conda activate mono_s2s
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

#### Full Test (Requires GPU Allocation)
```bash
# Submit actual jobs to test
sbatch jobs/job_0_setup.sh
# Monitor:
squeue -u $USER
tail -f logs/job_0_setup_*.out
```

**Test Checklist for Each Job:**
- [ ] Conda environment activates correctly
- [ ] All environment variables set correctly
- [ ] HuggingFace cache location correct
- [ ] GPU detected (for GPU jobs)
- [ ] Python script runs without import errors
- [ ] Checkpoint saving works
- [ ] Resume from checkpoint works
- [ ] Time limit respected (with buffer)
- [ ] Signal handler works (SIGUSR1)

---

## Coverage Summary

### Python Modules

| Module | Test File | Coverage | Notes |
|--------|-----------|----------|-------|
| `utils/checkpoint_manager.py` | `test_checkpoint_manager.py` | 100% | ✓ Fully tested (22 tests) |
| `utils/common_utils.py` | `test_common_utils.py` | High | ✓ Existing coverage |
| `configs/experiment_config.py` | `test_config.py` | High | ✓ Existing coverage |
| `scripts/stage_*.py` | `test_stage_scripts.py` | Medium | Partial coverage |

### Bash Scripts

| Script | Testing Method | Status |
|--------|----------------|--------|
| `bootstrap_curc.sh` | Manual | Requires manual test on Alpine |
| `monitor_and_resubmit.sh` | Manual | Requires SLURM environment |
| `run_all.sh` | Manual | Requires SLURM environment |
| `jobs/job_*.sh` | Manual/SLURM | Submit and monitor |

---

## Running Tests

### Python Tests

```bash
cd foundation_llm_experiments

# Run all tests
pytest

# Run specific test file
pytest tests/test_checkpoint_manager.py -v

# Run with coverage report
pytest --cov=utils/checkpoint_manager --cov-report=term-missing

# Run only checkpoint manager tests
pytest tests/test_checkpoint_manager.py -v --tb=short
```

### Expected Output

```
============================= test session starts ==============================
collected 22 items

tests/test_checkpoint_manager.py::TestCheckpointManager::test_initialization PASSED
tests/test_checkpoint_manager.py::TestCheckpointManager::test_should_save_step_based PASSED
tests/test_checkpoint_manager.py::TestCheckpointManager::test_should_save_time_based PASSED
[... 19 more tests ...]
============================== 22 passed in 3.13s ===============================
```

### Manual Testing

For bash scripts and SLURM integration:

1. **Test bootstrap on Alpine:**
   ```bash
   ssh user@login.rc.colorado.edu
   cd /projects/$USER/mono-s2s/foundation_llm_experiments
   bash bootstrap_curc.sh
   ```

2. **Test job submission:**
   ```bash
   ./run_all.sh
   # Monitor job execution
   squeue -u $USER
   tail -f logs/job_*.out
   ```

3. **Test auto-resubmission:**
   ```bash
   # Create short-timeout test job
   # Monitor with: ./monitor_and_resubmit.sh <JOB_ID>
   # Verify it resubmits on timeout
   ```

---

## Limitations and Gaps

### Cannot Easily Test with Pytest

1. **SLURM Integration**
   - Job submission (`sbatch`)
   - Job monitoring (`squeue`, `sacct`)
   - Signal handling (`SIGUSR1`)
   - Requires actual SLURM cluster

2. **Conda Environment Setup**
   - Conda installation
   - Environment creation
   - Package installation
   - Requires clean environment

3. **File System Paths**
   - Alpine-specific paths (`/projects`, `/scratch`)
   - HuggingFace cache setup
   - Requires actual HPC environment

### Mitigation Strategies

1. **Manual Testing Protocols**
   - Documented test procedures (above)
   - Test checklists for each component
   - Record test results

2. **Defensive Programming**
   - Extensive error checking in bash scripts
   - Clear error messages
   - Logging of all actions

3. **Idempotent Design**
   - Scripts safe to run multiple times
   - Detect and skip completed steps
   - Resume from checkpoints

---

## Test Maintenance

### When Adding New Features

1. **Python Code:**
   - Add tests to appropriate `test_*.py` file
   - Aim for >90% coverage
   - Include edge cases and error conditions

2. **Bash Scripts:**
   - Update manual test procedures
   - Add to test checklist
   - Document expected behavior

3. **Configuration Changes:**
   - Update `test_config.py`
   - Validate all new parameters

### Before Release

1. Run full Python test suite
2. Execute manual test procedures on Alpine
3. Submit test jobs to SLURM
4. Verify checkpoint and resume functionality
5. Test auto-resubmission with short timeout

---

## Conclusion

**Python Code:** Comprehensively tested with pytest (22 new tests for checkpoint_manager)

**Bash Scripts:** Require manual testing on SLURM cluster

**Overall Strategy:** Combination of automated testing (Python) and documented manual procedures (Bash/SLURM)

**Confidence Level:** High for Python code, Medium for Bash (pending manual validation on Alpine)

---

**Last Test Run:** 2026-01-29  
**Test Results:** 22/22 checkpoint_manager tests passing  
**Manual Testing:** Pending on CURC Alpine
