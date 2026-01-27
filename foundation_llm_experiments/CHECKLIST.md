# Foundation LLM Experiments - Implementation Checklist

Use this checklist to track progress on completing the experimental pipeline.

## Phase 1: Setup & Verification (Complete ✅)

- [x] Create directory structure
- [x] Write configuration file (`configs/experiment_config.py`)
- [x] Adapt utilities (`utils/common_utils.py`)
- [x] Create documentation (README, QUICKSTART, etc.)
- [x] Write stage 0 script (setup)
- [x] Write stage 1 script (apply monotonicity)
- [x] Create all SLURM job scripts
- [x] Write master submission script (`run_all.sh`)

## Phase 2: Core Implementation ✅ COMPLETE

### Stage 2-3: Training Scripts

- [x] **Stage 2: Baseline Training**
  - [x] Add Pile dataset loading
  - [x] Test with dummy data
  - [x] Test with small Pile subset (1000 samples)
  - [x] Verify checkpoint saving works
  - [x] Verify perplexity computation works

- [x] **Stage 3: Monotonic Training**
  - [x] Add Pile dataset loading (same as stage 2)
  - [x] Verify monotonic model loads correctly
  - [x] Test with dummy data
  - [x] Verify extended warmup works
  - [x] Check that weights stay non-negative

### Stage 4: Evaluation

- [x] **Implement Pile Test Evaluation**
  - [x] Load Pile test split
  - [x] Compute perplexity for baseline
  - [x] Compute perplexity for monotonic
  - [x] Save results to JSON

- [ ] **Implement Benchmark Evaluation** (Optional - not required)
  - [ ] LAMBADA accuracy
  - [ ] HellaSwag accuracy
  - [ ] Winogrande accuracy
  - [ ] TruthfulQA accuracy

### Stage 5-6: Attacks

- [x] **Stage 5: UAT Attacks**
  - [x] Implement from scratch for Pythia
  - [x] Use Pythia/GPT architecture (not T5)
  - [x] Use perplexity metrics (not ROUGE)
  - [x] Test trigger optimization logic
  - [x] Verify results format

- [x] **Stage 6: HotFlip Attacks**
  - [x] Implement from scratch for Pythia
  - [x] Use Pythia/GPT architecture (not T5)
  - [x] Use perplexity degradation metrics
  - [x] Adjust success threshold (15% vs 10%)
  - [x] Test on sample data

### Stage 7: Aggregation

- [x] **Aggregate Results**
  - [x] Load all intermediate results
  - [x] Combine into final_results.json
  - [x] Generate experiment_summary.txt
  - [x] Calculate derived metrics
  - [x] Create paper-ready output format

## Phase 3: Testing ✅ COMPLETE

### Quick Mode Testing

- [x] Set `USE_FULL_EVAL_SETS = False` in config (default)
- [x] Set appropriate `TRAINING_SAMPLES` in config
- [x] Local pipeline test completes successfully
- [x] All 8 stages validated with tiny model
- [x] Output files structure verified
- [x] Logs format validated

### Validation Testing

- [x] Perplexity computation verified (exp(loss) relationship)
- [x] Attack success rates tested
- [x] JSON files validated with schema
- [x] Checkpoint files save/load tested
- [x] Resume from checkpoint tested

### Integration Testing

- [x] Stages 0-1 tested independently
- [x] Stages 2-3 training logic validated
- [x] Stage 4 evaluation framework tested
- [x] Stages 5-6 attack logic validated
- [x] Stage 7 aggregation tested
- [x] final_results.json structure verified

## Phase 4: Full Execution (TODO)

### Pre-Execution Checks

- [ ] Set `USE_FULL_EVAL_SETS = True`
- [ ] Set `TRAINING_SAMPLES = None` (use full Pile)
- [ ] Verify storage space available (~500GB)
- [ ] Verify GPU hours available (~70 hours)
- [ ] Review time limits in job scripts

### Execution

- [ ] Run `bash run_all.sh`
- [ ] Record job IDs
- [ ] Monitor first few jobs (0-3)
- [ ] Check logs for errors
- [ ] Verify checkpoints are being saved

### During Execution

- [ ] Monitor job queue: `squeue -u $USER`
- [ ] Check logs periodically: `tail -f logs/job_2_baseline_*.out`
- [ ] Verify disk space: `du -sh $SCRATCH/foundation_llm_work`
- [ ] Watch for OOM or timeout errors

### Post-Execution

- [ ] Verify all 8 completion flags exist
- [ ] Check `final_results.json` exists
- [ ] Review `experiment_summary.txt`
- [ ] Verify perplexity values are reasonable
- [ ] Check attack results make sense

## Phase 5: Multi-Seed (Optional)

If running multiple seeds:

- [ ] Run seed 42 first, verify results
- [ ] Adjust hyperparameters if needed
- [ ] Submit seeds in parallel:
  - [ ] `EXPERIMENT_SEED=42 bash run_all.sh`
  - [ ] `EXPERIMENT_SEED=1337 bash run_all.sh`
  - [ ] `EXPERIMENT_SEED=2024 bash run_all.sh`
  - [ ] `EXPERIMENT_SEED=8888 bash run_all.sh`
  - [ ] `EXPERIMENT_SEED=12345 bash run_all.sh`
- [ ] Wait for all to complete
- [ ] Run multi-seed aggregation script
- [ ] Compute mean ± std across seeds

## Phase 6: Paper Update (TODO)

### Extract Results

- [ ] Extract baseline perplexity:
  ```bash
  jq '.pile_test.baseline_pythia.perplexity' \
    $SCRATCH/foundation_llm_results/evaluation_results.json
  ```

- [ ] Extract monotonic perplexity:
  ```bash
  jq '.pile_test.monotonic_pythia.perplexity' \
    $SCRATCH/foundation_llm_results/evaluation_results.json
  ```

- [ ] Calculate perplexity gap:
  ```
  Δ% = ((monotonic_ppl - baseline_ppl) / baseline_ppl) * 100
  ```

- [ ] Extract HotFlip success rates:
  ```bash
  jq '.results.baseline_pythia.success_rate' \
    $SCRATCH/foundation_llm_results/hotflip_results.json
  jq '.results.monotonic_pythia.success_rate' \
    $SCRATCH/foundation_llm_results/hotflip_results.json
  ```

### Update Paper

- [ ] Open `../documentation/monotone_llms_paper.tex`
- [ ] Find Table 7 (around line 672)
- [ ] Add Pythia-1.4B rows with actual values
- [ ] Remove `\textcolor{red}{...}` from all values
- [ ] Update line 656 with actual observations
- [ ] Update line 661 with actual patterns
- [ ] Add footnote about Pythia vs T5 metric differences
- [ ] Compile LaTeX, verify no errors

### Verify Paper Update

- [ ] All red text removed from Section 4.3
- [ ] Table 7 contains Pythia results
- [ ] Perplexity values are reasonable
- [ ] Attack reduction is significant
- [ ] Numbers match across paper
- [ ] Citations updated if needed

## Phase 7: Documentation (TODO)

### Update This Pipeline

- [ ] Mark completed items in `IMPLEMENTATION_GUIDE.md`
- [ ] Document any issues encountered
- [ ] Update `README.md` with actual results
- [ ] Add troubleshooting notes
- [ ] Update `PAPER_INTEGRATION.md` with actual values used

### Update Main Project

- [ ] Add note in `../README.md` pointing to this directory
- [ ] Update `../DOCUMENTATION_INDEX.md` to include foundation experiments
- [ ] Cross-reference in `../PROJECT_STATUS_SUMMARY.md`

## Common Roadblocks & Solutions

### Roadblock 1: Pile Dataset Won't Download

**Symptoms**: Timeout, 403 errors, "dataset not found"

**Solutions**:
- Use validation split instead of train
- Use smaller dataset (C4, OpenWebText)
- Cache dataset to $PROJECT first
- Use streaming mode

### Roadblock 2: OOM During Training

**Symptoms**: Job killed with exit code 137

**Solutions**:
- Enable gradient checkpointing
- Reduce batch size to 4 or 2
- Increase gradient accumulation steps
- Use mixed precision (fp16)

### Roadblock 3: Training Takes Too Long

**Symptoms**: Jobs timeout at 24/32 hours

**Solutions**:
- Reduce TRAINING_SAMPLES
- Use validation split instead of full train
- Request longer time limits
- Implement checkpoint/resume (already done)

### Roadblock 4: Attacks Too Weak

**Symptoms**: UAT/HotFlip success rate < 10%

**Solutions**:
- Increase number of iterations
- Use different candidate vocabulary
- Lower success threshold
- Document as finding (like weak UAT for T5)

### Roadblock 5: Results Don't Match T5 Pattern

**Symptoms**: Perplexity gap > 20% or attack reduction < 30%

**Solutions**:
- Check monotonicity was applied correctly
- Verify recovery training converged
- Review hyperparameters (warmup, LR)
- Consider that Pythia may behave differently (document)

## Success Indicators

### During Implementation

- ✅ All scripts run without syntax errors
- ✅ Dummy data tests pass
- ✅ Checkpoint files are created
- ✅ JSON outputs are well-formatted

### During Execution

- ✅ Jobs don't fail immediately
- ✅ Loss decreases during training
- ✅ Checkpoints are saved each epoch
- ✅ Perplexity values are in reasonable range (8-20)

### After Completion

- ✅ All 8 completion flags exist
- ✅ final_results.json contains all sections
- ✅ Perplexity gap is < 15%
- ✅ Attack reduction is > 40%
- ✅ Results are reproducible across runs

## Final Sign-Off

Before considering the pipeline complete:

- [ ] All 8 stages implemented and tested
- [ ] Full run completed successfully (at least seed 42)
- [ ] Results extracted and validated
- [ ] Paper updated with actual values
- [ ] Red text removed from Section 4.3
- [ ] Documentation updated with lessons learned
- [ ] Code committed to repository

---

**Current Status**: Phase 1 complete, ready for Phase 2
**Next Action**: Implement Pile data loading in stages 2-3
**Estimated Time to Completion**: 6-9 hours implementation + 60-70 hours execution
