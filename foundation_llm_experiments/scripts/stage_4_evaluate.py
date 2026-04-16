#!/usr/bin/env python3
"""
Stage 4: Evaluate Models on LLM Benchmarks

Evaluates both baseline and monotonic models on:
- Pile test set (perplexity)

Hardened against spot-instance deallocation:
- Per-model partial results are persisted atomically to the results disk
  immediately after each model finishes.
- Within a single model, perplexity accumulation is checkpointed every N
  batches to `partial/{model}_pile_progress.json`; a restarted run skips
  already-scored batches and continues from the last durable position.
- Aggregation and completion flag are only written once both partials
  exist. Re-running this stage after a crash is idempotent.

Inputs:
- baseline_checkpoints/best_model.pt
- monotonic_checkpoints/best_model.pt

Outputs:
- evaluation_results.json
- stage_4_evaluate_complete.flag
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, save_json, load_json, atomic_save_json,
    StageLogger, check_dependencies, compute_perplexity_resumable,
    make_model_monotonic, partial_results_dir,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_pile_test(logger, max_samples):
    """Load Pile test samples, preferring test/validation splits."""
    from datasets import load_dataset

    for split in ("test", "validation"):
        try:
            ds = load_dataset(
                Config.TRAINING_DATASET,
                split=split,
                streaming=False,
                cache_dir=Config.DATA_CACHE_DIR,
                trust_remote_code=True,
            )
            logger.log(f"  Loaded '{split}' split.")
            return ds
        except (ValueError, KeyError):
            continue

    slice_size = max_samples or 50000
    logger.log(f"  No test/validation split; taking last {slice_size} train rows.")
    return load_dataset(
        Config.TRAINING_DATASET,
        split=f"train[-{slice_size}:]",
        streaming=False,
        cache_dir=Config.DATA_CACHE_DIR,
        trust_remote_code=True,
    )


def _run_pile_perplexity(model, tokenizer, device, logger, progress_path, max_samples):
    """Compute pile perplexity with resumable per-batch checkpointing."""
    from torch.utils.data import DataLoader
    from utils.common_utils import LanguageModelingDataset

    pile_test = _load_pile_test(logger, max_samples)

    if max_samples:
        test_texts = [ex['text'] for i, ex in enumerate(pile_test) if i < max_samples]
    else:
        test_texts = [ex['text'] for ex in pile_test]

    dataset = LanguageModelingDataset(test_texts, tokenizer, max_length=Config.MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=Config.EVAL_BATCH_SIZE, shuffle=False)

    return compute_perplexity_resumable(
        model, dataloader, device,
        progress_path=progress_path,
        flush_every=10,
        log_fn=logger.log,
    )


def _load_baseline(device):
    """Materialize the baseline model from the recovery checkpoint."""
    path = os.path.join(Config.CHECKPOINT_DIR, 'baseline_checkpoints', 'best_model.pt')
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.DATA_CACHE_DIR,
        torch_dtype=torch.float32,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model


def _load_monotonic(device):
    """Materialize the monotonic-constrained model from its checkpoint."""
    path = os.path.join(Config.CHECKPOINT_DIR, 'monotonic_checkpoints', 'best_model.pt')
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.DATA_CACHE_DIR,
        torch_dtype=torch.float32,
    )
    model = make_model_monotonic(model)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    return model


def _eval_one_model(name, loader_fn, tokenizer, device, logger, max_samples):
    """Evaluate one model, reusing its persisted partial result if present."""
    partial = partial_results_dir()
    result_path = os.path.join(partial, f"{name}_pile.json")
    progress_path = os.path.join(partial, f"{name}_pile_progress.json")

    if os.path.exists(result_path):
        logger.log(f"[{name}] Using cached pile result from {result_path}")
        return load_json(result_path)

    logger.log(f"\n{'='*80}\nEVALUATING: {name}\n{'='*80}")
    logger.log(f"[{name}] Loading model...")
    model = loader_fn(device)

    logger.log(f"[{name}] Computing perplexity on Pile test set...")
    result = _run_pile_perplexity(
        model, tokenizer, device, logger,
        progress_path=progress_path,
        max_samples=max_samples,
    )
    logger.log(f"[{name}] Perplexity: {result['perplexity']:.2f}")

    atomic_save_json(result, result_path)

    try:
        if os.path.exists(progress_path):
            os.remove(progress_path)
    except OSError:
        pass

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    logger = StageLogger("stage_4_evaluate")

    try:
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic']):
            logger.complete(success=False)
            return 1

        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()

        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_samples = (
            Config.FULL_PILE_TEST_SIZE if Config.USE_FULL_EVAL_SETS
            else Config.QUICK_PILE_TEST_SIZE
        )

        baseline_pile = _eval_one_model(
            "baseline_pythia", _load_baseline, tokenizer, device, logger, max_samples
        )
        monotonic_pile = _eval_one_model(
            "monotonic_pythia", _load_monotonic, tokenizer, device, logger, max_samples
        )

        results = {
            'pile_test': {
                'baseline_pythia': baseline_pile,
                'monotonic_pythia': monotonic_pile,
            },
            'metadata': {
                'seed': Config.CURRENT_SEED,
                'model_name': Config.MODEL_NAME,
                'use_full_eval_sets': Config.USE_FULL_EVAL_SETS,
                'max_pile_samples': max_samples,
            },
        }

        save_json(results, os.path.join(Config.RESULTS_DIR, 'evaluation_results.json'))

        logger.log("\nEvaluation complete.")
        logger.complete(success=True)
        return 0

    except Exception as e:
        logger.log(f"\nERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())
