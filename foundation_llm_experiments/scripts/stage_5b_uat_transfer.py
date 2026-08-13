#!/usr/bin/env python3
"""
Stage 5b: UAT Trigger Transfer Matrix for Foundation LLMs

Replays the universal triggers already learned in Stage 5 (one trigger per
model, optimized against that model's own loss) against every model, to
build the 2x2 source-trigger x target-model NLL-increase transfer matrix.
No new trigger optimization happens here -- this stage only evaluates.

Motivation: a HotFlip-style robustness gap could reflect a genuine
difference in each model's function, or an artifact of each attack being
optimized against its own loss landscape. Evaluating a trigger learned on
model A against model B tests whether the vulnerability it exploits is
specific to A or shared. This mirrors the T5-track transfer matrix already
reported in the paper (Section "Trigger Transferability").

Hardened against spot-instance deallocation:
- Each of the 4 (source, target) cells is persisted independently and
  atomically to `partial/uat_transfer_{source}_on_{target}.json`.
- Re-running this stage after a crash skips already-computed cells and
  resumes at the next one. One target model is held in memory at a time;
  both source triggers are evaluated against it before moving on, so each
  model is only loaded once.

Inputs:
- uat_results.json (from Stage 5), containing each model's learned
  trigger_ids
- baseline_checkpoints/best_model.pt, monotonic_checkpoints/best_model.pt
- The cached Pile attack texts from Stage 5 (falls back to a fresh load
  with identical sampling parameters if the cache is unavailable)

Outputs:
- uat_transfer_results.json (full 2x2 matrix + per-cell detail)
- stage_5b_uat_transfer_complete.flag
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Reuse the exact evaluation/model-loading logic from Stage 5 rather than
# duplicating it, so the transfer matrix is computed with identical
# tokenization, batching, and NLL-increase definitions as the diagonal
# (same-model) entries already reported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, save_json, load_json, atomic_save_json, load_json_safe,
    StageLogger, check_dependencies, partial_results_dir, load_pile_eval_texts,
)
from stage_5_uat_attacks import evaluate_trigger, _load_model

from transformers import AutoTokenizer

MODEL_TYPES = {
    'baseline_pythia': 'baseline',
    'monotonic_pythia': 'monotonic',
}


def _load_eval_texts(logger):
    """
    Recover the exact held-out evaluation split used by Stage 5, from its
    cached attack-texts file, so transfer-matrix entries are directly
    comparable to Stage 5's same-model (diagonal) results. Falls back to a
    fresh, deterministic load if the cache was cleaned up.
    """
    if Config.UAT_MAX_SAMPLES > 0:
        max_samples = Config.UAT_MAX_SAMPLES
    else:
        max_samples = 1500 if Config.USE_FULL_EVAL_SETS else 300

    partial = partial_results_dir()
    texts_cache = os.path.join(partial, f"pile_attack_texts_{max_samples}.json")
    if os.path.exists(texts_cache):
        logger.log(f"  Loading cached attack texts from {texts_cache}")
        all_texts = load_json(texts_cache)
    else:
        logger.log("  No cached attack texts found; reloading with Stage 5 parameters")
        all_texts = load_pile_eval_texts(max_samples, log_fn=logger.log)
        atomic_save_json(all_texts, texts_cache)

    split_idx = int(len(all_texts) * 0.4)
    eval_texts = all_texts[split_idx:]
    logger.log(f"  Evaluation set: {len(eval_texts)} examples (matches Stage 5 split)")
    return eval_texts


def main():
    logger = StageLogger("stage_5b_uat_transfer")

    try:
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic', 'stage_5_uat']):
            logger.complete(success=False)
            return 1

        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()

        uat_results_path = os.path.join(Config.RESULTS_DIR, 'uat_results.json')
        if not os.path.exists(uat_results_path):
            raise FileNotFoundError(
                f"Stage 5 results not found: {uat_results_path}. "
                "Run stage_5_uat_attacks.py first."
            )
        uat_results = load_json(uat_results_path)
        source_triggers = {}
        for model_name in MODEL_TYPES:
            if model_name not in uat_results['results']:
                raise KeyError(f"Missing trigger for '{model_name}' in {uat_results_path}")
            source_triggers[model_name] = uat_results['results'][model_name]['trigger_ids']
            trig_text = uat_results['results'][model_name]['trigger_text']
            logger.log(f"  Loaded trigger for {model_name}: \"{trig_text}\"")

        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, cache_dir=Config.DATA_CACHE_DIR,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        eval_texts = _load_eval_texts(logger)

        partial = partial_results_dir()
        transfer_cells = {}

        for target_name, target_type in MODEL_TYPES.items():
            # Skip loading the target model entirely if every cell that
            # targets it is already cached from a prior (interrupted) run.
            pending_sources = [
                src for src in MODEL_TYPES
                if not os.path.exists(
                    os.path.join(partial, f"uat_transfer_{src}_on_{target_name}.json")
                )
            ]
            if not pending_sources:
                logger.log(f"\n[{target_name}] All transfer cells already cached, skipping load.")
                for src in MODEL_TYPES:
                    cell_path = os.path.join(partial, f"uat_transfer_{src}_on_{target_name}.json")
                    transfer_cells[(src, target_name)] = load_json(cell_path)
                continue

            logger.log(f"\n{'='*80}\nLOADING TARGET MODEL: {target_name}\n{'='*80}")
            model, ckpt_path = _load_model(target_type, device)
            logger.log(f"  Checkpoint: {ckpt_path}")

            for src_name in MODEL_TYPES:
                cell_path = os.path.join(partial, f"uat_transfer_{src_name}_on_{target_name}.json")
                if os.path.exists(cell_path):
                    logger.log(f"  [{src_name} -> {target_name}] cached, skipping")
                    transfer_cells[(src_name, target_name)] = load_json(cell_path)
                    continue

                logger.log(f"  Evaluating trigger from {src_name} against {target_name}...")
                result = evaluate_trigger(
                    model, tokenizer, source_triggers[src_name], eval_texts, device,
                )
                logger.log(
                    f"    NLL increase: {result['nll_increase_percent']:.2f}% "
                    f"(clean={result['clean_loss']:.4f}, attacked={result['attacked_loss']:.4f})"
                )
                atomic_save_json(result, cell_path)
                transfer_cells[(src_name, target_name)] = result

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.log("\nAssembling transfer matrix...")
        transfer_matrix = {
            src: {
                tgt: transfer_cells[(src, tgt)]['nll_increase_percent']
                for tgt in MODEL_TYPES
            }
            for src in MODEL_TYPES
        }
        full_results = {
            src: {tgt: transfer_cells[(src, tgt)] for tgt in MODEL_TYPES}
            for src in MODEL_TYPES
        }

        logger.log("\nTransfer matrix (NLL increase %, source trigger x target model):")
        header = "source \\ target".ljust(20) + "".join(t.ljust(20) for t in MODEL_TYPES)
        logger.log("  " + header)
        for src in MODEL_TYPES:
            row = src.ljust(20) + "".join(
                f"{transfer_matrix[src][tgt]:+.2f}%".ljust(20) for tgt in MODEL_TYPES
            )
            logger.log("  " + row)

        results = {
            'seed': Config.CURRENT_SEED,
            'source_note': 'Triggers replayed from stage_5_uat_attacks.py (no re-optimization).',
            'num_eval_samples': len(eval_texts),
            'transfer_matrix': transfer_matrix,
            'full_results': full_results,
        }
        save_json(results, os.path.join(Config.RESULTS_DIR, 'uat_transfer_results.json'))

        logger.log("\nUAT transfer matrix complete.")
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
