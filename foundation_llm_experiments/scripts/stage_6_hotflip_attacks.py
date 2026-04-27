#!/usr/bin/env python3
"""
Stage 6: HotFlip Gradient-Based Attacks for Foundation LLMs

Performs gradient-based token flipping attacks to maximize perplexity.
Adapted from the T5 HotFlip implementation for decoder-only models.

Hardened against spot-instance deallocation:
- Results are streamed to a per-model JSONL file (`partial/hotflip_{model}.jsonl`)
  with `flush() + fsync()` after every example. Each record carries its
  sample index so a restarted run skips already-completed examples.
- Per-model summaries land atomically at `partial/hotflip_{model}.json`
  the moment a model finishes all samples.
- Aggregation, statistical tests, and the completion flag are only written
  once both models are fully done. Re-running this stage after a crash is
  idempotent and resumes from the exact example that was in flight.

Inputs:
- baseline_checkpoints/best_model.pt
- monotonic_checkpoints/best_model.pt
- Pile test data

Outputs:
- hotflip_results.json
- stage_6_hotflip_complete.flag
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, save_json, load_json, atomic_save_json,
    append_jsonl, load_jsonl, partial_results_dir,
    StageLogger, check_dependencies, make_model_monotonic,
    load_pile_eval_texts,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


class HotFlipAttacker:
    """HotFlip attack for causal language models."""

    def __init__(self, model, tokenizer, device, num_flips=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_flips = num_flips
        self.vocab_size = len(tokenizer)

    def attack_single_example(self, text):
        """Attack one text. Returns scalar metrics only (JSON-safe)."""
        encoding = self.tokenizer(
            text, return_tensors='pt', truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
        ).to(self.device)

        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        self.model.eval()
        with torch.no_grad():
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            clean_outputs = self.model(input_ids=input_ids, labels=labels)
            clean_loss = clean_outputs.loss.item()

        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = self.model(inputs_embeds=embeddings, labels=labels)
        loss = outputs.loss
        loss.backward()

        token_gradients = embeddings.grad[0]
        grad_magnitudes = token_gradients.norm(dim=1).detach()
        grad_magnitudes[attention_mask[0] == 0] = 0

        topk_positions = grad_magnitudes.topk(
            min(self.num_flips, len(grad_magnitudes))
        ).indices

        flipped_ids = input_ids.clone()

        for pos in topk_positions:
            pos_idx = pos.item()
            if pos_idx >= input_ids.size(1) or attention_mask[0, pos_idx] == 0:
                continue
            all_embeddings = embedding_layer.weight
            grad_at_pos = token_gradients[pos_idx]
            scores = torch.matmul(all_embeddings, grad_at_pos)
            best_token = scores.argmax().item()
            flipped_ids[0, pos_idx] = best_token

        with torch.no_grad():
            attacked_labels = flipped_ids.clone()
            attacked_labels[attention_mask == 0] = -100
            attacked_outputs = self.model(
                input_ids=flipped_ids, labels=attacked_labels,
            )
            attacked_loss = attacked_outputs.loss.item()

        degradation = (attacked_loss - clean_loss) / clean_loss if clean_loss else 0.0

        return {
            'clean_loss': float(clean_loss),
            'attacked_loss': float(attacked_loss),
            'degradation': float(degradation),
            'success': bool(degradation > Config.ATTACK_SUCCESS_THRESHOLD),
        }


def _attack_resumable(attacker, texts, jsonl_path, logger):
    """
    Run HotFlip attacks with per-example durable append.

    On restart, already-completed sample indices are reloaded from the
    JSONL file and skipped; the run continues from the next index. This
    guarantees O(examples_completed) rather than O(total_examples) work
    is lost to any single deallocation.
    """
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming: {len(done_by_idx)}/{len(texts)} examples already done.")

    pbar = tqdm(
        range(len(texts)),
        initial=len(done_by_idx),
        total=len(texts),
        desc="HotFlip attacks",
    )

    for i in pbar:
        if i in done_by_idx:
            continue
        try:
            result = attacker.attack_single_example(texts[i])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.log(f"  OOM on example {i}; recording a skip marker.")
            result = {
                'clean_loss': float('nan'),
                'attacked_loss': float('nan'),
                'degradation': 0.0,
                'success': False,
                'oom': True,
            }
        record = {"idx": i, **result}
        append_jsonl(jsonl_path, record)
        done_by_idx[i] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


def _aggregate(records):
    """Aggregate per-example records into summary metrics."""
    valid = [r for r in records if not r.get('oom', False) and
             not np.isnan(r.get('clean_loss', float('nan')))]
    if not valid:
        return {
            'avg_degradation': 0.0,
            'std_degradation': 0.0,
            'success_rate': 0.0,
            'avg_orig_loss': 0.0,
            'avg_attack_loss': 0.0,
            'num_samples': 0,
            'num_oom_skipped': sum(1 for r in records if r.get('oom', False)),
        }

    degs = np.array([r['degradation'] for r in valid])
    return {
        'avg_degradation': float(degs.mean()),
        'std_degradation': float(degs.std()),
        'success_rate': float(np.mean([r['success'] for r in valid])),
        'avg_orig_loss': float(np.mean([r['clean_loss'] for r in valid])),
        'avg_attack_loss': float(np.mean([r['attacked_loss'] for r in valid])),
        'num_samples': len(valid),
        'num_oom_skipped': sum(1 for r in records if r.get('oom', False)),
    }


def _load_model(model_type, device):
    if model_type == 'baseline':
        path = os.path.join(Config.CHECKPOINT_DIR, 'baseline_checkpoints', 'best_model.pt')
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME, cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.float32,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    else:
        path = os.path.join(Config.CHECKPOINT_DIR, 'monotonic_checkpoints', 'best_model.pt')
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME, cache_dir=Config.DATA_CACHE_DIR,
            torch_dtype=torch.float32,
        )
        model = make_model_monotonic(model, variant=Config.MONOTONIC_VARIANT)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model = model.to(device)
    model.eval()
    return model


def _load_pile_texts(logger, max_samples):
    """Stream Pile eval texts (skips training prefix to avoid leakage)."""
    return load_pile_eval_texts(max_samples, log_fn=logger.log)


def main():
    logger = StageLogger("stage_6_hotflip")

    try:
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic']):
            logger.complete(success=False)
            return 1

        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()

        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, cache_dir=Config.DATA_CACHE_DIR,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.log("Loading Pile test data...")
        max_samples = Config.HOTFLIP_NUM_SAMPLES

        partial = partial_results_dir()
        texts_cache = os.path.join(partial, f"hotflip_texts_{max_samples}.json")
        if os.path.exists(texts_cache):
            logger.log(f"  Loading cached hotflip texts from {texts_cache}")
            test_texts = load_json(texts_cache)
        else:
            test_texts = _load_pile_texts(logger, max_samples=max_samples)
            atomic_save_json(test_texts, texts_cache)
            logger.log(f"  Saved hotflip texts to {texts_cache}")
        logger.log(f"  Test set: {len(test_texts)} samples.")
        all_results = {}

        for model_name, model_type in [
            ('baseline_pythia', 'baseline'),
            ('monotonic_pythia', 'monotonic'),
        ]:
            summary_path = os.path.join(partial, f"hotflip_{model_name}.json")
            jsonl_path = os.path.join(partial, f"hotflip_{model_name}.jsonl")

            if os.path.exists(summary_path):
                logger.log(f"\n[{model_name}] Using cached summary from {summary_path}")
                all_results[model_name] = load_json(summary_path)
                continue

            logger.log(f"\n{'='*80}\nHOTFLIP ATTACK ON: {model_name}\n{'='*80}")
            logger.log("Loading model...")
            model = _load_model(model_type, device)

            logger.log(f"Running HotFlip on {len(test_texts)} examples "
                       f"(num_flips={Config.HOTFLIP_NUM_FLIPS})...")
            attacker = HotFlipAttacker(
                model, tokenizer, device, num_flips=Config.HOTFLIP_NUM_FLIPS,
            )

            records = _attack_resumable(attacker, test_texts, jsonl_path, logger)
            summary = _aggregate(records)
            summary['model_name'] = model_name

            logger.log(f"  Avg degradation: {summary['avg_degradation']*100:.2f}%")
            logger.log(f"  Success rate:    {summary['success_rate']*100:.1f}%")
            logger.log(f"  OOM-skipped:     {summary['num_oom_skipped']}")

            atomic_save_json(summary, summary_path)
            all_results[model_name] = summary

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.log("\nComputing statistical comparison...")
        baseline = all_results['baseline_pythia']
        monotonic = all_results['monotonic_pythia']
        statistical_tests = {
            'baseline_vs_monotonic': {
                'baseline_success_rate': baseline['success_rate'],
                'monotonic_success_rate': monotonic['success_rate'],
                'monotonic_more_robust': (
                    monotonic['success_rate'] < baseline['success_rate']
                ),
                'note': ('Per-example records persist in '
                         'partial/hotflip_{model}.jsonl for downstream t-tests.'),
            }
        }

        logger.log("\nSaving aggregated HotFlip results...")
        final_results = {
            'seed': Config.CURRENT_SEED,
            'attack_config': {
                'num_flips': Config.HOTFLIP_NUM_FLIPS,
                'num_samples': Config.HOTFLIP_NUM_SAMPLES,
                'success_threshold': Config.ATTACK_SUCCESS_THRESHOLD,
            },
            'results': all_results,
            'statistical_tests': statistical_tests,
        }

        save_json(
            final_results,
            os.path.join(Config.RESULTS_DIR, 'hotflip_results.json'),
        )

        logger.log("\nHotFlip attacks complete.")
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
