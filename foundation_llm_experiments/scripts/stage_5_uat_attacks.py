#!/usr/bin/env python3
"""
Stage 5: Universal Adversarial Trigger (UAT) Attacks for Foundation LLMs

Learns universal triggers that maximize perplexity when prepended to any input.
Adapted from the T5 UAT implementation for decoder-only models.

Hardened against spot-instance deallocation:
- Per-model partial results are persisted atomically to the results disk
  after each model (baseline / monotonic) completes.
- Within a single model, each completed restart is appended to a durable
  state file (`partial/uat_{model}_state.json`). A restarted process
  reloads this file, skips finished restarts, and resumes at the next
  restart index with a deterministic per-restart seed.
- Aggregation and completion flag are only written once both models are
  fully done. Re-running this stage after a crash is idempotent.

Inputs:
- baseline_checkpoints/best_model.pt
- monotonic_checkpoints/best_model.pt
- Pile test data

Outputs:
- uat_results.json
- learned_triggers.csv
- stage_5_uat_complete.flag
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, save_json, load_json, atomic_save_json, load_json_safe,
    StageLogger, check_dependencies, make_model_monotonic,
    partial_results_dir, load_pile_eval_texts,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


class UATOptimizer:
    """Universal adversarial trigger optimizer for causal LMs."""

    def __init__(self, model, tokenizer, device, trigger_length=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.trigger_length = trigger_length
        self.vocab_size = len(tokenizer)

    def _get_candidate_tokens(self):
        candidates = list(range(1000, min(5000, self.vocab_size)))
        disruptive_words = [
            'not', 'never', 'ignore', 'disregard', 'false', 'wrong',
            'error', 'invalid', 'corrupt', 'random', 'noise',
            '!!!', '???', '###', '***', '...', '---',
        ]
        for word in disruptive_words:
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            candidates.extend(token_ids)

        candidates = list({c for c in candidates if 0 < c < self.vocab_size})
        return candidates[:Config.ATTACK_NUM_CANDIDATES]

    def compute_trigger_loss(self, trigger_ids, texts, batch_size=8):
        self.model.eval()
        total_loss = 0.0
        num_examples = 0
        trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = [trigger_text + " " + t for t in texts[i:i+batch_size]]

                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_SEQ_LENGTH,
                ).to(self.device)

                labels = encodings.input_ids.clone()
                labels[encodings.attention_mask == 0] = -100

                outputs = self.model(
                    input_ids=encodings.input_ids,
                    attention_mask=encodings.attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item() * len(batch_texts)
                num_examples += len(batch_texts)

        return total_loss / num_examples if num_examples > 0 else 0.0

    def run_single_restart(self, texts, num_iterations, candidate_tokens, restart_seed):
        """
        Run one coordinate-ascent restart. Uses a dedicated numpy RandomState
        seeded with (global_seed + restart_index) so resumed runs remain
        deterministic and do not collide with the main numpy state that
        gets advanced between restarts.
        """
        rng = np.random.RandomState(restart_seed)
        trigger_ids = rng.choice(candidate_tokens, size=self.trigger_length).tolist()

        current_loss = self.compute_trigger_loss(trigger_ids, texts)

        for iteration in range(num_iterations):
            improved = False
            for pos in range(self.trigger_length):
                best_pos_loss = current_loss
                best_pos_token = trigger_ids[pos]

                for _ in range(20):
                    candidate = int(rng.choice(candidate_tokens))
                    trigger_ids[pos] = candidate
                    loss = self.compute_trigger_loss(trigger_ids, texts)
                    if loss > best_pos_loss:
                        best_pos_loss = loss
                        best_pos_token = candidate
                        improved = True

                trigger_ids[pos] = best_pos_token
                current_loss = best_pos_loss

            if iteration % 20 == 0:
                trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
                print(f"    iter {iteration}: loss={current_loss:.4f} trigger=\"{trigger_text}\"")

            if not improved:
                print(f"    converged at iteration {iteration}")
                break

        final_loss = self.compute_trigger_loss(trigger_ids, texts)
        return trigger_ids, final_loss


def optimize_trigger_resumable(
    optimizer, texts, num_iterations, num_restarts, state_path, base_seed, logger
):
    """
    Drive UAT restarts with per-restart durable checkpoints.

    State schema at `state_path`:
      {
        "completed_restarts": [
            {"trigger_ids": [...], "final_loss": float, "restart_seed": int},
            ...
        ],
        "best_trigger": [...] | None,
        "best_loss": float,
      }
    """
    default_state = {
        "completed_restarts": [],
        "best_trigger": None,
        "best_loss": float('-inf'),
    }
    state = load_json_safe(state_path, default=None)
    if not isinstance(state, dict) or "completed_restarts" not in state:
        state = default_state

    candidate_tokens = optimizer._get_candidate_tokens()
    completed = len(state["completed_restarts"])
    if completed > 0:
        logger.log(f"  Resuming: {completed}/{num_restarts} restarts already done.")

    for restart in range(completed, num_restarts):
        logger.log(f"  Restart {restart + 1}/{num_restarts}")
        restart_seed = base_seed + 1000 * (restart + 1)
        trigger_ids, final_loss = optimizer.run_single_restart(
            texts=texts,
            num_iterations=num_iterations,
            candidate_tokens=candidate_tokens,
            restart_seed=restart_seed,
        )

        state["completed_restarts"].append({
            "trigger_ids": trigger_ids,
            "final_loss": float(final_loss),
            "restart_seed": restart_seed,
        })
        if final_loss > state["best_loss"]:
            state["best_loss"] = float(final_loss)
            state["best_trigger"] = list(trigger_ids)

        atomic_save_json(state, state_path)
        logger.log(f"  Restart {restart + 1} done: loss={final_loss:.4f} "
                   f"(best so far={state['best_loss']:.4f})")

    return state["best_trigger"], state["best_loss"]


def evaluate_trigger(model, tokenizer, trigger_ids, texts, device):
    """Evaluate trigger impact on held-out texts."""
    model.eval()

    clean_encodings = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True,
        max_length=Config.MAX_SEQ_LENGTH,
    ).to(device)

    with torch.no_grad():
        clean_labels = clean_encodings.input_ids.clone()
        clean_labels[clean_encodings.attention_mask == 0] = -100
        clean_outputs = model(**clean_encodings, labels=clean_labels)
        clean_loss = clean_outputs.loss.item()

    trigger_text = tokenizer.decode(trigger_ids, skip_special_tokens=True)
    attacked_texts = [trigger_text + " " + text for text in texts]

    attacked_encodings = tokenizer(
        attacked_texts, return_tensors='pt', padding=True, truncation=True,
        max_length=Config.MAX_SEQ_LENGTH,
    ).to(device)

    with torch.no_grad():
        attacked_labels = attacked_encodings.input_ids.clone()
        attacked_labels[attacked_encodings.attention_mask == 0] = -100
        attacked_outputs = model(**attacked_encodings, labels=attacked_labels)
        attacked_loss = attacked_outputs.loss.item()

    nll_increase = (attacked_loss - clean_loss) / clean_loss

    return {
        'trigger_text': trigger_text,
        'trigger_ids': list(trigger_ids),
        'clean_loss': clean_loss,
        'attacked_loss': attacked_loss,
        'nll_increase': nll_increase,
        'nll_increase_percent': nll_increase * 100,
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
        model = make_model_monotonic(model)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model = model.to(device)
    model.eval()
    return model, path


def _load_pile_texts(logger, max_samples):
    """Stream Pile eval texts (skips training prefix to avoid leakage)."""
    return load_pile_eval_texts(max_samples, log_fn=logger.log)


def main():
    logger = StageLogger("stage_5_uat")

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
        max_samples = 1500 if Config.USE_FULL_EVAL_SETS else 300

        partial = partial_results_dir()
        texts_cache = os.path.join(partial, f"pile_attack_texts_{max_samples}.json")
        if os.path.exists(texts_cache):
            logger.log(f"  Loading cached attack texts from {texts_cache}")
            all_texts = load_json(texts_cache)
        else:
            all_texts = _load_pile_texts(logger, max_samples=max_samples)
            atomic_save_json(all_texts, texts_cache)
            logger.log(f"  Saved attack texts to {texts_cache}")
        logger.log(f"  Loaded {len(all_texts)} test samples.")

        split_idx = int(len(all_texts) * 0.4)
        opt_texts = all_texts[:split_idx]
        eval_texts = all_texts[split_idx:]
        logger.log(f"  Optimization set: {len(opt_texts)}, evaluation set: {len(eval_texts)}")

        partial = partial_results_dir()
        all_results = {}

        for model_name, model_type in [
            ('baseline_pythia', 'baseline'),
            ('monotonic_pythia', 'monotonic'),
        ]:
            result_path = os.path.join(partial, f"uat_{model_name}.json")
            state_path = os.path.join(partial, f"uat_{model_name}_state.json")

            if os.path.exists(result_path):
                logger.log(f"\n[{model_name}] Using cached result from {result_path}")
                all_results[model_name] = load_json(result_path)
                continue

            logger.log(f"\n{'='*80}\nOPTIMIZING TRIGGER FOR: {model_name}\n{'='*80}")
            logger.log("Loading model...")
            model, ckpt_path = _load_model(model_type, device)
            logger.log(f"  Checkpoint: {ckpt_path}")

            logger.log("Optimizing universal trigger...")
            logger.log(f"  Trigger length: {Config.ATTACK_TRIGGER_LENGTH}")
            logger.log(f"  Iterations:     {Config.ATTACK_NUM_ITERATIONS}")
            logger.log(f"  Restarts:       {Config.ATTACK_NUM_RESTARTS}")

            optimizer = UATOptimizer(
                model, tokenizer, device,
                trigger_length=Config.ATTACK_TRIGGER_LENGTH,
            )

            base_seed = Config.CURRENT_SEED + (0 if model_type == 'baseline' else 7)
            best_trigger, best_loss = optimize_trigger_resumable(
                optimizer=optimizer,
                texts=opt_texts,
                num_iterations=Config.ATTACK_NUM_ITERATIONS,
                num_restarts=Config.ATTACK_NUM_RESTARTS,
                state_path=state_path,
                base_seed=base_seed,
                logger=logger,
            )

            trigger_text = tokenizer.decode(best_trigger, skip_special_tokens=True)
            logger.log(f"  Best trigger: \"{trigger_text}\" (opt loss={best_loss:.4f})")

            logger.log("Evaluating trigger on held-out set...")
            eval_results = evaluate_trigger(
                model, tokenizer, best_trigger, eval_texts, device,
            )
            logger.log(f"  Clean loss:    {eval_results['clean_loss']:.4f}")
            logger.log(f"  Attacked loss: {eval_results['attacked_loss']:.4f}")
            logger.log(f"  NLL increase:  {eval_results['nll_increase_percent']:.2f}%")

            atomic_save_json(eval_results, result_path)
            all_results[model_name] = eval_results

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.log("\nSaving aggregated UAT results...")
        results = {
            'seed': Config.CURRENT_SEED,
            'attack_config': {
                'trigger_length': Config.ATTACK_TRIGGER_LENGTH,
                'num_iterations': Config.ATTACK_NUM_ITERATIONS,
                'num_restarts': Config.ATTACK_NUM_RESTARTS,
                'num_opt_samples': len(opt_texts),
                'num_eval_samples': len(eval_texts),
            },
            'results': all_results,
        }

        save_json(results, os.path.join(Config.RESULTS_DIR, 'uat_results.json'))

        import csv
        csv_path = os.path.join(Config.RESULTS_DIR, 'learned_triggers.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'trigger', 'nll_increase_percent'])
            for model_name, result in all_results.items():
                writer.writerow([
                    model_name,
                    result['trigger_text'],
                    f"{result['nll_increase_percent']:.2f}",
                ])

        logger.log(f"  Saved triggers to: {csv_path}")
        logger.log("\nUAT attacks complete.")
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
