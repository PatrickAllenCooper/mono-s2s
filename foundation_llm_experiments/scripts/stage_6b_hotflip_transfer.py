#!/usr/bin/env python3
"""
Stage 6b: HotFlip Substitution Transfer + Gradient-Free Controls (Foundation LLMs)

Answers a question Stage 6 cannot: is the baseline-vs-monotonic HotFlip
success-rate gap a property of each model's function, or an artifact of the
fact that each model's attack is optimized against its own gradients?

Three checks, all evaluated on the same held-out text sample:

1. Substitution transfer -- HotFlip is re-run on each source model (with
   gradients, exactly as in Stage 6) and the resulting token substitutions
   are persisted, then replayed as fixed inputs against BOTH models. This
   yields the full 2x2 (crafted-on x evaluated-on) matrix. The diagonal
   reproduces Stage 6's own numbers; the off-diagonal cells test transfer.
2. Random-substitution control -- the same number of flips (matched budget)
   at uniformly random positions, with replacement tokens drawn from the
   same restricted candidate vocabulary used elsewhere in this codebase
   (see UATOptimizer._get_candidate_tokens), evaluated on both models.
   No model or gradient is involved in generating this control at all.
3. Query-based (gradient-free) attack -- a greedy per-position search that
   scores candidate flips purely by the *target* model's own loss (no
   gradients anywhere), run directly against each model. If monotonicity's
   robustness gain survives an attack that cannot see gradients at all,
   the gain is unlikely to be a gradient-masking artifact of HotFlip's
   specific optimizer. This control is the most expensive (many forward
   passes per example) and can be skipped via Config.RUN_QUERY_ATTACK.

Hardened against spot-instance deallocation, following the Stage 6 pattern:
- Per-example JSONL logs with idx-based resume for both the re-attack step
  and each cross-model evaluation.
- Per-stage summaries persisted atomically; a rerun after a crash skips
  everything already completed.

Inputs:
- baseline_checkpoints/best_model.pt, monotonic_checkpoints/best_model.pt
- The cached Stage 6 Pile texts (falls back to a fresh identical load)

Outputs:
- hotflip_transfer_results.json
- stage_6b_hotflip_transfer_complete.flag
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, save_json, load_json, atomic_save_json,
    append_jsonl, load_jsonl, partial_results_dir,
    StageLogger, check_dependencies, load_pile_eval_texts,
)
from stage_6_hotflip_attacks import HotFlipAttacker, _load_model, _aggregate
from stage_5_uat_attacks import UATOptimizer

from transformers import AutoTokenizer

MODEL_TYPES = {
    'baseline_pythia': 'baseline',
    'monotonic_pythia': 'monotonic',
}


def _load_test_texts(logger):
    """Recover the exact Stage 6 text sample (or an identical fresh load)."""
    max_samples = Config.HOTFLIP_TRANSFER_NUM_SAMPLES
    partial = partial_results_dir()
    texts_cache = os.path.join(partial, f"hotflip_texts_{max_samples}.json")
    if os.path.exists(texts_cache):
        logger.log(f"  Loading cached hotflip texts from {texts_cache}")
        return load_json(texts_cache)
    # Fall back to Stage 6's default sample size if the transfer-specific
    # cache doesn't exist but Stage 6's own cache does (common case: same
    # sample count used for both stages).
    stage6_cache = os.path.join(partial, f"hotflip_texts_{Config.HOTFLIP_NUM_SAMPLES}.json")
    if os.path.exists(stage6_cache) and max_samples == Config.HOTFLIP_NUM_SAMPLES:
        logger.log(f"  Loading cached hotflip texts from {stage6_cache}")
        return load_json(stage6_cache)
    logger.log("  No cached hotflip texts found; loading fresh (same parameters as Stage 6)")
    texts = load_pile_eval_texts(max_samples, log_fn=logger.log)
    atomic_save_json(texts, texts_cache)
    return texts


def _evaluate_precomputed(model, orig_ids, flipped_ids, attention_mask, device):
    """Forward-only (no gradient) re-evaluation of a fixed attacked sequence
    on a possibly different model than the one that produced it."""
    model.eval()
    orig = torch.tensor([orig_ids], device=device)
    flipped = torch.tensor([flipped_ids], device=device)
    mask = torch.tensor([attention_mask], device=device)

    with torch.no_grad():
        clean_labels = orig.clone()
        clean_labels[mask == 0] = -100
        clean_loss = model(input_ids=orig, attention_mask=mask, labels=clean_labels).loss.item()

        attacked_labels = flipped.clone()
        attacked_labels[mask == 0] = -100
        attacked_loss = model(input_ids=flipped, attention_mask=mask, labels=attacked_labels).loss.item()

    degradation = (attacked_loss - clean_loss) / clean_loss if clean_loss else 0.0
    return {
        'clean_loss': float(clean_loss),
        'attacked_loss': float(attacked_loss),
        'degradation': float(degradation),
        'success': bool(degradation > Config.ATTACK_SUCCESS_THRESHOLD),
    }


def _craft_resumable(attacker, texts, jsonl_path, logger):
    """Re-run gradient-based HotFlip on the source model, persisting ids."""
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming craft: {len(done_by_idx)}/{len(texts)} examples already done.")

    for i in tqdm(range(len(texts)), initial=len(done_by_idx), total=len(texts), desc="Crafting"):
        if i in done_by_idx:
            continue
        try:
            result = attacker.attack_single_example(texts[i], return_ids=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.log(f"  OOM on example {i}; recording a skip marker.")
            result = {'oom': True}
        record = {"idx": i, **result}
        append_jsonl(jsonl_path, record)
        done_by_idx[i] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


def _cross_eval_resumable(model, craft_records, jsonl_path, logger, label):
    """Replay each crafted attack (from a possibly different source model)
    against `model`, resumable by example index."""
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming eval[{label}]: {len(done_by_idx)}/{len(craft_records)} done.")

    for rec in tqdm(craft_records, desc=f"Eval[{label}]"):
        idx = rec["idx"]
        if idx in done_by_idx:
            continue
        if rec.get('oom', False):
            record = {"idx": idx, "oom": True}
        else:
            try:
                metrics = _evaluate_precomputed(
                    model, rec['orig_ids'], rec['flipped_ids'], rec['attention_mask'],
                    next(model.parameters()).device,
                )
                record = {"idx": idx, **metrics}
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                record = {"idx": idx, "oom": True}
        append_jsonl(jsonl_path, record)
        done_by_idx[idx] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


def _random_substitution_attack(text, tokenizer, rng, num_flips, candidate_tokens, max_length):
    """Purely random control: no model, no gradient. Uniform positions,
    replacements drawn from the same restricted candidate vocabulary used
    by the UAT and query attacks."""
    encoding = tokenizer(text, truncation=True, max_length=max_length)
    orig_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', [1] * len(orig_ids))
    seq_len = len(orig_ids)
    if seq_len == 0:
        return None

    k = min(num_flips, seq_len)
    positions = rng.choice(seq_len, size=k, replace=False)
    flipped_ids = list(orig_ids)
    for pos in positions:
        flipped_ids[int(pos)] = int(rng.choice(candidate_tokens))

    return {
        'orig_ids': orig_ids,
        'flipped_ids': flipped_ids,
        'attention_mask': attention_mask,
        'positions_flipped': sorted(int(p) for p in positions),
    }


def _random_control_resumable(texts, tokenizer, candidate_tokens, jsonl_path, base_seed, logger):
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming random control: {len(done_by_idx)}/{len(texts)} done.")

    for i in tqdm(range(len(texts)), initial=len(done_by_idx), total=len(texts), desc="Random control"):
        if i in done_by_idx:
            continue
        rng = np.random.RandomState(base_seed + i)
        attacked = _random_substitution_attack(
            texts[i], tokenizer, rng, Config.HOTFLIP_NUM_FLIPS, candidate_tokens,
            Config.MAX_SEQ_LENGTH,
        )
        record = {"idx": i, **attacked} if attacked else {"idx": i, "oom": True}
        append_jsonl(jsonl_path, record)
        done_by_idx[i] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


class QueryAttacker:
    """
    Gradient-free, query-based attack. Positions are chosen uniformly at
    random (same policy as the random-substitution control); at each
    position, `candidates_per_position` candidate tokens are tried and the
    one that most increases the target model's own loss is kept greedily.
    No gradients are computed anywhere in this class.

    Total forward passes per example: num_flips * candidates_per_position + 2
    (one for the clean baseline, one for the final joint evaluation). This is
    a deliberately coarse, cost-bounded stand-in for "matching HotFlip's
    effective candidate evaluations": HotFlip scores the full vocabulary at
    every position via a single matmul against the gradient, which has no
    literal forward-pass equivalent, so we match the per-position candidate
    *count* to the UAT attack's own coordinate-ascent budget (20) instead.
    """

    def __init__(self, model, tokenizer, device, num_flips, candidates_per_position, candidate_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_flips = num_flips
        self.candidates_per_position = candidates_per_position
        self.candidate_tokens = candidate_tokens

    def _loss(self, ids, mask):
        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.tensor([mask], device=self.device)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attn, labels=labels).loss.item()

    def attack_single_example(self, text, rng):
        encoding = self.tokenizer(text, truncation=True, max_length=Config.MAX_SEQ_LENGTH)
        orig_ids = encoding['input_ids']
        attention_mask = encoding.get('attention_mask', [1] * len(orig_ids))
        seq_len = len(orig_ids)
        if seq_len == 0:
            return None

        clean_loss = self._loss(orig_ids, attention_mask)

        k = min(self.num_flips, seq_len)
        positions = rng.choice(seq_len, size=k, replace=False)
        flipped_ids = list(orig_ids)
        current_loss = clean_loss

        for pos in positions:
            pos = int(pos)
            best_loss = current_loss
            best_token = flipped_ids[pos]
            for _ in range(self.candidates_per_position):
                candidate = int(rng.choice(self.candidate_tokens))
                trial = list(flipped_ids)
                trial[pos] = candidate
                loss = self._loss(trial, attention_mask)
                if loss > best_loss:
                    best_loss = loss
                    best_token = candidate
            flipped_ids[pos] = best_token
            current_loss = best_loss

        attacked_loss = self._loss(flipped_ids, attention_mask)
        degradation = (attacked_loss - clean_loss) / clean_loss if clean_loss else 0.0
        return {
            'clean_loss': float(clean_loss),
            'attacked_loss': float(attacked_loss),
            'degradation': float(degradation),
            'success': bool(degradation > Config.ATTACK_SUCCESS_THRESHOLD),
        }


def _query_attack_resumable(attacker, texts, jsonl_path, base_seed, logger):
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming query attack: {len(done_by_idx)}/{len(texts)} done.")

    for i in tqdm(range(len(texts)), initial=len(done_by_idx), total=len(texts), desc="Query attack"):
        if i in done_by_idx:
            continue
        rng = np.random.RandomState(base_seed + i)
        try:
            result = attacker.attack_single_example(texts[i], rng)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = None
        record = {"idx": i, **result} if result else {"idx": i, "oom": True}
        append_jsonl(jsonl_path, record)
        done_by_idx[i] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


def main():
    logger = StageLogger("stage_6b_hotflip_transfer")

    try:
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic', 'stage_6_hotflip']):
            logger.complete(success=False)
            return 1

        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()
        partial = partial_results_dir()

        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, cache_dir=Config.DATA_CACHE_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_texts = _load_test_texts(logger)
        logger.log(f"  Test set: {len(test_texts)} samples.")

        # Candidate vocabulary shared by the random control and the query
        # attack (same restricted pool the UAT attack uses).
        candidate_tokens = UATOptimizer(
            model=None, tokenizer=tokenizer, device=device,
        )._get_candidate_tokens()
        logger.log(f"  Candidate vocabulary size: {len(candidate_tokens)}")

        # ------------------------------------------------------------
        # Step 1: craft (re-attack with gradients) on each source model
        # ------------------------------------------------------------
        craft_records = {}
        for source_name, source_type in MODEL_TYPES.items():
            craft_path = os.path.join(partial, f"hotflip_transfer_craft_{source_name}.jsonl")
            done = load_jsonl(craft_path)
            if len(done) >= len(test_texts):
                logger.log(f"\n[{source_name}] Craft already complete ({len(done)} examples), skipping load.")
                craft_records[source_name] = sorted(done, key=lambda r: r['idx'])
                continue

            logger.log(f"\n{'='*80}\nCRAFTING ATTACKED INPUTS ON: {source_name}\n{'='*80}")
            model = _load_model(source_type, device)
            attacker = HotFlipAttacker(model, tokenizer, device, num_flips=Config.HOTFLIP_NUM_FLIPS)
            craft_records[source_name] = _craft_resumable(attacker, test_texts, craft_path, logger)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------
        # Step 2: cross-evaluate every source's crafted inputs on every model
        # ------------------------------------------------------------
        substitution_transfer = {src: {} for src in MODEL_TYPES}
        for target_name, target_type in MODEL_TYPES.items():
            pending = [
                src for src in MODEL_TYPES
                if not os.path.exists(
                    os.path.join(partial, f"hotflip_transfer_eval_{src}_on_{target_name}_summary.json")
                )
            ]
            if not pending:
                logger.log(f"\n[{target_name}] All cross-eval cells cached, skipping load.")
                for src in MODEL_TYPES:
                    summary_path = os.path.join(
                        partial, f"hotflip_transfer_eval_{src}_on_{target_name}_summary.json"
                    )
                    substitution_transfer[src][target_name] = load_json(summary_path)
                continue

            logger.log(f"\n{'='*80}\nCROSS-EVALUATING TARGET: {target_name}\n{'='*80}")
            model = _load_model(target_type, device)
            for src in MODEL_TYPES:
                summary_path = os.path.join(
                    partial, f"hotflip_transfer_eval_{src}_on_{target_name}_summary.json"
                )
                if os.path.exists(summary_path):
                    substitution_transfer[src][target_name] = load_json(summary_path)
                    continue
                eval_jsonl = os.path.join(partial, f"hotflip_transfer_eval_{src}_on_{target_name}.jsonl")
                label = f"{src}->{target_name}"
                records = _cross_eval_resumable(model, craft_records[src], eval_jsonl, logger, label)
                summary = _aggregate(records)
                summary['crafted_on'] = src
                summary['evaluated_on'] = target_name
                atomic_save_json(summary, summary_path)
                substitution_transfer[src][target_name] = summary
                logger.log(f"  [{label}] success_rate={summary['success_rate']*100:.1f}% "
                           f"avg_degradation={summary['avg_degradation']*100:.2f}%")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------
        # Step 3: random-substitution control (model-agnostic generation,
        # evaluated on both models)
        # ------------------------------------------------------------
        logger.log(f"\n{'='*80}\nRANDOM-SUBSTITUTION CONTROL\n{'='*80}")
        random_craft_path = os.path.join(partial, "hotflip_transfer_random_craft.jsonl")
        random_records = _random_control_resumable(
            test_texts, tokenizer, candidate_tokens, random_craft_path,
            base_seed=Config.CURRENT_SEED + 9999, logger=logger,
        )
        random_control = {}
        for target_name, target_type in MODEL_TYPES.items():
            summary_path = os.path.join(partial, f"hotflip_transfer_random_eval_{target_name}_summary.json")
            if os.path.exists(summary_path):
                random_control[target_name] = load_json(summary_path)
                continue
            logger.log(f"  Evaluating random control on {target_name}...")
            model = _load_model(target_type, device)
            eval_jsonl = os.path.join(partial, f"hotflip_transfer_random_eval_{target_name}.jsonl")
            records = _cross_eval_resumable(model, random_records, eval_jsonl, logger, f"random->{target_name}")
            summary = _aggregate(records)
            summary['evaluated_on'] = target_name
            atomic_save_json(summary, summary_path)
            random_control[target_name] = summary
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------
        # Step 4 (optional): query-based, gradient-free attack directly
        # against each model
        # ------------------------------------------------------------
        query_attack = None
        if Config.RUN_QUERY_ATTACK:
            logger.log(f"\n{'='*80}\nQUERY-BASED (GRADIENT-FREE) ATTACK\n{'='*80}")
            query_texts = test_texts[:Config.QUERY_ATTACK_NUM_SAMPLES]
            query_attack = {}
            for target_name, target_type in MODEL_TYPES.items():
                summary_path = os.path.join(partial, f"hotflip_transfer_query_{target_name}_summary.json")
                if os.path.exists(summary_path):
                    query_attack[target_name] = load_json(summary_path)
                    continue
                logger.log(f"  Running query attack directly against {target_name}...")
                model = _load_model(target_type, device)
                attacker = QueryAttacker(
                    model, tokenizer, device,
                    num_flips=Config.HOTFLIP_NUM_FLIPS,
                    candidates_per_position=Config.QUERY_ATTACK_CANDIDATES_PER_POSITION,
                    candidate_tokens=candidate_tokens,
                )
                query_jsonl = os.path.join(partial, f"hotflip_transfer_query_{target_name}.jsonl")
                base_seed = Config.CURRENT_SEED + (0 if target_type == 'baseline' else 7) + 5000
                records = _query_attack_resumable(attacker, query_texts, query_jsonl, base_seed, logger)
                summary = _aggregate(records)
                summary['evaluated_on'] = target_name
                summary['num_flips'] = Config.HOTFLIP_NUM_FLIPS
                summary['candidates_per_position'] = Config.QUERY_ATTACK_CANDIDATES_PER_POSITION
                atomic_save_json(summary, summary_path)
                query_attack[target_name] = summary
                logger.log(f"  [{target_name}] success_rate={summary['success_rate']*100:.1f}% "
                           f"avg_degradation={summary['avg_degradation']*100:.2f}%")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.log("\nSkipping query-based attack (Config.RUN_QUERY_ATTACK=False).")

        logger.log("\nSaving aggregated transfer/control results...")
        final_results = {
            'seed': Config.CURRENT_SEED,
            'attack_config': {
                'num_flips': Config.HOTFLIP_NUM_FLIPS,
                'num_samples': len(test_texts),
                'query_attack_num_samples': Config.QUERY_ATTACK_NUM_SAMPLES if Config.RUN_QUERY_ATTACK else 0,
                'query_attack_candidates_per_position': Config.QUERY_ATTACK_CANDIDATES_PER_POSITION,
                'success_threshold': Config.ATTACK_SUCCESS_THRESHOLD,
            },
            'substitution_transfer': substitution_transfer,
            'random_control': random_control,
            'query_attack': query_attack,
        }
        save_json(final_results, os.path.join(Config.RESULTS_DIR, 'hotflip_transfer_results.json'))

        logger.log("\nHotFlip substitution transfer + controls complete.")
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
