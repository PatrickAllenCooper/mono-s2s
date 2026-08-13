#!/usr/bin/env python3
"""
Stage 6b: HotFlip Substitution Transfer + Gradient-Free Controls (T5)

Answers a question Stage 6 cannot: is the baseline-vs-monotonic HotFlip
success-rate gap a property of each model's function, or an artifact of the
fact that each model's attack is optimized against its own gradients?

Three checks, mirroring foundation_llm_experiments/scripts/stage_6b_hotflip_transfer.py
for the Pythia track:

1. Substitution transfer -- HotFlip is re-run on baseline T5 and monotonic T5
   (with gradients, exactly as in Stage 6) and the resulting token
   substitutions are persisted as raw ids, then replayed as fixed inputs
   against BOTH models. This yields the full 2x2 (crafted-on x
   evaluated-on) matrix; the diagonal reproduces Stage 6's own numbers.
2. Random-substitution control -- the same number of flips
   (ExperimentConfig.ATTACK_TRIGGER_LENGTH = 5) at uniformly random
   positions, replacements drawn from the same restricted candidate
   vocabulary used by the UAT attack, evaluated on both models. No model or
   gradient is involved in generating this control.
3. Query-based (gradient-free) attack -- a greedy per-position search that
   scores candidate flips purely by the *target* model's own loss (no
   gradients anywhere), run directly against each model. Skippable via
   ExperimentConfig.RUN_QUERY_ATTACK if time-constrained.

Only baseline and monotonic T5 participate (not standard T5), matching the
plan's framing: "apply substitutions found via baseline's gradients to the
monotone model, and vice versa."

Hardened with the same JSONL/atomic-save resume pattern used by the
Pythia-track stages (see utils/common_utils.py: append_jsonl, load_jsonl,
atomic_save_json, partial_results_dir).

Inputs:
- attack_data.pt (from stage 1) -- evaluation split
- baseline_checkpoints/best_model.pt, monotonic_checkpoints/best_model.pt

Outputs:
- hotflip_transfer_results.json
- stage_6b_hotflip_transfer_complete.flag
"""

import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, check_dependencies, save_json, load_json, load_json_safe,
    atomic_save_json, append_jsonl, load_jsonl, partial_results_dir,
    StageLogger, load_model,
)
from stage_6_hotflip_attacks import HotFlipT5Attack

from transformers import T5Tokenizer

MODEL_TYPES = {
    'baseline_t5': 'baseline',
    'monotonic_t5': 'monotonic',
}
SPECIAL_WORDS = ['not', 'never', 'ignore', 'disregard', 'false', 'error', '!!!', '???', '###']


def _get_candidate_vocab(tokenizer):
    """Restricted candidate vocabulary shared by the random control and the
    query attack. Mirrors AggressiveUATAttack._get_disruptive_vocab in
    stage_5_uat_attacks.py so all attack families here draw from a
    consistent, model-agnostic token pool."""
    vocab_size = tokenizer.vocab_size
    candidates = list(range(1000, min(5000, vocab_size)))
    for word in SPECIAL_WORDS:
        candidates.extend(tokenizer.encode(word, add_special_tokens=False))
    candidates = list({c for c in candidates if 0 < c < vocab_size})
    return candidates


def _tokenize_source(tokenizer, text, device, max_len=512):
    enc = tokenizer(
        "summarize: " + text, return_tensors="pt", truncation=True, max_length=max_len,
    ).to(device)
    return enc.input_ids, enc.attention_mask


def _forward_loss(model, input_ids, attention_mask, target_ids):
    with torch.no_grad():
        return model(
            input_ids=input_ids, attention_mask=attention_mask, labels=target_ids,
        ).loss.item()


def _load_target_ids(tokenizer, summary, device, max_len=128):
    return tokenizer(
        summary, return_tensors="pt", truncation=True, max_length=max_len,
    ).to(device).input_ids


def _craft_resumable(attacker, texts, summaries, jsonl_path, logger, num_flips):
    """Re-run gradient-based HotFlip on the source model, persisting ids."""
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming craft: {len(done_by_idx)}/{len(texts)} examples already done.")

    for i in tqdm(range(len(texts)), initial=len(done_by_idx), total=len(texts), desc="Crafting"):
        if i in done_by_idx:
            continue
        clean_loss = attacker.compute_loss(texts[i], summaries[i])
        _, _, ids_dict = attacker.attack_single(
            texts[i], summaries[i], num_flips=num_flips, return_ids=True,
        )
        record = {"idx": i, "clean_loss": clean_loss, **ids_dict}
        append_jsonl(jsonl_path, record)
        done_by_idx[i] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


def _cross_eval_resumable(model, tokenizer, craft_records, summaries, device, jsonl_path, logger, label):
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
        orig_ids = torch.tensor([rec['orig_ids']], device=device)
        attacked_ids = torch.tensor([rec['attacked_ids']], device=device)
        attn = torch.tensor([rec['attention_mask']], device=device)
        target_ids = _load_target_ids(tokenizer, summaries[idx], device)

        clean_loss = _forward_loss(model, orig_ids, attn, target_ids)
        attacked_loss = _forward_loss(model, attacked_ids, attn, target_ids)
        degradation = (attacked_loss - clean_loss) / clean_loss if clean_loss else 0.0
        record = {
            "idx": idx,
            "clean_loss": float(clean_loss),
            "attacked_loss": float(attacked_loss),
            "degradation": float(degradation),
            "success": bool(degradation > 0.1),
        }
        append_jsonl(jsonl_path, record)
        done_by_idx[idx] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


def _aggregate(records):
    valid = [r for r in records if not r.get('oom', False)]
    if not valid:
        return {'avg_degradation': 0.0, 'std_degradation': 0.0, 'success_rate': 0.0, 'num_samples': 0}
    degs = np.array([r['degradation'] for r in valid])
    return {
        'avg_degradation': float(degs.mean()),
        'std_degradation': float(degs.std()),
        'success_rate': float(np.mean([r['success'] for r in valid])),
        'avg_orig_loss': float(np.mean([r['clean_loss'] for r in valid])),
        'avg_attack_loss': float(np.mean([r['attacked_loss'] for r in valid])),
        'num_samples': len(valid),
    }


def _random_substitution(tokenizer, text, device, rng, num_flips, candidate_tokens):
    """Purely random control: no model, no gradient. Uniform positions
    (excluding the first token / special tokens), replacements drawn from
    the shared candidate vocabulary."""
    input_ids, attention_mask = _tokenize_source(tokenizer, text, device)
    seq_len = input_ids.size(1)
    special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
    valid_positions = [
        i for i in range(1, seq_len)
        if attention_mask[0, i].item() == 1 and input_ids[0, i].item() not in special_ids
    ]
    if not valid_positions:
        return None
    k = min(num_flips, len(valid_positions))
    positions = rng.choice(valid_positions, size=k, replace=False)
    flipped = input_ids.clone()
    for pos in positions:
        flipped[0, int(pos)] = int(rng.choice(candidate_tokens))

    return {
        'orig_ids': input_ids[0].cpu().tolist(),
        'attacked_ids': flipped[0].cpu().tolist(),
        'attention_mask': attention_mask[0].cpu().tolist(),
    }


def _random_control_resumable(tokenizer, texts, device, candidate_tokens, jsonl_path, base_seed, num_flips, logger):
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming random control: {len(done_by_idx)}/{len(texts)} done.")

    for i in tqdm(range(len(texts)), initial=len(done_by_idx), total=len(texts), desc="Random control"):
        if i in done_by_idx:
            continue
        rng = np.random.RandomState(base_seed + i)
        attacked = _random_substitution(tokenizer, texts[i], device, rng, num_flips, candidate_tokens)
        record = {"idx": i, **attacked} if attacked else {"idx": i, "oom": True}
        append_jsonl(jsonl_path, record)
        done_by_idx[i] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


class QueryAttacker:
    """
    Gradient-free, query-based attack for T5. Positions are chosen uniformly
    at random (same policy as the random-substitution control); at each
    position, `candidates_per_position` candidate tokens are tried and the
    one that most increases the target model's own loss is kept greedily.
    No gradients are computed anywhere in this class. See the Pythia-track
    QueryAttacker (stage_6b_hotflip_transfer.py) for the rationale behind
    the query-budget choice.
    """

    def __init__(self, model, tokenizer, device, num_flips, candidates_per_position, candidate_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_flips = num_flips
        self.candidates_per_position = candidates_per_position
        self.candidate_tokens = candidate_tokens
        self.special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}

    def attack_single_example(self, text, summary, rng):
        input_ids, attention_mask = _tokenize_source(self.tokenizer, text, self.device)
        target_ids = _load_target_ids(self.tokenizer, summary, self.device)
        seq_len = input_ids.size(1)
        valid_positions = [
            i for i in range(1, seq_len)
            if attention_mask[0, i].item() == 1 and input_ids[0, i].item() not in self.special_ids
        ]
        if not valid_positions:
            return None

        clean_loss = _forward_loss(self.model, input_ids, attention_mask, target_ids)
        k = min(self.num_flips, len(valid_positions))
        positions = rng.choice(valid_positions, size=k, replace=False)
        flipped = input_ids.clone()
        current_loss = clean_loss

        for pos in positions:
            pos = int(pos)
            best_loss = current_loss
            best_token = flipped[0, pos].item()
            for _ in range(self.candidates_per_position):
                candidate = int(rng.choice(self.candidate_tokens))
                trial = flipped.clone()
                trial[0, pos] = candidate
                loss = _forward_loss(self.model, trial, attention_mask, target_ids)
                if loss > best_loss:
                    best_loss = loss
                    best_token = candidate
            flipped[0, pos] = best_token
            current_loss = best_loss

        attacked_loss = _forward_loss(self.model, flipped, attention_mask, target_ids)
        degradation = (attacked_loss - clean_loss) / clean_loss if clean_loss else 0.0
        return {
            'clean_loss': float(clean_loss),
            'attacked_loss': float(attacked_loss),
            'degradation': float(degradation),
            'success': bool(degradation > 0.1),
        }


def _query_attack_resumable(attacker, texts, summaries, jsonl_path, base_seed, logger):
    existing = load_jsonl(jsonl_path)
    done_by_idx = {r.get("idx"): r for r in existing if r.get("idx") is not None}
    if done_by_idx:
        logger.log(f"  Resuming query attack: {len(done_by_idx)}/{len(texts)} done.")

    for i in tqdm(range(len(texts)), initial=len(done_by_idx), total=len(texts), desc="Query attack"):
        if i in done_by_idx:
            continue
        rng = np.random.RandomState(base_seed + i)
        result = attacker.attack_single_example(texts[i], summaries[i], rng)
        record = {"idx": i, **result} if result else {"idx": i, "oom": True}
        append_jsonl(jsonl_path, record)
        done_by_idx[i] = record

    return [done_by_idx[i] for i in sorted(done_by_idx)]


def main():
    logger = StageLogger("stage_6b_hotflip_transfer")

    try:
        logger.log("Checking dependencies...")
        required = ['stage_0_setup', 'stage_1_data_prep',
                    'stage_2_train_baseline', 'stage_3_train_monotonic', 'stage_6_hotflip']
        if not check_dependencies(required):
            logger.complete(success=False)
            return 1

        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        device = ExperimentConfig.get_device()
        partial = partial_results_dir()

        logger.log("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(ExperimentConfig.MODEL_NAME)

        logger.log("Loading attack data...")
        attack_data = torch.load(
            os.path.join(ExperimentConfig.DATA_CACHE_DIR, 'attack_data.pt'), weights_only=False,
        )
        all_texts = attack_data['evaluation']['texts']
        all_summaries = attack_data['evaluation']['summaries']
        max_samples = ExperimentConfig.HOTFLIP_TRANSFER_NUM_SAMPLES
        if max_samples <= 0:
            max_samples = 200 if ExperimentConfig.USE_FULL_TEST_SETS else 100
        texts = all_texts[:max_samples]
        summaries = all_summaries[:max_samples]
        logger.log(f"  Attack samples: {len(texts)}")

        num_flips = ExperimentConfig.ATTACK_TRIGGER_LENGTH
        candidate_tokens = _get_candidate_vocab(tokenizer)
        logger.log(f"  Candidate vocabulary size: {len(candidate_tokens)}")

        # ------------------------------------------------------------
        # Step 1: craft (re-attack with gradients) on each source model
        # ------------------------------------------------------------
        craft_records = {}
        for source_name, source_type in MODEL_TYPES.items():
            craft_path = os.path.join(partial, f"hotflip_transfer_craft_{source_name}.jsonl")
            done = load_jsonl(craft_path)
            if len(done) >= len(texts):
                logger.log(f"\n[{source_name}] Craft already complete ({len(done)} examples), skipping load.")
                craft_records[source_name] = sorted(done, key=lambda r: r['idx'])
                continue

            logger.log(f"\n{'='*80}\nCRAFTING ATTACKED INPUTS ON: {source_name}\n{'='*80}")
            checkpoint_dir = (
                ExperimentConfig.BASELINE_CHECKPOINT_DIR if source_type == 'baseline'
                else ExperimentConfig.CHECKPOINT_DIR
            )
            checkpoint_path = os.path.join(checkpoint_dir, f'{source_type}_checkpoints', 'best_model.pt')
            model, _ = load_model(source_type, checkpoint_path=checkpoint_path, device=device)
            attacker = HotFlipT5Attack(model, tokenizer, device)
            craft_records[source_name] = _craft_resumable(
                attacker, texts, summaries, craft_path, logger, num_flips,
            )
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
            checkpoint_dir = (
                ExperimentConfig.BASELINE_CHECKPOINT_DIR if target_type == 'baseline'
                else ExperimentConfig.CHECKPOINT_DIR
            )
            checkpoint_path = os.path.join(checkpoint_dir, f'{target_type}_checkpoints', 'best_model.pt')
            model, _ = load_model(target_type, checkpoint_path=checkpoint_path, device=device)
            for src in MODEL_TYPES:
                summary_path = os.path.join(
                    partial, f"hotflip_transfer_eval_{src}_on_{target_name}_summary.json"
                )
                if os.path.exists(summary_path):
                    substitution_transfer[src][target_name] = load_json(summary_path)
                    continue
                eval_jsonl = os.path.join(partial, f"hotflip_transfer_eval_{src}_on_{target_name}.jsonl")
                label = f"{src}->{target_name}"
                records = _cross_eval_resumable(
                    model, tokenizer, craft_records[src], summaries, device, eval_jsonl, logger, label,
                )
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
        # Step 3: random-substitution control
        # ------------------------------------------------------------
        logger.log(f"\n{'='*80}\nRANDOM-SUBSTITUTION CONTROL\n{'='*80}")
        random_craft_path = os.path.join(partial, "hotflip_transfer_random_craft.jsonl")
        random_records = _random_control_resumable(
            tokenizer, texts, device, candidate_tokens, random_craft_path,
            base_seed=ExperimentConfig.CURRENT_SEED + 9999, num_flips=num_flips, logger=logger,
        )
        random_control = {}
        for target_name, target_type in MODEL_TYPES.items():
            summary_path = os.path.join(partial, f"hotflip_transfer_random_eval_{target_name}_summary.json")
            if os.path.exists(summary_path):
                random_control[target_name] = load_json(summary_path)
                continue
            logger.log(f"  Evaluating random control on {target_name}...")
            checkpoint_dir = (
                ExperimentConfig.BASELINE_CHECKPOINT_DIR if target_type == 'baseline'
                else ExperimentConfig.CHECKPOINT_DIR
            )
            checkpoint_path = os.path.join(checkpoint_dir, f'{target_type}_checkpoints', 'best_model.pt')
            model, _ = load_model(target_type, checkpoint_path=checkpoint_path, device=device)
            eval_jsonl = os.path.join(partial, f"hotflip_transfer_random_eval_{target_name}.jsonl")
            records = _cross_eval_resumable(
                model, tokenizer, random_records, summaries, device, eval_jsonl, logger,
                f"random->{target_name}",
            )
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
        if ExperimentConfig.RUN_QUERY_ATTACK:
            logger.log(f"\n{'='*80}\nQUERY-BASED (GRADIENT-FREE) ATTACK\n{'='*80}")
            n_query = min(ExperimentConfig.QUERY_ATTACK_NUM_SAMPLES, len(texts))
            query_texts = texts[:n_query]
            query_summaries = summaries[:n_query]
            query_attack = {}
            for target_name, target_type in MODEL_TYPES.items():
                summary_path = os.path.join(partial, f"hotflip_transfer_query_{target_name}_summary.json")
                if os.path.exists(summary_path):
                    query_attack[target_name] = load_json(summary_path)
                    continue
                logger.log(f"  Running query attack directly against {target_name}...")
                checkpoint_dir = (
                    ExperimentConfig.BASELINE_CHECKPOINT_DIR if target_type == 'baseline'
                    else ExperimentConfig.CHECKPOINT_DIR
                )
                checkpoint_path = os.path.join(checkpoint_dir, f'{target_type}_checkpoints', 'best_model.pt')
                model, _ = load_model(target_type, checkpoint_path=checkpoint_path, device=device)
                attacker = QueryAttacker(
                    model, tokenizer, device,
                    num_flips=num_flips,
                    candidates_per_position=ExperimentConfig.QUERY_ATTACK_CANDIDATES_PER_POSITION,
                    candidate_tokens=candidate_tokens,
                )
                query_jsonl = os.path.join(partial, f"hotflip_transfer_query_{target_name}.jsonl")
                base_seed = ExperimentConfig.CURRENT_SEED + (0 if target_type == 'baseline' else 7) + 5000
                records = _query_attack_resumable(
                    attacker, query_texts, query_summaries, query_jsonl, base_seed, logger,
                )
                summary = _aggregate(records)
                summary['evaluated_on'] = target_name
                summary['num_flips'] = num_flips
                summary['candidates_per_position'] = ExperimentConfig.QUERY_ATTACK_CANDIDATES_PER_POSITION
                atomic_save_json(summary, summary_path)
                query_attack[target_name] = summary
                logger.log(f"  [{target_name}] success_rate={summary['success_rate']*100:.1f}% "
                           f"avg_degradation={summary['avg_degradation']*100:.2f}%")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.log("\nSkipping query-based attack (ExperimentConfig.RUN_QUERY_ATTACK=False).")

        logger.log("\nSaving aggregated transfer/control results...")
        final_results = {
            'seed': ExperimentConfig.CURRENT_SEED,
            'ablation_mode': ExperimentConfig.T5_ABLATION_MODE,
            'attack_config': {
                'num_flips': num_flips,
                'num_samples': len(texts),
                'query_attack_num_samples': ExperimentConfig.QUERY_ATTACK_NUM_SAMPLES if ExperimentConfig.RUN_QUERY_ATTACK else 0,
                'query_attack_candidates_per_position': ExperimentConfig.QUERY_ATTACK_CANDIDATES_PER_POSITION,
            },
            'substitution_transfer': substitution_transfer,
            'random_control': random_control,
            'query_attack': query_attack,
        }
        save_json(final_results, os.path.join(ExperimentConfig.RESULTS_DIR, 'hotflip_transfer_results.json'))

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
