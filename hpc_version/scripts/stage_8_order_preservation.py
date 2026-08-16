#!/usr/bin/env python3
"""
Stage 8: End-to-End Order Preservation

Empirically tests the question the paper is named after: does the
coordinatewise, per-sublayer monotonicity enforced during training
(Definition "Monotone Transformer") induce order preservation along
interpretable *semantic* directions, end to end through the network?

Method:
1. Load the templated ordered-pair corpus (build_ordered_pairs.py), split
   into a probe-fit set and a disjoint evaluation set.
2. For baseline T5 and monotonic T5, extract encoder hidden states at every
   layer (embeddings + each of the 6 encoder layers) for every unique
   sentence in the corpus, pooled by mean over token positions (with
   final-token pooling computed too, as a robustness check).
3. Fit p=64 probe directions A (one per model) via logistic regression on
   the FIT split's final-layer representations -- bootstrap-resampled so
   the 64 directions are distinct-but-correlated readouts of the same
   underlying "stronger vs. weaker" signal, in the spirit of a concept-
   activation-vector ensemble. This is a diagnostic instrument only: A is
   fit post hoc and plays no role in training or in the monotonicity
   constraint itself (see paper Section 2 / Definition 3.2).
4. For every eval pair (weaker, stronger) and every layer, compute
   s = A h and report the fraction of the 64 coordinates where
   s_weaker <= s_stronger, with bootstrap 95% CIs, as a function of depth.
   A per-layer fraction near 1.0 means the network's semantic coordinates
   almost always increase along the expected direction at that depth; a
   fraction that degrades with depth would indicate order is being lost
   deeper in the network even though each individual FFN sublayer is
   coordinatewise monotone (since attention, LayerNorm, and residual mixing
   between sublayers are unconstrained).
5. Optional (best-effort, skipped gracefully if unavailable): correlates
   per-example probe-space disruption under HotFlip (using the persisted
   clean/attacked token ids from stage_6b_hotflip_transfer.py) with that
   example's HotFlip degradation, as a mechanistic cross-check.

Hardened for interrupted runs: hidden-state extraction is cached (one
atomic JSON per model); everything downstream is fast enough (a few
hundred pairs x 7 layers x 64-dim probes) that per-step resumability isn't
needed once the cache exists.

Inputs:
- ../data/ordered_pairs.json (from build_ordered_pairs.py)
- baseline_checkpoints/best_model.pt, monotonic_checkpoints/best_model.pt
- (optional) partial/hotflip_transfer_craft_{model}.jsonl from Stage 6b

Outputs:
- order_preservation_results.json
- order_preservation_depth.png (line plot, one curve per model)
- stage_8_order_preservation_complete.flag
"""

import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, check_dependencies, save_json, load_json, load_json_safe,
    atomic_save_json, load_jsonl, partial_results_dir, StageLogger, load_model,
)

from transformers import T5Tokenizer
from utils.order_preservation_math import (
    text_key as _text_key,
    collect_unique_texts as _collect_unique_texts,
    fit_logreg_direction as _fit_logreg_direction,
    fit_probe_directions,
    bootstrap_ci as _bootstrap_ci,
    measure_order_preservation,
    NUM_PROBE_DIRECTIONS,
    ORDER_EPS,
)

ORDERED_PAIRS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ordered_pairs.json')
MODEL_TYPES = {
    'baseline_t5': 'baseline',
    'monotonic_t5': 'monotonic',
}


def _checkpoint_path(model_type):
    checkpoint_dir = (
        ExperimentConfig.BASELINE_CHECKPOINT_DIR if model_type == 'baseline'
        else ExperimentConfig.CHECKPOINT_DIR
    )
    return os.path.join(checkpoint_dir, f'{model_type}_checkpoints', 'best_model.pt')


def _encode_text(tokenizer, text, device, max_len=64):
    enc = tokenizer(
        "summarize: " + text, return_tensors="pt", truncation=True, max_length=max_len,
    ).to(device)
    return enc.input_ids, enc.attention_mask


def _extract_hidden_states(model, tokenizer, text, device):
    """Encoder hidden states at every layer, pooled two ways."""
    input_ids, attention_mask = _encode_text(tokenizer, text, device)
    encoder = model.get_encoder()
    with torch.no_grad():
        out = encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
    mask = attention_mask[0].float()
    last_idx = max(int(mask.sum().item()) - 1, 0)

    mean_pooled, last_pooled = [], []
    for layer_h in out.hidden_states:
        h = layer_h[0]  # (seq_len, d)
        mean_h = (h * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(min=1)
        mean_pooled.append(mean_h.cpu().numpy().tolist())
        last_pooled.append(h[last_idx].cpu().numpy().tolist())
    return {'mean': mean_pooled, 'last': last_pooled}


def _hidden_states_from_ids(model, input_ids, attention_mask, device):
    """Same as _extract_hidden_states but from pre-tokenized ids (used for
    the optional HotFlip-disruption cross-check, which replays stage_6b's
    persisted attacked token sequences without re-tokenizing text)."""
    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device)
    encoder = model.get_encoder()
    with torch.no_grad():
        out = encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
    mask = attention_mask[0].float()
    h = out.hidden_states[-1][0]
    mean_h = (h * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(min=1)
    return mean_h.cpu().numpy()


def _build_hidden_cache(model, tokenizer, texts, device, cache_path, logger):
    cached = load_json_safe(cache_path, default=None)
    if cached is not None and cached.get('_text_count') == len(texts):
        logger.log(f"  Using cached hidden states from {cache_path}")
        return cached

    logger.log(f"  Extracting hidden states for {len(texts)} unique sentences...")
    cache = {'_text_count': len(texts)}
    for i, text in enumerate(texts):
        cache[_text_key(text)] = _extract_hidden_states(model, tokenizer, text, device)
        if (i + 1) % 100 == 0:
            logger.log(f"    {i + 1}/{len(texts)} done")
    atomic_save_json(cache, cache_path)
    return cache


def _plot_depth_curves(results_by_model, num_layers, out_path, logger):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.log("  matplotlib not available; skipping depth plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    layers = list(range(num_layers))
    colors = {'baseline_t5': 'tab:orange', 'monotonic_t5': 'tab:blue'}
    labels = {'baseline_t5': 'Baseline T5', 'monotonic_t5': 'Monotonic T5'}
    for model_name, per_layer in results_by_model.items():
        means = [per_layer[l]['mean'] for l in layers]
        los = [per_layer[l]['ci_low'] for l in layers]
        his = [per_layer[l]['ci_high'] for l in layers]
        ax.plot(layers, means, marker='o', label=labels.get(model_name, model_name),
                color=colors.get(model_name))
        ax.fill_between(layers, los, his, alpha=0.2, color=colors.get(model_name))

    ax.set_xlabel("Encoder depth (0 = embeddings, 6 = final layer)")
    ax.set_ylabel("Fraction of probe coordinates with $Ah \\leq Ah'$")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Order preservation vs. depth (held-out ordered pairs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.log(f"  Saved depth plot to {out_path}")


def _hotflip_correlation_check(models, tokenizer, device, partial, logger):
    """Optional, best-effort cross-check: does probe-space disruption under
    a HotFlip attack correlate with that attack's loss degradation? Skipped
    entirely if Stage 6b's craft records aren't available."""
    from scipy.stats import spearmanr

    correlations = {}
    for model_name, model_type in MODEL_TYPES.items():
        craft_path = os.path.join(partial, f"hotflip_transfer_craft_{model_name}.jsonl")
        records = load_jsonl(craft_path)
        if not records:
            logger.log(f"  [{model_name}] No Stage 6b craft records found; skipping HotFlip correlation.")
            continue

        model = models[model_name]
        disruptions, degradations = [], []
        for rec in records:
            if rec.get('oom', False):
                continue
            h_orig = _hidden_states_from_ids(model, rec['orig_ids'], rec['attention_mask'], device)
            h_attacked = _hidden_states_from_ids(model, rec['attacked_ids'], rec['attention_mask'], device)
            disruption = float(np.linalg.norm(h_attacked - h_orig))
            disruptions.append(disruption)
            degradations.append(rec['degradation'])

        if len(disruptions) < 5:
            logger.log(f"  [{model_name}] Too few valid records for correlation.")
            continue

        rho, p_value = spearmanr(disruptions, degradations)
        correlations[model_name] = {
            'spearman_rho': float(rho),
            'p_value': float(p_value),
            'num_examples': len(disruptions),
            'note': (
                'Correlates final-layer hidden-state displacement (clean vs. '
                'HotFlip-attacked, L2 norm) with per-example HotFlip loss '
                'degradation. Exploratory: HotFlip perturbations are not '
                'constructed to move along the probe-fit "stronger" '
                'direction, so this checks whether attack potency tracks '
                'representational disruption in general, not order '
                'violation specifically.'
            ),
        }
        logger.log(f"  [{model_name}] disruption vs. degradation: "
                   f"rho={rho:.3f}, p={p_value:.4f}, n={len(disruptions)}")
    return correlations


def main():
    logger = StageLogger("stage_8_order_preservation")

    try:
        logger.log("Checking dependencies...")
        required = ['stage_0_setup', 'stage_1_data_prep',
                    'stage_2_train_baseline', 'stage_3_train_monotonic']
        if not check_dependencies(required):
            logger.complete(success=False)
            return 1

        if not os.path.exists(ORDERED_PAIRS_PATH):
            raise FileNotFoundError(
                f"{ORDERED_PAIRS_PATH} not found. Run build_ordered_pairs.py first."
            )

        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        device = ExperimentConfig.get_device()
        partial = partial_results_dir()

        logger.log("Loading ordered-pairs corpus...")
        dataset = load_json(ORDERED_PAIRS_PATH)
        chains = dataset['chains']
        fit_pairs = [p for p in dataset['pairs'] if p['split'] == 'fit']
        eval_pairs = [p for p in dataset['pairs'] if p['split'] == 'eval']
        unique_texts = _collect_unique_texts(chains)
        logger.log(f"  {len(chains)} chains, {len(fit_pairs)} fit pairs, "
                   f"{len(eval_pairs)} eval pairs, {len(unique_texts)} unique sentences.")

        logger.log("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(ExperimentConfig.MODEL_NAME)

        models = {}
        hidden_caches = {}
        num_layers = None
        for model_name, model_type in MODEL_TYPES.items():
            logger.log(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
            model, _ = load_model(model_type, checkpoint_path=_checkpoint_path(model_type), device=device)
            models[model_name] = model
            cache_path = os.path.join(partial, f"order_hidden_states_{model_name}.json")
            hidden_caches[model_name] = _build_hidden_cache(
                model, tokenizer, unique_texts, device, cache_path, logger,
            )
            n_layers_this_model = len(next(
                v['mean'] for k, v in hidden_caches[model_name].items() if k != '_text_count'
            ))
            num_layers = n_layers_this_model if num_layers is None else num_layers
            assert n_layers_this_model == num_layers, "Encoder depth mismatch between models"

        results_by_model = {}
        pair_details_by_model = {}
        for pooling in ('mean', 'last'):
            logger.log(f"\n{'='*80}\nFITTING PROBES AND MEASURING ORDER PRESERVATION ({pooling}-pooling)\n{'='*80}")
            for model_name in MODEL_TYPES:
                logger.log(f"\n[{model_name}] Fitting {NUM_PROBE_DIRECTIONS} probe directions "
                           f"on the fit split (reference layer = {num_layers - 1})...")
                A = fit_probe_directions(
                    fit_pairs, hidden_caches[model_name], layer_idx=num_layers - 1,
                    pooling=pooling, num_directions=NUM_PROBE_DIRECTIONS,
                    seed=ExperimentConfig.CURRENT_SEED,
                )
                logger.log(f"[{model_name}] Measuring order preservation on {len(eval_pairs)} eval pairs...")
                per_layer_summary, per_pair_records = measure_order_preservation(
                    eval_pairs, hidden_caches[model_name], A, pooling, num_layers,
                )
                for layer in range(num_layers):
                    s = per_layer_summary[layer]
                    logger.log(f"    layer {layer}: {s['mean']*100:.1f}% "
                               f"[{s['ci_low']*100:.1f}, {s['ci_high']*100:.1f}]")
                if pooling == 'mean':
                    results_by_model[model_name] = per_layer_summary
                    pair_details_by_model[model_name] = per_pair_records
                else:
                    results_by_model.setdefault(f"{model_name}_last_token", per_layer_summary)

        plot_path = os.path.join(ExperimentConfig.RESULTS_DIR, 'order_preservation_depth.png')
        _plot_depth_curves(
            {k: v for k, v in results_by_model.items() if not k.endswith('_last_token')},
            num_layers, plot_path, logger,
        )

        logger.log(f"\n{'='*80}\nOPTIONAL: HOTFLIP DISRUPTION CORRELATION\n{'='*80}")
        try:
            hotflip_correlation = _hotflip_correlation_check(models, tokenizer, device, partial, logger)
        except Exception as e:
            logger.log(f"  Skipping HotFlip correlation check due to error: {e}")
            hotflip_correlation = {}

        final_results = {
            'seed': ExperimentConfig.CURRENT_SEED,
            'ablation_mode': ExperimentConfig.T5_ABLATION_MODE,
            'config': {
                'num_probe_directions': NUM_PROBE_DIRECTIONS,
                'probe_reference_layer': num_layers - 1,
                'num_encoder_layers': num_layers,
                'num_fit_pairs': len(fit_pairs),
                'num_eval_pairs': len(eval_pairs),
                'order_eps': ORDER_EPS,
            },
            'per_layer_order_preservation': results_by_model,
            'per_pair_details': pair_details_by_model,
            'hotflip_disruption_correlation': hotflip_correlation,
        }
        save_json(final_results, os.path.join(ExperimentConfig.RESULTS_DIR, 'order_preservation_results.json'))

        logger.log("\nOrder-preservation measurement complete.")
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
