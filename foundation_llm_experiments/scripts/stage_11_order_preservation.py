#!/usr/bin/env python3
"""
Stage 11: End-to-End Order Preservation (Pythia / decoder-only)

Ports the T5 Stage 8 probe-fit / Ah <= Ah' measurement onto decoder
residual streams so the title claim is substantiated on both
architectures. Probe math is imported from the T5 stage (no A is used
in training; this is a post-hoc instrument).

Primary pooling is last-token (the causal-LM readout). Mean pooling is
retained as a robustness check, matching the T5 protocol in reverse.
"""

import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, check_dependencies, save_json, load_json, load_json_safe,
    atomic_save_json, partial_results_dir, StageLogger, make_model_monotonic,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util
_MATH_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'hpc_version', 'utils', 'order_preservation_math.py'
)
_spec = importlib.util.spec_from_file_location("order_preservation_math", _MATH_PATH)
_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)
_text_key = _math.text_key
_collect_unique_texts = _math.collect_unique_texts
fit_probe_directions = _math.fit_probe_directions
measure_order_preservation = _math.measure_order_preservation
NUM_PROBE_DIRECTIONS = _math.NUM_PROBE_DIRECTIONS
ORDER_EPS = _math.ORDER_EPS

ORDERED_PAIRS_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'hpc_version', 'data', 'ordered_pairs.json'
)
MODEL_TYPES = {
    'baseline_pythia': 'baseline',
    'monotonic_pythia': 'monotonic',
}


def _checkpoint_path(model_type):
    return os.path.join(Config.CHECKPOINT_DIR, f'{model_type}_checkpoints', 'best_model.pt')


def _load_model(model_type, device):
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME, cache_dir=Config.DATA_CACHE_DIR, torch_dtype=dtype,
    )
    if model_type == 'monotonic':
        model = make_model_monotonic(model, variant=Config.MONOTONIC_VARIANT)
    path = _checkpoint_path(model_type)
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _extract_hidden_states(model, tokenizer, text, device):
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=64,
    ).to(device)
    input_ids, attention_mask = enc.input_ids, enc.attention_mask
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    mask = attention_mask[0].float()
    last_idx = max(int(mask.sum().item()) - 1, 0)

    mean_pooled, last_pooled = [], []
    for layer_h in out.hidden_states:
        h = layer_h[0]
        mean_h = (h * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(min=1)
        mean_pooled.append(mean_h.float().cpu().numpy().tolist())
        last_pooled.append(h[last_idx].float().cpu().numpy().tolist())
    return {'mean': mean_pooled, 'last': last_pooled}


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
    colors = {'baseline_pythia': 'tab:orange', 'monotonic_pythia': 'tab:blue'}
    labels = {'baseline_pythia': 'Baseline Pythia', 'monotonic_pythia': 'Monotonic Pythia'}
    for model_name, per_layer in results_by_model.items():
        means = [per_layer[l]['mean'] for l in layers]
        los = [per_layer[l]['ci_low'] for l in layers]
        his = [per_layer[l]['ci_high'] for l in layers]
        ax.plot(layers, means, marker='o', label=labels.get(model_name, model_name),
                color=colors.get(model_name))
        ax.fill_between(layers, los, his, alpha=0.2, color=colors.get(model_name))

    ax.set_xlabel("Decoder depth (0 = embeddings)")
    ax.set_ylabel("Fraction of probe coordinates with Ah <= Ah'")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Order preservation vs. depth (Pythia, last-token pooling)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.log(f"  Saved depth plot to {out_path}")


def main():
    logger = StageLogger("stage_11_order_preservation")

    try:
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic']):
            logger.complete(success=False)
            return 1

        if not os.path.exists(ORDERED_PAIRS_PATH):
            raise FileNotFoundError(
                f"{ORDERED_PAIRS_PATH} not found. Run hpc_version/scripts/build_ordered_pairs.py first."
            )

        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()
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
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, cache_dir=Config.DATA_CACHE_DIR,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        hidden_caches = {}
        num_layers = None
        for model_name, model_type in MODEL_TYPES.items():
            logger.log(f"\nMODEL: {model_name}")
            model = _load_model(model_type, device)
            cache_path = os.path.join(partial, f"order_hidden_states_{model_name}.json")
            hidden_caches[model_name] = _build_hidden_cache(
                model, tokenizer, unique_texts, device, cache_path, logger,
            )
            n_layers_this_model = len(next(
                v['mean'] for k, v in hidden_caches[model_name].items() if k != '_text_count'
            ))
            num_layers = n_layers_this_model if num_layers is None else num_layers
            assert n_layers_this_model == num_layers, "Decoder depth mismatch between models"
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_by_model = {}
        pair_details_by_model = {}
        for pooling in ('last', 'mean'):
            logger.log(f"\nFITTING PROBES ({pooling}-pooling)")
            for model_name in MODEL_TYPES:
                logger.log(f"[{model_name}] Fitting {NUM_PROBE_DIRECTIONS} probe directions "
                           f"on the fit split (reference layer = {num_layers - 1})...")
                A = fit_probe_directions(
                    fit_pairs, hidden_caches[model_name], layer_idx=num_layers - 1,
                    pooling=pooling, num_directions=NUM_PROBE_DIRECTIONS,
                    seed=Config.CURRENT_SEED,
                )
                per_layer_summary, per_pair_records = measure_order_preservation(
                    eval_pairs, hidden_caches[model_name], A, pooling, num_layers,
                )
                for layer in range(num_layers):
                    s = per_layer_summary[layer]
                    logger.log(f"    layer {layer}: {s['mean']*100:.1f}% "
                               f"[{s['ci_low']*100:.1f}, {s['ci_high']*100:.1f}]")
                if pooling == 'last':
                    results_by_model[model_name] = per_layer_summary
                    pair_details_by_model[model_name] = per_pair_records
                else:
                    results_by_model[f"{model_name}_mean_pool"] = per_layer_summary

        plot_path = os.path.join(Config.RESULTS_DIR, 'order_preservation_depth.png')
        _plot_depth_curves(
            {k: v for k, v in results_by_model.items() if not k.endswith('_mean_pool')},
            num_layers, plot_path, logger,
        )

        final_results = {
            'seed': Config.CURRENT_SEED,
            'model_name': Config.MODEL_NAME,
            'monotonic_variant': Config.MONOTONIC_VARIANT,
            'config': {
                'num_probe_directions': NUM_PROBE_DIRECTIONS,
                'probe_reference_layer': num_layers - 1,
                'num_decoder_layers': num_layers,
                'num_fit_pairs': len(fit_pairs),
                'num_eval_pairs': len(eval_pairs),
                'order_eps': ORDER_EPS,
                'primary_pooling': 'last',
            },
            'per_layer_order_preservation': results_by_model,
            'per_pair_details': pair_details_by_model,
        }
        save_json(final_results, os.path.join(Config.RESULTS_DIR, 'order_preservation_results.json'))

        logger.log("Order-preservation measurement complete.")
        logger.complete(success=True)
        return 0

    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
