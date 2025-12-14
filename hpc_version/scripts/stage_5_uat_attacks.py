#!/usr/bin/env python3
"""
Stage 5: UAT (Universal Adversarial Trigger) Attacks + Transfer Matrix

This stage performs aggressive UAT attacks on all three models and evaluates
trigger transferability across models.

Attack Strategy:
- Learn universal triggers optimized for each model
- Evaluate on HELD-OUT test set (not used for trigger optimization)
- Test transfer attacks: trigger from Model A tested on Model B
- Compute ROUGE degradation and NLL increase

Models attacked:
1. Standard T5 (pre-trained) - Reference
2. Baseline T5 (fine-tuned, unconstrained) - Fair baseline
3. Monotonic T5 (fine-tuned, W≥0 FFN constraints) - Treatment

Data splits:
- Trigger optimization: CNN/DM validation split (500 samples)
- Attack evaluation: CNN/DM test split (1000 samples)

Inputs:
- attack_data.pt (from stage 1) - optimization and evaluation splits
- evaluation_results.json (from stage 4) - for baseline ROUGE
- All three model checkpoints

Outputs:
- uat_results.json (attack results with transfer matrix)
- learned_triggers.csv (trigger texts and IDs)
- stage_5_uat_complete.flag
"""

# Set environment variables BEFORE importing torch
import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rouge_score import rouge_scorer

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, create_completion_flag, check_dependencies,
    save_json, StageLogger, load_model
)

# Import transformers AFTER environment setup
from transformers import T5Tokenizer


class AggressiveUATAttack:
    """
    Enhanced UAT attack with multiple strategies for substantial impact.
    Learns universal triggers using gradient-based optimization.
    """
    def __init__(self, model, tokenizer, device, trigger_length=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.trigger_length = trigger_length or ExperimentConfig.ATTACK_TRIGGER_LENGTH
        
        # Expand vocabulary to include more disruptive tokens
        self.vocab_candidates = self._get_disruptive_vocab()
    
    def _insert_ids_prefix(self, input_ids, trigger_ids):
        """Insert trigger IDs at the beginning"""
        trig = torch.tensor(trigger_ids, device=input_ids.device).unsqueeze(0)
        return torch.cat([trig, input_ids], dim=1)
    
    def _encode_source(self, text):
        """Encode source text with 'summarize:' prefix"""
        enc = self.tokenizer("summarize: " + text, return_tensors="pt", truncation=False)
        return {k: v.to(self.device) for k, v in enc.items()}
    
    def _safe_pack(self, enc, max_len=512):
        """Truncate from left (trigger side) to keep source content.

        Supports both:
        - tensor batch: {"input_ids": (B, L), "attention_mask": (B, L)}
        - list batch:   {"input_ids": List[List[int]], "attention_mask": List[List[int]]}
        """
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        # Tensor path
        if torch.is_tensor(input_ids):
            if input_ids.size(1) > max_len:
                extra = input_ids.size(1) - max_len
                input_ids = input_ids[:, extra:]
                attn = attn[:, extra:]
            return {"input_ids": input_ids, "attention_mask": attn}

        # List path (ragged)
        packed_ids, packed_attn = [], []
        for ids_i, attn_i in zip(input_ids, attn):
            if len(ids_i) > max_len:
                ids_i = ids_i[-max_len:]
                attn_i = attn_i[-max_len:]
            packed_ids.append(ids_i)
            packed_attn.append(attn_i)
        return {"input_ids": packed_ids, "attention_mask": packed_attn}
    
    def _get_disruptive_vocab(self):
        """Get vocabulary likely to disrupt the model"""
        candidates = []
        
        # Rare/unusual words (high token IDs)
        vocab_size = self.tokenizer.vocab_size
        candidates.extend(list(range(1000, min(5000, vocab_size))))
        
        # Common confusing words
        disruptive_words = [
            'not', 'never', 'always', 'must', 'cannot', 'should', 'would',
            'however', 'although', 'despite', 'nevertheless',
            'ignore', 'disregard', 'false', 'incorrect', 'error',
            'random', 'noise', 'gibberish', 'nonsense',
            '!!!', '???', '###', '***',
        ]
        
        for word in disruptive_words:
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            candidates.extend(token_ids)
        
        # Numbers and special tokens
        for num in ['0', '1', '999', '2024', '####']:
            token_ids = self.tokenizer.encode(num, add_special_tokens=False)
            candidates.extend(token_ids)
        
        # Remove duplicates and invalid tokens
        candidates = list(set(candidates))
        candidates = [c for c in candidates if 0 < c < vocab_size]
        
        return candidates
    
    def compute_loss_batch(self, texts, summaries, trigger_ids, max_src_len=512):
        """
        Compute average per-example NLL of reference summaries given (trigger ⊕ source).

        This is a true batched implementation (with micro-batching) for speed.
        """
        self.model.eval()

        bs = int(getattr(ExperimentConfig, "ATTACK_LOSS_BATCH_SIZE", 8))
        all_losses = []

        # Tokenize targets once per micro-batch; labels use -100 on padding.
        pad_id = self.tokenizer.pad_token_id

        for start in range(0, len(texts), bs):
            batch_texts = texts[start:start + bs]
            batch_summaries = summaries[start:start + bs]

            # Ragged encode sources without truncation; we'll do left-truncation after trigger insert.
            src = self.tokenizer(
                ["summarize: " + t for t in batch_texts],
                padding=False,
                truncation=False
            )

            if trigger_ids is not None and len(trigger_ids) > 0:
                trig = list(map(int, trigger_ids))
                src["input_ids"] = [trig + ids for ids in src["input_ids"]]
                src["attention_mask"] = [[1] * len(trig) + m for m in src["attention_mask"]]

            src = self._safe_pack(src, max_len=max_src_len)

            # Pad to tensor batch
            src_t = self.tokenizer.pad(src, padding=True, return_tensors="pt").to(self.device)

            # Targets
            tgt = self.tokenizer(
                batch_summaries,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            labels = tgt["input_ids"].clone()
            labels[labels == pad_id] = -100

            with torch.no_grad():
                out = self.model(
                    input_ids=src_t["input_ids"],
                    attention_mask=src_t["attention_mask"],
                    labels=labels
                )

                # Per-example loss (mean over non-pad target tokens)
                # logits: (B, T, V), labels: (B, T)
                logits = out.logits
                token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                    ignore_index=-100
                ).view(labels.size(0), labels.size(1))

                denom = (labels != -100).sum(dim=1).clamp(min=1)
                ex_loss = token_loss.sum(dim=1) / denom
                all_losses.extend(ex_loss.detach().cpu().tolist())

        return float(np.mean(all_losses)) if all_losses else 0.0, all_losses
    
    def learn_universal_trigger(self, texts, summaries, num_iterations=50, num_restarts=3):
        """
        Learn universal trigger with multiple random restarts.
        """
        print(f"\nLearning AGGRESSIVE universal trigger...")
        print(f"  Trigger length: {self.trigger_length} tokens")
        print(f"  Training samples: {len(texts)}")
        print(f"  Optimization runs: {num_iterations} iterations × {num_restarts} restarts")
        
        best_global_trigger = None
        best_global_loss = -float('inf')
        
        for restart in range(num_restarts):
            print(f"\n--- Random Restart {restart + 1}/{num_restarts} ---")
            
            # Initialize random trigger
            trigger_ids = np.random.choice(self.vocab_candidates, self.trigger_length)
            best_trigger = trigger_ids.copy()
            best_avg_loss = -float('inf')
            
            # Compute baseline (no trigger)
            baseline_loss, _ = self.compute_loss_batch(texts, summaries, [])
            print(f"  Baseline loss (clean): {baseline_loss:.4f}")
            
            progress_bar = tqdm(range(num_iterations), desc=f"Restart {restart+1}")
            
            for iteration in progress_bar:
                # Evaluate current trigger
                avg_loss, _ = self.compute_loss_batch(texts, summaries, trigger_ids)
                
                # Update best for this restart
                if avg_loss > best_avg_loss:
                    best_avg_loss = avg_loss
                    best_trigger = trigger_ids.copy()
                
                improvement = (avg_loss / baseline_loss - 1) * 100 if baseline_loss > 0 else 0
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'best': f'{best_avg_loss:.3f}',
                    'impact': f'{improvement:+.1f}%'
                })
                
                # AGGRESSIVE local search: try replacing multiple positions
                positions_to_change = np.random.choice(
                    self.trigger_length,
                    size=min(3, self.trigger_length),
                    replace=False
                )
                
                for pos in positions_to_change:
                    original_token = trigger_ids[pos]
                    
                    # Try more candidates per position
                    candidates = np.random.choice(
                        self.vocab_candidates,
                        min(50, len(self.vocab_candidates)),
                        replace=False
                    )
                    
                    best_local_loss = avg_loss
                    best_local_token = original_token
                    
                    # Evaluate on larger subset
                    subset_size = min(30, len(texts))
                    subset_indices = np.random.choice(len(texts), subset_size, replace=False)
                    
                    for candidate in candidates:
                        trigger_ids[pos] = candidate
                        
                        subset_texts = [texts[i] for i in subset_indices]
                        subset_summaries = [summaries[i] for i in subset_indices]
                        
                        subset_loss, _ = self.compute_loss_batch(
                            subset_texts, subset_summaries, trigger_ids
                        )
                        
                        if subset_loss > best_local_loss:
                            best_local_loss = subset_loss
                            best_local_token = candidate
                    
                    trigger_ids[pos] = best_local_token
                
                # Early stopping if impact is substantial
                if iteration > 10 and improvement >= 15:
                    print(f"\n  ✓ Achieved {improvement:.1f}% impact - stopping early")
                    break
            
            # Update global best across all restarts
            if best_avg_loss > best_global_loss:
                best_global_loss = best_avg_loss
                best_global_trigger = best_trigger.copy()
                print(f"  New global best: {best_avg_loss:.4f} ({(best_avg_loss/baseline_loss-1)*100:+.1f}% impact)")
        
        trigger_text = self.tokenizer.decode(best_global_trigger, skip_special_tokens=True)
        final_impact = (best_global_loss / baseline_loss - 1) * 100 if baseline_loss > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"✓ BEST Universal Trigger Found:")
        print(f"  '{trigger_text}'")
        print(f"  Baseline loss: {baseline_loss:.4f}")
        print(f"  Attacked loss: {best_global_loss:.4f}")
        print(f"  Impact: {final_impact:+.1f}%")
        print(f"{'='*80}")
        
        return best_global_trigger, trigger_text
    
    def evaluate_trigger(self, texts, summaries, trigger_ids):
        """
        Evaluate trigger on test set using ROUGE deltas and NLL increase.
        """
        # Primary: ROUGE deltas
        rouge_report = self.eval_generation(texts, summaries, trigger_ids=trigger_ids)
        
        # Auxiliary: NLL deltas
        avg_clean_loss, _ = self.compute_loss_batch(texts, summaries, [], max_src_len=512)
        avg_attack_loss, _ = self.compute_loss_batch(texts, summaries, trigger_ids, max_src_len=512)
        loss_deg = (avg_attack_loss - avg_clean_loss) / max(avg_clean_loss, 1e-8)
        
        return {
            "rouge": rouge_report,
            "aux_ce": {
                "avg_clean_loss": avg_clean_loss,
                "avg_attack_loss": avg_attack_loss,
                "relative_increase": loss_deg
            }
        }
    
    def eval_generation(self, texts, refs, trigger_ids=None, max_src_len=512):
        """
        Generate summaries with/without trigger and compute ROUGE deltas.
        Uses FIXED decoding parameters from config for fair comparison.
        """
        gen_kwargs = dict(
            max_new_tokens=ExperimentConfig.DECODE_MAX_NEW_TOKENS,
            min_new_tokens=ExperimentConfig.DECODE_MIN_NEW_TOKENS,
            num_beams=ExperimentConfig.DECODE_NUM_BEAMS,
            length_penalty=ExperimentConfig.DECODE_LENGTH_PENALTY,
            no_repeat_ngram_size=ExperimentConfig.DECODE_NO_REPEAT_NGRAM_SIZE,
            early_stopping=ExperimentConfig.DECODE_EARLY_STOPPING
        )
        
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
        
        def encode(text):
            enc = self._encode_source(text)
            if trigger_ids is not None and len(trigger_ids) > 0:
                enc["input_ids"] = self._insert_ids_prefix(enc["input_ids"], trigger_ids)
                enc["attention_mask"] = torch.cat(
                    [torch.ones(1, len(trigger_ids), device=self.device), enc["attention_mask"]],
                    dim=1
                )
            return self._safe_pack(enc, max_len=max_src_len)
        
        clean_scores, atk_scores = [], []
        clean_outs, atk_outs = [], []
        
        for x, y_ref in tqdm(zip(texts, refs), total=len(texts), desc="Evaluating attack"):
            with torch.no_grad():
                # Clean
                enc_clean = self._safe_pack(self._encode_source(x), max_len=max_src_len)
                gen_clean = self.model.generate(**enc_clean, **gen_kwargs)
                y_clean = self.tokenizer.decode(gen_clean[0], skip_special_tokens=True)
                
                # Attacked
                enc_atk = encode(x)
                gen_atk = self.model.generate(**enc_atk, **gen_kwargs)
                y_atk = self.tokenizer.decode(gen_atk[0], skip_special_tokens=True)
            
            clean_outs.append(y_clean)
            atk_outs.append(y_atk)
            
            sc_clean = scorer.score(y_ref, y_clean)
            sc_atk = scorer.score(y_ref, y_atk)
            
            clean_scores.append({k: v.fmeasure for k, v in sc_clean.items()})
            atk_scores.append({k: v.fmeasure for k, v in sc_atk.items()})
        
        def mean(lst, key):
            return float(np.mean([d[key] for d in lst]))
        
        report = {
            "clean_means": {k: mean(clean_scores, k) for k in ["rouge1", "rouge2", "rougeLsum"]},
            "attack_means": {k: mean(atk_scores, k) for k in ["rouge1", "rouge2", "rougeLsum"]},
            "delta": {k: mean(atk_scores, k) - mean(clean_scores, k) for k in ["rouge1", "rouge2", "rougeLsum"]},
        }
        return report


def main():
    """Run UAT attacks with transfer matrix"""
    logger = StageLogger("stage_5_uat")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        required = ['stage_0_setup', 'stage_1_data_prep', 
                   'stage_2_train_baseline', 'stage_3_train_monotonic', 'stage_4_evaluate']
        if not check_dependencies(required):
            logger.complete(success=False)
            return 1
        
        # Set seeds
        logger.log("Setting random seeds...")
        set_all_seeds(ExperimentConfig.CURRENT_SEED)
        
        # Get device
        device = ExperimentConfig.get_device()
        
        # Load tokenizer
        logger.log("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(ExperimentConfig.MODEL_NAME)
        
        # Load attack data
        logger.log("Loading attack data...")
        data_cache_dir = ExperimentConfig.DATA_CACHE_DIR
        attack_data = torch.load(os.path.join(data_cache_dir, 'attack_data.pt'))
        
        trigger_opt_texts = attack_data['optimization']['texts']
        trigger_opt_summaries = attack_data['optimization']['summaries']
        trigger_eval_texts = attack_data['evaluation']['texts']
        trigger_eval_summaries = attack_data['evaluation']['summaries']
        
        logger.log(f"  Trigger optimization: {len(trigger_opt_texts)} samples")
        logger.log(f"  Trigger evaluation: {len(trigger_eval_texts)} samples")
        
        # Load models
        logger.log("\n" + "="*80)
        logger.log("LOADING MODELS")
        logger.log("="*80)
        
        models = {}
        
        # Standard T5
        logger.log("\n1. Loading Standard T5...")
        models['standard_t5'], _ = load_model('standard', checkpoint_path=None, device=device)
        
        # Baseline T5
        logger.log("\n2. Loading Baseline T5...")
        baseline_checkpoint = os.path.join(
            ExperimentConfig.CHECKPOINT_DIR, 'baseline_checkpoints', 'best_model.pt'
        )
        models['baseline_t5'], _ = load_model('baseline', checkpoint_path=baseline_checkpoint, device=device)
        
        # Monotonic T5
        logger.log("\n3. Loading Monotonic T5...")
        monotonic_checkpoint = os.path.join(
            ExperimentConfig.CHECKPOINT_DIR, 'monotonic_checkpoints', 'best_model.pt'
        )
        models['monotonic_t5'], _ = load_model('monotonic', checkpoint_path=monotonic_checkpoint, device=device)
        
        logger.log("\n✓ All models loaded")
        
        # Learn triggers for each model
        logger.log("\n" + "="*80)
        logger.log("LEARNING UNIVERSAL TRIGGERS")
        logger.log("="*80)
        
        triggers = {}
        all_results = {}
        
        for model_key, model_name in [
            ('standard_t5', 'Standard T5'),
            ('baseline_t5', 'Baseline T5'),
            ('monotonic_t5', 'Monotonic T5')
        ]:
            logger.log(f"\n{'='*80}")
            logger.log(f"ATTACKING: {model_name}")
            logger.log(f"{'='*80}")
            
            model = models[model_key]
            attacker = AggressiveUATAttack(model, tokenizer, device)
            
            # Learn trigger
            trigger_ids, trigger_text = attacker.learn_universal_trigger(
                trigger_opt_texts,
                trigger_opt_summaries,
                num_iterations=ExperimentConfig.ATTACK_NUM_ITERATIONS,
                num_restarts=ExperimentConfig.ATTACK_NUM_RESTARTS
            )
            
            triggers[model_key] = {
                'ids': trigger_ids,
                'text': trigger_text
            }
            
            # Evaluate on test set
            logger.log(f"\nEvaluating on test set ({len(trigger_eval_texts)} samples)...")
            test_results = attacker.evaluate_trigger(
                trigger_eval_texts,
                trigger_eval_summaries,
                trigger_ids
            )
            
            rouge_rep = test_results["rouge"]
            aux_ce = test_results["aux_ce"]
            
            logger.log(f"\nTest Set Results:")
            logger.log(f"  Trigger: '{trigger_text}'")
            logger.log(f"  ROUGE (clean): R1={rouge_rep['clean_means']['rouge1']:.3f} "
                      f"R2={rouge_rep['clean_means']['rouge2']:.3f} RL={rouge_rep['clean_means']['rougeLsum']:.3f}")
            logger.log(f"  ROUGE (attack): R1={rouge_rep['attack_means']['rouge1']:.3f} "
                      f"R2={rouge_rep['attack_means']['rouge2']:.3f} RL={rouge_rep['attack_means']['rougeLsum']:.3f}")
            logger.log(f"  ΔROUGE: ΔR1={rouge_rep['delta']['rouge1']:.3f} "
                      f"ΔR2={rouge_rep['delta']['rouge2']:.3f} ΔRL={rouge_rep['delta']['rougeLsum']:.3f}")
            logger.log(f"  NLL clean: {aux_ce['avg_clean_loss']:.4f}")
            logger.log(f"  NLL attack: {aux_ce['avg_attack_loss']:.4f}")
            logger.log(f"  NLL increase: {aux_ce['relative_increase']*100:+.1f}%")
            
            all_results[model_key] = {
                'trigger_text': trigger_text,
                'trigger_ids': trigger_ids.tolist(),
                'rouge_deltas': rouge_rep['delta'],
                'nll_increase': aux_ce['relative_increase'],
                'clean_rouge': rouge_rep['clean_means'],
                'attack_rouge': rouge_rep['attack_means']
            }
        
        # Transfer Matrix: Test each trigger on each model
        logger.log("\n" + "="*80)
        logger.log("TRANSFER ATTACK MATRIX")
        logger.log("="*80)
        logger.log("Testing each trigger on each model...")
        
        transfer_matrix = {}
        
        for source_model_key, source_model_name in [
            ('standard_t5', 'Standard T5'),
            ('baseline_t5', 'Baseline T5'),
            ('monotonic_t5', 'Monotonic T5')
        ]:
            transfer_matrix[source_model_key] = {}
            trigger_ids = triggers[source_model_key]['ids']
            trigger_text = triggers[source_model_key]['text']
            
            logger.log(f"\nTrigger from {source_model_name}: '{trigger_text}'")
            
            for target_model_key, target_model_name in [
                ('standard_t5', 'Standard T5'),
                ('baseline_t5', 'Baseline T5'),
                ('monotonic_t5', 'Monotonic T5')
            ]:
                model = models[target_model_key]
                attacker = AggressiveUATAttack(model, tokenizer, device)
                
                # Evaluate trigger on target model
                results = attacker.evaluate_trigger(
                    trigger_eval_texts[:100],  # Subset for transfer matrix
                    trigger_eval_summaries[:100],
                    trigger_ids
                )
                
                rouge_delta = results["rouge"]["delta"]["rougeLsum"]
                transfer_matrix[source_model_key][target_model_key] = rouge_delta
                
                marker = "★" if source_model_key == target_model_key else " "
                logger.log(f"  → {target_model_name}: ΔROUGE-L = {rouge_delta:.3f} {marker}")
        
        # Save results
        logger.log("\n" + "="*80)
        logger.log("SAVING RESULTS")
        logger.log("="*80)
        
        output = {
            'seed': ExperimentConfig.CURRENT_SEED,
            'attack_config': {
                'trigger_length': ExperimentConfig.ATTACK_TRIGGER_LENGTH,
                'num_iterations': ExperimentConfig.ATTACK_NUM_ITERATIONS,
                'num_restarts': ExperimentConfig.ATTACK_NUM_RESTARTS,
                'trigger_opt_samples': len(trigger_opt_texts),
                'trigger_eval_samples': len(trigger_eval_texts),
            },
            'learned_triggers': {k: v['trigger_text'] for k, v in triggers.items()},
            'results': all_results,
            'transfer_matrix': transfer_matrix
        }
        
        results_file = os.path.join(ExperimentConfig.RESULTS_DIR, 'uat_results.json')
        save_json(output, results_file)
        
        # Save triggers to CSV
        import pandas as pd
        triggers_csv = os.path.join(ExperimentConfig.RESULTS_DIR, 'learned_triggers.csv')
        trigger_df = pd.DataFrame([
            {
                'model_name': model_key,
                'trigger_text': triggers[model_key]['text'],
                'trigger_ids': json.dumps(triggers[model_key]['ids'].tolist())
            }
            for model_key in triggers
        ])
        trigger_df.to_csv(triggers_csv, index=False)
        logger.log(f"✓ Saved triggers to: {triggers_csv}")
        
        logger.log(f"\n✓ UAT attack evaluation complete!")
        logger.complete(success=True)
        return 0
        
    except Exception as e:
        logger.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.complete(success=False)
        return 1


if __name__ == "__main__":
    exit(main())

