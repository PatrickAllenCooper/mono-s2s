#!/usr/bin/env python3
"""
Stage 5: Universal Adversarial Trigger (UAT) Attacks for Foundation LLMs

Learns universal triggers that maximize perplexity when prepended to any input.
Adapted from T5 UAT implementation for decoder-only models.

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
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds, create_completion_flag, save_json,
    StageLogger, check_dependencies, make_model_monotonic
)

from transformers import AutoModelForCausalLM, AutoTokenizer


class UATOptimizer:
    """
    Universal Adversarial Trigger optimizer for causal LMs.
    Finds token sequences that maximize perplexity across examples.
    """
    
    def __init__(self, model, tokenizer, device, trigger_length=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.trigger_length = trigger_length
        self.vocab_size = len(tokenizer)
        
    def _get_candidate_tokens(self):
        """Get vocabulary of candidate tokens for trigger"""
        candidates = []
        
        # Rare tokens (high IDs)
        candidates.extend(list(range(1000, min(5000, self.vocab_size))))
        
        # Disruptive words
        disruptive_words = [
            'not', 'never', 'ignore', 'disregard', 'false', 'wrong',
            'error', 'invalid', 'corrupt', 'random', 'noise',
            '!!!', '???', '###', '***', '...', '---',
        ]
        
        for word in disruptive_words:
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            candidates.extend(token_ids)
        
        # Remove duplicates
        candidates = list(set(candidates))
        candidates = [c for c in candidates if 0 < c < self.vocab_size]
        
        return candidates[:Config.ATTACK_NUM_CANDIDATES]
    
    def compute_trigger_loss(self, trigger_ids, texts, batch_size=8):
        """
        Compute average loss when trigger is prepended to texts.
        Higher loss = more effective trigger.
        """
        self.model.eval()
        total_loss = 0.0
        num_examples = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Prepend trigger to each text
                triggered_texts = []
                trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
                for text in batch_texts:
                    triggered_texts.append(trigger_text + " " + text)
                
                # Tokenize
                encodings = self.tokenizer(
                    triggered_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_SEQ_LENGTH
                ).to(self.device)
                
                # Compute loss
                labels = encodings.input_ids.clone()
                labels[encodings.attention_mask == 0] = -100
                
                outputs = self.model(
                    input_ids=encodings.input_ids,
                    attention_mask=encodings.attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item() * len(batch_texts)
                num_examples += len(batch_texts)
        
        return total_loss / num_examples if num_examples > 0 else 0.0
    
    def optimize_trigger(self, texts, num_iterations=100, num_restarts=5):
        """
        Optimize trigger via coordinate ascent.
        Returns best trigger found across restarts.
        """
        candidate_tokens = self._get_candidate_tokens()
        best_trigger = None
        best_loss = float('-inf')
        
        for restart in range(num_restarts):
            print(f"\nRestart {restart + 1}/{num_restarts}")
            
            # Random initialization
            trigger_ids = np.random.choice(candidate_tokens, size=self.trigger_length).tolist()
            
            for iteration in range(num_iterations):
                current_loss = self.compute_trigger_loss(trigger_ids, texts)
                
                if iteration % 20 == 0:
                    trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
                    print(f"  Iter {iteration}: loss={current_loss:.4f}, trigger=\"{trigger_text}\"")
                
                # Try flipping each position
                improved = False
                for pos in range(self.trigger_length):
                    original_token = trigger_ids[pos]
                    best_pos_loss = current_loss
                    best_pos_token = original_token
                    
                    # Try random candidates
                    for _ in range(20):  # Try 20 random replacements per position
                        candidate = np.random.choice(candidate_tokens)
                        trigger_ids[pos] = candidate
                        
                        loss = self.compute_trigger_loss(trigger_ids, texts)
                        
                        if loss > best_pos_loss:
                            best_pos_loss = loss
                            best_pos_token = candidate
                            improved = True
                    
                    trigger_ids[pos] = best_pos_token
                    current_loss = best_pos_loss
                
                if not improved:
                    print(f"  Converged at iteration {iteration}")
                    break
            
            # Check if best overall
            final_loss = self.compute_trigger_loss(trigger_ids, texts)
            if final_loss > best_loss:
                best_loss = final_loss
                best_trigger = trigger_ids.copy()
        
        return best_trigger, best_loss


def evaluate_trigger(model, tokenizer, trigger_ids, texts, device):
    """Evaluate trigger impact on held-out texts"""
    model.eval()
    
    # Compute clean perplexity
    clean_encodings = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=Config.MAX_SEQ_LENGTH
    ).to(device)
    
    with torch.no_grad():
        clean_labels = clean_encodings.input_ids.clone()
        clean_labels[clean_encodings.attention_mask == 0] = -100
        clean_outputs = model(**clean_encodings, labels=clean_labels)
        clean_loss = clean_outputs.loss.item()
    
    # Compute attacked perplexity
    trigger_text = tokenizer.decode(trigger_ids, skip_special_tokens=True)
    attacked_texts = [trigger_text + " " + text for text in texts]
    
    attacked_encodings = tokenizer(
        attacked_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=Config.MAX_SEQ_LENGTH
    ).to(device)
    
    with torch.no_grad():
        attacked_labels = attacked_encodings.input_ids.clone()
        attacked_labels[attacked_encodings.attention_mask == 0] = -100
        attacked_outputs = model(**attacked_encodings, labels=attacked_labels)
        attacked_loss = attacked_outputs.loss.item()
    
    nll_increase = (attacked_loss - clean_loss) / clean_loss
    
    return {
        'trigger_text': trigger_text,
        'trigger_ids': trigger_ids,
        'clean_loss': clean_loss,
        'attacked_loss': attacked_loss,
        'nll_increase': nll_increase,
        'nll_increase_percent': nll_increase * 100,
    }


def main():
    """Run UAT attacks"""
    logger = StageLogger("stage_5_uat")
    
    try:
        # Check dependencies
        logger.log("Checking dependencies...")
        if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic']):
            logger.complete(success=False)
            return 1
        
        set_all_seeds(Config.CURRENT_SEED)
        device = Config.get_device()
        
        # Load tokenizer
        logger.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.DATA_CACHE_DIR
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load test data
        logger.log("Loading test data from Pile...")
        
        from datasets import load_dataset
        
        # Load Pile test/validation data
        try:
            pile_test = load_dataset(
                Config.TRAINING_DATASET,
                split="test",
                streaming=False,
                cache_dir=Config.DATA_CACHE_DIR,
                trust_remote_code=True
            )
        except (ValueError, KeyError):
            # Fallback to validation
            logger.log("  Test split not available, trying validation...")
            try:
                pile_test = load_dataset(
                    Config.TRAINING_DATASET,
                    split="validation",
                    streaming=False,
                    cache_dir=Config.DATA_CACHE_DIR,
                    trust_remote_code=True
                )
            except (ValueError, KeyError):
                # Final fallback: use end of train split as test set
                logger.log("  Using end of train split as test set...")
                pile_test = load_dataset(
                    Config.TRAINING_DATASET,
                    split="train[-10000:]",
                    streaming=False,
                    cache_dir=Config.DATA_CACHE_DIR,
                    trust_remote_code=True
                )
        
        # Collect texts
        max_samples = 1500 if Config.USE_FULL_EVAL_SETS else 300
        all_texts = [example['text'] for i, example in enumerate(pile_test) if i < max_samples]
        
        logger.log(f"  ✓ Loaded {len(all_texts)} test samples")
        
        # Split: optimization set and evaluation set
        split_idx = int(len(all_texts) * 0.4)  # 40% for optimization, 60% for evaluation
        opt_texts = all_texts[:split_idx]
        eval_texts = all_texts[split_idx:]
        
        logger.log(f"  Optimization set: {len(opt_texts)} samples")
        logger.log(f"  Evaluation set: {len(eval_texts)} samples")
        
        # Dictionary to store results
        all_results = {}
        
        # Process each model
        for model_name, model_type in [
            ('baseline_pythia', 'baseline'),
            ('monotonic_pythia', 'monotonic')
        ]:
            logger.log(f"\n{'='*80}")
            logger.log(f"OPTIMIZING TRIGGER FOR: {model_name}")
            logger.log(f"{'='*80}")
            
            # Load model
            logger.log("Loading model...")
            if model_type == 'baseline':
                checkpoint_path = os.path.join(
                    Config.CHECKPOINT_DIR,
                    'baseline_checkpoints',
                    'best_model.pt'
                )
                model = AutoModelForCausalLM.from_pretrained(
                    Config.MODEL_NAME,
                    cache_dir=Config.DATA_CACHE_DIR,
                    torch_dtype=torch.float32
                ).to(device)
                model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
            else:
                checkpoint_path = os.path.join(
                    Config.CHECKPOINT_DIR,
                    'monotonic_checkpoints',
                    'best_model.pt'
                )
                model = AutoModelForCausalLM.from_pretrained(
                    Config.MODEL_NAME,
                    cache_dir=Config.DATA_CACHE_DIR,
                    torch_dtype=torch.float32
                )
                model = make_model_monotonic(model)
                model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
                model = model.to(device)
            
            model.eval()
            logger.log(f"✓ Model loaded from: {checkpoint_path}")
            
            # Optimize trigger
            logger.log("\nOptimizing universal trigger...")
            logger.log(f"  Trigger length: {Config.ATTACK_TRIGGER_LENGTH}")
            logger.log(f"  Iterations: {Config.ATTACK_NUM_ITERATIONS}")
            logger.log(f"  Restarts: {Config.ATTACK_NUM_RESTARTS}")
            
            optimizer = UATOptimizer(
                model, tokenizer, device,
                trigger_length=Config.ATTACK_TRIGGER_LENGTH
            )
            
            best_trigger, best_loss = optimizer.optimize_trigger(
                opt_texts,
                num_iterations=Config.ATTACK_NUM_ITERATIONS,
                num_restarts=Config.ATTACK_NUM_RESTARTS
            )
            
            trigger_text = tokenizer.decode(best_trigger, skip_special_tokens=True)
            logger.log(f"\n✓ Trigger optimized!")
            logger.log(f"  Best trigger: \"{trigger_text}\"")
            logger.log(f"  Optimization loss: {best_loss:.4f}")
            
            # Evaluate on held-out set
            logger.log("\nEvaluating trigger on held-out set...")
            eval_results = evaluate_trigger(
                model, tokenizer, best_trigger,
                eval_texts, device
            )
            
            logger.log(f"  Clean loss: {eval_results['clean_loss']:.4f}")
            logger.log(f"  Attacked loss: {eval_results['attacked_loss']:.4f}")
            logger.log(f"  NLL increase: {eval_results['nll_increase_percent']:.2f}%")
            
            all_results[model_name] = eval_results
        
        # Save results
        logger.log("\nSaving UAT results...")
        results = {
            'seed': Config.CURRENT_SEED,
            'attack_config': {
                'trigger_length': Config.ATTACK_TRIGGER_LENGTH,
                'num_iterations': Config.ATTACK_NUM_ITERATIONS,
                'num_restarts': Config.ATTACK_NUM_RESTARTS,
                'num_opt_samples': len(opt_texts),
                'num_eval_samples': len(eval_texts),
            },
            'results': all_results
        }
        
        save_json(
            results,
            os.path.join(Config.RESULTS_DIR, 'uat_results.json')
        )
        
        # Save triggers to CSV
        import csv
        csv_path = os.path.join(Config.RESULTS_DIR, 'learned_triggers.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'trigger', 'nll_increase_percent'])
            for model_name, results in all_results.items():
                writer.writerow([
                    model_name,
                    results['trigger_text'],
                    f"{results['nll_increase_percent']:.2f}"
                ])
        
        logger.log(f"✓ Saved triggers to: {csv_path}")
        
        logger.log("\n✓ UAT attacks complete!")
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
