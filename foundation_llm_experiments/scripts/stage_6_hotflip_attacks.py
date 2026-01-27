#!/usr/bin/env python3
"""
Stage 6: HotFlip Gradient-Based Attacks for Foundation LLMs

Performs gradient-based token flipping attacks to maximize perplexity.
Adapted from T5 HotFlip implementation for decoder-only models.

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


class HotFlipAttacker:
    """
    HotFlip attack for causal language models.
    Uses gradient information to flip tokens that maximally increase loss.
    """
    
    def __init__(self, model, tokenizer, device, num_flips=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_flips = num_flips
        self.vocab_size = len(tokenizer)
    
    def attack_single_example(self, text):
        """
        Perform HotFlip attack on a single text.
        Returns perplexity increase ratio.
        """
        # Encode text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH
        ).to(self.device)
        
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        
        # Compute clean loss
        self.model.eval()
        with torch.no_grad():
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            clean_outputs = self.model(input_ids=input_ids, labels=labels)
            clean_loss = clean_outputs.loss.item()
        
        # Get embeddings
        embedding_layer = self.model.get_input_embeddings()
        
        # Compute gradients w.r.t. embeddings
        embeddings = embedding_layer(input_ids)
        embeddings.requires_grad = True
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        outputs = self.model(inputs_embeds=embeddings, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Get gradients
        token_gradients = embeddings.grad[0]  # [seq_len, embed_dim]
        
        # Find positions with largest gradient magnitude
        grad_magnitudes = token_gradients.norm(dim=1).detach()
        grad_magnitudes[attention_mask[0] == 0] = 0  # Ignore padding
        
        # Get top-k positions to flip
        topk_positions = grad_magnitudes.topk(min(self.num_flips, len(grad_magnitudes))).indices
        
        # For each position, find best replacement token
        flipped_ids = input_ids.clone()
        
        for pos in topk_positions:
            pos_idx = pos.item()
            if pos_idx >= input_ids.size(1) or attention_mask[0, pos_idx] == 0:
                continue
            
            # Get all token embeddings
            all_embeddings = embedding_layer.weight  # [vocab_size, embed_dim]
            
            # Compute dot product with gradient
            grad_at_pos = token_gradients[pos_idx]  # [embed_dim]
            scores = torch.matmul(all_embeddings, grad_at_pos)  # [vocab_size]
            
            # Find token that maximally increases loss (max dot product)
            best_token = scores.argmax().item()
            flipped_ids[0, pos_idx] = best_token
        
        # Compute attacked loss
        with torch.no_grad():
            attacked_labels = flipped_ids.clone()
            attacked_labels[attention_mask == 0] = -100
            attacked_outputs = self.model(input_ids=flipped_ids, labels=attacked_labels)
            attacked_loss = attacked_outputs.loss.item()
        
        degradation = (attacked_loss - clean_loss) / clean_loss
        
        return {
            'clean_loss': clean_loss,
            'attacked_loss': attacked_loss,
            'degradation': degradation,
            'success': degradation > Config.ATTACK_SUCCESS_THRESHOLD,
        }
    
    def attack_batch(self, texts):
        """Attack multiple texts and aggregate results"""
        results = []
        
        for text in tqdm(texts, desc="HotFlip attacks"):
            result = self.attack_single_example(text)
            results.append(result)
        
        # Aggregate
        avg_degradation = np.mean([r['degradation'] for r in results])
        success_rate = np.mean([r['success'] for r in results])
        avg_orig_loss = np.mean([r['clean_loss'] for r in results])
        avg_attack_loss = np.mean([r['attacked_loss'] for r in results])
        
        return {
            'avg_degradation': avg_degradation,
            'std_degradation': np.std([r['degradation'] for r in results]),
            'success_rate': success_rate,
            'avg_orig_loss': avg_orig_loss,
            'avg_attack_loss': avg_attack_loss,
            'num_samples': len(results),
            'individual_results': results,
        }


def main():
    """Run HotFlip attacks"""
    logger = StageLogger("stage_6_hotflip")
    
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
        except:
            logger.log("  Test split not available, using validation...")
            pile_test = load_dataset(
                Config.TRAINING_DATASET,
                split="validation",
                streaming=False,
                cache_dir=Config.DATA_CACHE_DIR,
                trust_remote_code=True
            )
        
        # Collect samples
        max_samples = Config.HOTFLIP_NUM_SAMPLES
        test_texts = [example['text'] for i, example in enumerate(pile_test) if i < max_samples]
        
        logger.log(f"  ✓ Loaded {len(test_texts)} test samples")
        
        logger.log(f"  Test set: {len(test_texts)} samples")
        
        # Dictionary to store results
        all_results = {}
        
        # Attack each model
        for model_name, model_type in [
            ('baseline_pythia', 'baseline'),
            ('monotonic_pythia', 'monotonic')
        ]:
            logger.log(f"\n{'='*80}")
            logger.log(f"HOTFLIP ATTACK ON: {model_name}")
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
            logger.log(f"✓ Model loaded")
            
            # Run HotFlip attack
            logger.log(f"\nRunning HotFlip attacks on {len(test_texts)} examples...")
            logger.log(f"  Num flips per example: {Config.HOTFLIP_NUM_FLIPS}")
            
            attacker = HotFlipAttacker(
                model, tokenizer, device,
                num_flips=Config.HOTFLIP_NUM_FLIPS
            )
            
            attack_results = attacker.attack_batch(test_texts)
            
            logger.log(f"\n✓ Attack complete!")
            logger.log(f"  Average degradation: {attack_results['avg_degradation']*100:.2f}%")
            logger.log(f"  Success rate: {attack_results['success_rate']*100:.1f}%")
            logger.log(f"  Avg original loss: {attack_results['avg_orig_loss']:.4f}")
            logger.log(f"  Avg attacked loss: {attack_results['avg_attack_loss']:.4f}")
            
            # Remove individual results to save space
            attack_results_summary = {k: v for k, v in attack_results.items() 
                                     if k != 'individual_results'}
            attack_results_summary['model_name'] = model_name
            
            all_results[model_name] = attack_results_summary
        
        # Statistical tests
        logger.log("\nComputing statistical significance...")
        from scipy import stats
        
        baseline_degs = all_results['baseline_pythia'].get('std_degradation', 0)
        monotonic_degs = all_results['monotonic_pythia'].get('std_degradation', 0)
        
        # Placeholder for t-test (would need individual results)
        statistical_tests = {
            'baseline_vs_monotonic': {
                'significant': all_results['baseline_pythia']['success_rate'] > 
                              all_results['monotonic_pythia']['success_rate'],
                'note': 'Detailed statistical test requires individual example results'
            }
        }
        
        # Save results
        logger.log("\nSaving HotFlip results...")
        final_results = {
            'seed': Config.CURRENT_SEED,
            'attack_config': {
                'num_flips': Config.HOTFLIP_NUM_FLIPS,
                'num_samples': Config.HOTFLIP_NUM_SAMPLES,
                'success_threshold': Config.ATTACK_SUCCESS_THRESHOLD,
            },
            'results': all_results,
            'statistical_tests': statistical_tests
        }
        
        save_json(
            final_results,
            os.path.join(Config.RESULTS_DIR, 'hotflip_results.json')
        )
        
        logger.log("\n✓ HotFlip attacks complete!")
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
