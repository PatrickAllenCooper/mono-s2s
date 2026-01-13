#!/usr/bin/env python3
"""
Stage 6: HotFlip Gradient-Based Attacks

This stage performs gradient-based HotFlip attacks on all three models.
HotFlip flips input tokens to maximize loss using embedding gradient information.

Attack Strategy:
- For each example, compute gradients w.r.t. input embeddings
- Identify positions with highest gradient norms
- Replace tokens to maximize loss increase
- Evaluate loss degradation and attack success rate

Models attacked:
1. Standard T5 (pre-trained) - Reference
2. Baseline T5 (fine-tuned, unconstrained) - Fair baseline
3. Monotonic T5 (fine-tuned, W≥0 FFN constraints) - Treatment

Attack parameters:
- num_flips: Number of tokens to flip per example (default: 5)
- beam_size: Number of candidate replacements per position (default: 10)

Inputs:
- attack_data.pt (from stage 1) - evaluation split
- All three model checkpoints

Outputs:
- hotflip_results.json (attack statistics per model)
- stage_6_hotflip_complete.flag
"""

# Set environment variables BEFORE importing torch
import os
os.environ["PYTHONHASHSEED"] = str(os.environ.get("EXPERIMENT_SEED", "42"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import time
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.experiment_config import ExperimentConfig
from utils.common_utils import (
    set_all_seeds, create_completion_flag, check_dependencies,
    save_json, StageLogger, load_model
)

# Import transformers AFTER environment setup
from transformers import T5Tokenizer


class HotFlipT5Attack:
    """
    Gradient-based HotFlip attack for T5 summarization models.
    Replaces tokens by maximizing loss using gradient information.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Get embedding layer
        self.embedding_layer = model.get_input_embeddings()
        self.vocab_size = self.embedding_layer.num_embeddings
    
    def compute_loss(self, text, summary):
        """Compute loss for text-summary pair"""
        inputs = self.tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        targets = self.tokenizer(
            summary,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=targets.input_ids
            )
        
        return outputs.loss.item()
    
    def get_embedding_gradients(self, text, summary):
        """
        Compute gradients w.r.t. input embeddings.
        Returns: (gradients, input_ids, attention_mask)
        """
        # Tokenize input
        inputs = self.tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        targets = self.tokenizer(
            summary,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(self.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Get embeddings and enable gradients
        embeddings = self.embedding_layer(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # Forward pass through encoder with custom embeddings
        encoder = self.model.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Forward through decoder with target labels
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=targets.input_ids
        )
        
        # Compute loss and backprop
        loss = outputs.loss
        loss.backward()
        
        # Get gradients
        gradients = embeddings.grad.data
        
        return gradients, input_ids, attention_mask
    
    def find_best_replacement(self, position, gradient_vector, current_token_id, top_k=10):
        """
        Find the token that maximizes loss increase at given position.
        Uses dot product between gradient and embedding vectors.
        """
        # Get all embedding vectors
        embedding_matrix = self.embedding_layer.weight.data  # [vocab_size, embedding_dim]
        
        # Compute dot products: higher = more loss increase
        scores = torch.matmul(embedding_matrix, gradient_vector)  # [vocab_size]
        
        # Get top-k candidates (excluding current token and special tokens)
        scores[current_token_id] = -float('inf')  # Don't replace with same token
        scores[0] = -float('inf')  # Don't use PAD
        scores[1] = -float('inf')  # Don't use EOS
        
        # Get top-k tokens
        top_values, top_indices = torch.topk(scores, top_k)
        
        return top_indices.cpu().numpy(), top_values.cpu().numpy()
    
    def attack_single(self, text, summary, num_flips=5, beam_size=10):
        """
        Attack a single example by flipping num_flips tokens.
        
        Args:
            text: Input text
            summary: Target summary
            num_flips: Number of tokens to flip
            beam_size: Number of candidate replacements to try per position
        
        Returns:
            attacked_text, attack_info
        """
        # Get gradients
        gradients, input_ids, attention_mask = self.get_embedding_gradients(text, summary)
        
        # Compute gradient norms for each position
        grad_norms = gradients.norm(dim=-1).squeeze()  # [seq_len]
        
        # Mask special tokens (don't flip them)
        special_token_ids = [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
        ]
        
        tokens = input_ids.squeeze().cpu().numpy()
        for idx, token in enumerate(tokens):
            if token in special_token_ids or idx == 0:  # Also protect first token
                grad_norms[idx] = 0
        
        # Find top-k positions with highest gradient norms
        top_positions = grad_norms.argsort(descending=True)[:num_flips]
        
        # Create attacked version
        attacked_ids = input_ids.clone()
        flip_info = []
        
        for pos in top_positions:
            pos_idx = pos.item()
            current_token = tokens[pos_idx]
            gradient_vec = gradients[0, pos_idx]
            
            # Find best replacement token
            candidate_tokens, candidate_scores = self.find_best_replacement(
                pos_idx, gradient_vec, current_token, top_k=beam_size
            )
            
            # Use the best candidate
            best_token = candidate_tokens[0]
            attacked_ids[0, pos_idx] = best_token
            
            flip_info.append({
                'position': pos_idx,
                'original_token': self.tokenizer.decode([current_token]),
                'new_token': self.tokenizer.decode([best_token]),
                'gradient_norm': grad_norms[pos_idx].item(),
                'score': candidate_scores[0]
            })
        
        # Decode attacked text
        attacked_text = self.tokenizer.decode(attacked_ids[0], skip_special_tokens=True)
        attacked_text = attacked_text.replace("summarize: ", "").strip()
        
        return attacked_text, flip_info
    
    def evaluate_attack_batch(self, texts, summaries, num_flips=5):
        """
        Attack multiple examples and compute statistics.
        """
        results = []
        
        progress_bar = tqdm(zip(texts, summaries), total=len(texts), desc="HotFlip Attack")
        
        for text, summary in progress_bar:
            # Compute original loss
            orig_loss = self.compute_loss(text, summary)
            
            # Perform attack
            attacked_text, flip_info = self.attack_single(text, summary, num_flips=num_flips)
            
            # Compute attacked loss
            attack_loss = self.compute_loss(attacked_text, summary)
            
            # Compute degradation
            degradation = (attack_loss - orig_loss) / orig_loss if orig_loss > 0 else 0
            
            results.append({
                'original_text': text,
                'attacked_text': attacked_text,
                'original_summary': summary,
                'orig_loss': orig_loss,
                'attack_loss': attack_loss,
                'degradation': degradation,
                'num_flips': len(flip_info),
                'flip_info': flip_info
            })
            
            # Update progress bar
            avg_deg = np.mean([r['degradation'] for r in results])
            progress_bar.set_postfix({
                'avg_degradation': f'{avg_deg:.2%}',
                'last_degradation': f'{degradation:.2%}'
            })
        
        return results


def main():
    """Run HotFlip attacks on all models"""
    logger = StageLogger("stage_6_hotflip")
    
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
        attack_data = torch.load(os.path.join(data_cache_dir, 'attack_data.pt'), weights_only=False)
        
        # Use evaluation split for HotFlip
        attack_texts = attack_data['evaluation']['texts']
        attack_summaries = attack_data['evaluation']['summaries']
        
        # Limit to a reasonable subset for HotFlip (it's expensive)
        max_samples = 200 if ExperimentConfig.USE_FULL_TEST_SETS else 100
        attack_texts = attack_texts[:max_samples]
        attack_summaries = attack_summaries[:max_samples]
        
        logger.log(f"  Attack samples: {len(attack_texts)}")
        
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
        
        # Attack each model
        logger.log("\n" + "="*80)
        logger.log("HOTFLIP GRADIENT-BASED ATTACKS")
        logger.log("="*80)
        logger.log(f"Attack parameters:")
        logger.log(f"  num_flips: {ExperimentConfig.ATTACK_TRIGGER_LENGTH}")
        logger.log(f"  beam_size: {ExperimentConfig.ATTACK_NUM_CANDIDATES}")
        
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
            attacker = HotFlipT5Attack(model, tokenizer, device)
            
            # Perform attacks
            results = attacker.evaluate_attack_batch(
                attack_texts,
                attack_summaries,
                num_flips=ExperimentConfig.ATTACK_TRIGGER_LENGTH
            )
            
            # Compute statistics
            degradations = [r['degradation'] for r in results]
            orig_losses = [r['orig_loss'] for r in results]
            attack_losses = [r['attack_loss'] for r in results]
            
            avg_degradation = np.mean(degradations)
            std_degradation = np.std(degradations)
            success_rate = sum(1 for d in degradations if d > 0.1) / len(degradations)
            
            logger.log(f"\nResults for {model_name}:")
            logger.log(f"  Avg degradation: {avg_degradation:.2%} ± {std_degradation:.2%}")
            logger.log(f"  Success rate (>10% degradation): {success_rate:.1%}")
            logger.log(f"  Avg original loss: {np.mean(orig_losses):.4f}")
            logger.log(f"  Avg attacked loss: {np.mean(attack_losses):.4f}")
            logger.log(f"  Min degradation: {min(degradations):.2%}")
            logger.log(f"  Max degradation: {max(degradations):.2%}")
            
            # Store results (without full examples to save space)
            all_results[model_key] = {
                'model_name': model_name,
                'avg_degradation': avg_degradation,
                'std_degradation': std_degradation,
                'success_rate': success_rate,
                'avg_orig_loss': float(np.mean(orig_losses)),
                'avg_attack_loss': float(np.mean(attack_losses)),
                'min_degradation': float(min(degradations)),
                'max_degradation': float(max(degradations)),
                'num_samples': len(results),
                'degradations': degradations  # Keep for statistical tests
            }
        
        # Comparative analysis
        logger.log("\n" + "="*80)
        logger.log("COMPARATIVE ANALYSIS")
        logger.log("="*80)
        
        logger.log("\nVulnerability Ranking (Higher = More Vulnerable):")
        ranked_models = sorted(
            all_results.items(),
            key=lambda x: x[1]['avg_degradation'],
            reverse=True
        )
        
        for rank, (model_key, stats) in enumerate(ranked_models, 1):
            logger.log(f"{rank}. {stats['model_name']:20s} | "
                      f"Degradation: {stats['avg_degradation']:.2%} | "
                      f"Success: {stats['success_rate']:.1%}")
        
        # Robustness improvement
        std_deg = all_results['standard_t5']['avg_degradation']
        baseline_deg = all_results['baseline_t5']['avg_degradation']
        mono_deg = all_results['monotonic_t5']['avg_degradation']
        
        improvement_baseline = (std_deg - baseline_deg) / std_deg * 100
        improvement_mono = (std_deg - mono_deg) / std_deg * 100
        
        logger.log(f"\n🛡️ Robustness Improvement vs Standard T5:")
        logger.log(f"  Baseline T5: {improvement_baseline:+.1f}%")
        logger.log(f"  Monotonic T5: {improvement_mono:+.1f}%")
        
        # Statistical significance
        from scipy import stats
        
        logger.log(f"\n📈 Statistical Significance Testing:")
        
        std_degradations = all_results['standard_t5']['degradations']
        baseline_degradations = all_results['baseline_t5']['degradations']
        mono_degradations = all_results['monotonic_t5']['degradations']
        
        # Test 1: Standard vs Baseline
        t_stat1, p_value1 = stats.ttest_ind(std_degradations, baseline_degradations)
        sig1 = "***" if p_value1 < 0.001 else "**" if p_value1 < 0.01 else "*" if p_value1 < 0.05 else "ns"
        logger.log(f"  Standard vs Baseline: t={t_stat1:.3f}, p={p_value1:.4f} {sig1}")
        
        # Test 2: Standard vs Monotonic
        t_stat2, p_value2 = stats.ttest_ind(std_degradations, mono_degradations)
        sig2 = "***" if p_value2 < 0.001 else "**" if p_value2 < 0.01 else "*" if p_value2 < 0.05 else "ns"
        logger.log(f"  Standard vs Monotonic: t={t_stat2:.3f}, p={p_value2:.4f} {sig2}")
        
        # Test 3: Baseline vs Monotonic
        t_stat3, p_value3 = stats.ttest_ind(baseline_degradations, mono_degradations)
        sig3 = "***" if p_value3 < 0.001 else "**" if p_value3 < 0.01 else "*" if p_value3 < 0.05 else "ns"
        logger.log(f"  Baseline vs Monotonic: t={t_stat3:.3f}, p={p_value3:.4f} {sig3}")
        
        # Save results
        logger.log("\n" + "="*80)
        logger.log("SAVING RESULTS")
        logger.log("="*80)
        
        # Remove degradations list for JSON (too large)
        results_for_json = {}
        for model_key in all_results:
            results_for_json[model_key] = {k: v for k, v in all_results[model_key].items() if k != 'degradations'}
        
        output = {
            'seed': ExperimentConfig.CURRENT_SEED,
            'attack_config': {
                'num_flips': ExperimentConfig.ATTACK_TRIGGER_LENGTH,
                'num_samples': len(attack_texts),
            },
            'results': results_for_json,
            'statistical_tests': {
                'standard_vs_baseline': {'t_stat': t_stat1, 'p_value': p_value1, 'significant': sig1},
                'standard_vs_monotonic': {'t_stat': t_stat2, 'p_value': p_value2, 'significant': sig2},
                'baseline_vs_monotonic': {'t_stat': t_stat3, 'p_value': p_value3, 'significant': sig3},
            }
        }
        
        results_file = os.path.join(ExperimentConfig.RESULTS_DIR, 'hotflip_results.json')
        save_json(output, results_file)
        
        logger.log(f"\n✓ HotFlip attack evaluation complete!")
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

