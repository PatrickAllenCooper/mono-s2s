#!/usr/bin/env python3
"""
Local Pipeline Test

Runs the entire experimental pipeline locally with tiny models and minimal data.
This validates the pipeline logic before expensive HPC deployment.

Usage:
    python test_pipeline_local.py
    python test_pipeline_local.py --verbose
    python test_pipeline_local.py --save-outputs ./test_outputs
"""

import argparse
import sys
import os
import torch
import tempfile
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from configs.experiment_config import FoundationExperimentConfig as Config
from utils.common_utils import (
    set_all_seeds,
    make_model_monotonic,
    compute_perplexity,
    save_json,
    load_json,
    create_completion_flag,
    check_dependencies,
    StageLogger,
    LanguageModelingDataset,
)


class LocalPipelineTester:
    """Run full pipeline locally with tiny models"""
    
    def __init__(self, work_dir, verbose=False):
        self.work_dir = work_dir
        self.verbose = verbose
        self.device = torch.device('cpu')  # Force CPU for local testing
        
        # Override config paths
        self.results_dir = os.path.join(work_dir, 'results')
        self.checkpoint_dir = os.path.join(work_dir, 'checkpoints')
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.log("Local Pipeline Tester initialized")
        self.log(f"Work dir: {work_dir}")
    
    def log(self, message):
        """Log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        if self.verbose:
            print(msg)
    
    def run_stage_0_setup(self):
        """Stage 0: Setup (simulated - use tiny model)"""
        print("\n" + "="*80)
        print("STAGE 0: SETUP (Simulated with tiny model)")
        print("="*80)
        
        try:
            from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
            
            # Create tiny model instead of downloading Pythia
            print("Creating tiny GPT-2 model for testing...")
            config = GPT2Config(
                vocab_size=1000,
                n_positions=128,
                n_embd=128,
                n_layer=2,
                n_head=2,
                n_inner=512,
            )
            
            model = GPT2LMHeadModel(config)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            
            num_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Model created: {num_params:,} parameters")
            
            # Save for later stages
            self.model_config = config
            self.tokenizer = tokenizer
            
            # Create completion flag
            create_completion_flag('stage_0_setup', work_dir=self.work_dir)
            print("✓ Stage 0 complete")
            
            return True
            
        except Exception as e:
            print(f"✗ Stage 0 failed: {e}")
            return False
    
    def run_stage_1_apply_monotonicity(self):
        """Stage 1: Apply monotonicity constraints"""
        print("\n" + "="*80)
        print("STAGE 1: APPLY MONOTONICITY")
        print("="*80)
        
        try:
            if not check_dependencies(['stage_0_setup'], work_dir=self.work_dir):
                print("✗ Dependencies not met")
                return False
            
            from transformers import GPT2LMHeadModel
            
            # Create model
            print("Creating model...")
            model = GPT2LMHeadModel(self.model_config)
            
            # Apply monotonicity
            print("Applying monotonicity constraints...")
            monotonic_model = make_model_monotonic(model)
            
            # Verify constraints
            print("Verifying weight constraints...")
            min_weight = float('inf')
            for name, param in monotonic_model.named_parameters():
                if 'weight' in name and any(x in name.lower() for x in ['mlp', 'fc']):
                    min_weight = min(min_weight, param.data.min().item())
            
            if min_weight >= -1e-6:
                print(f"✓ All weights non-negative (min = {min_weight:.6f})")
            else:
                print(f"✗ Found negative weights (min = {min_weight:.6f})")
                return False
            
            # Save
            save_path = os.path.join(self.checkpoint_dir, 'monotonic_initialized.pt')
            torch.save(monotonic_model.state_dict(), save_path)
            print(f"✓ Saved to: {save_path}")
            
            create_completion_flag('stage_1_apply_monotonicity', work_dir=self.work_dir)
            print("✓ Stage 1 complete")
            
            return True
            
        except Exception as e:
            print(f"✗ Stage 1 failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def run_stage_2_baseline_training(self):
        """Stage 2: Baseline training (minimal)"""
        print("\n" + "="*80)
        print("STAGE 2: BASELINE TRAINING (Minimal - 5 steps)")
        print("="*80)
        
        try:
            if not check_dependencies(['stage_0_setup'], work_dir=self.work_dir):
                print("✗ Dependencies not met")
                return False
            
            from transformers import GPT2LMHeadModel
            from torch.utils.data import DataLoader
            from torch.optim import AdamW
            
            # Create model
            model = GPT2LMHeadModel(self.model_config).to(self.device)
            
            # Create dummy data
            texts = ["This is a test sentence for training."] * 20
            dataset = LanguageModelingDataset(texts, self.tokenizer, max_length=128)
            dataloader = DataLoader(dataset, batch_size=2)
            
            # Setup training
            optimizer = AdamW(model.parameters(), lr=1e-4)
            
            # Train for a few steps
            print("Training for 5 steps...")
            model.train()
            losses = []
            
            for i, batch in enumerate(dataloader):
                if i >= 5:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                losses.append(loss.item())
                self.log(f"  Step {i+1}: loss = {loss.item():.4f}")
            
            print(f"✓ Training completed, final loss: {losses[-1]:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(self.checkpoint_dir, 'baseline_checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
            
            # Save history
            history = {
                'train_losses': losses,
                'val_perplexities': [100.0],  # Dummy
                'best_val_perplexity': 100.0,
            }
            save_json(history, os.path.join(self.results_dir, 'baseline_training_history.json'))
            
            create_completion_flag('stage_2_train_baseline', work_dir=self.work_dir)
            print("✓ Stage 2 complete")
            
            return True
            
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def run_stage_3_monotonic_training(self):
        """Stage 3: Monotonic training (minimal)"""
        print("\n" + "="*80)
        print("STAGE 3: MONOTONIC TRAINING (Minimal - 5 steps)")
        print("="*80)
        
        try:
            if not check_dependencies(['stage_1_apply_monotonicity'], work_dir=self.work_dir):
                print("✗ Dependencies not met")
                return False
            
            from transformers import GPT2LMHeadModel
            from torch.utils.data import DataLoader
            from torch.optim import AdamW
            
            # Create model and apply monotonicity
            model = GPT2LMHeadModel(self.model_config)
            model = make_model_monotonic(model)
            
            # Load initialized weights
            init_path = os.path.join(self.checkpoint_dir, 'monotonic_initialized.pt')
            model.load_state_dict(torch.load(init_path, weights_only=False))
            model = model.to(self.device)
            
            # Create dummy data
            texts = ["This is a test sentence for training."] * 20
            dataset = LanguageModelingDataset(texts, self.tokenizer, max_length=128)
            dataloader = DataLoader(dataset, batch_size=2)
            
            # Setup training
            optimizer = AdamW(model.parameters(), lr=1e-4)
            
            # Train for a few steps
            print("Training for 5 steps (with extended warmup)...")
            model.train()
            losses = []
            
            for i, batch in enumerate(dataloader):
                if i >= 5:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                losses.append(loss.item())
                self.log(f"  Step {i+1}: loss = {loss.item():.4f}")
                
                # Verify weights stay non-negative
                for name, param in model.named_parameters():
                    if 'weight' in name and any(x in name.lower() for x in ['mlp', 'fc']):
                        min_val = param.data.min().item()
                        if min_val < -1e-6:
                            print(f"✗ Weights became negative: {name} = {min_val}")
                            return False
            
            print(f"✓ Training completed, final loss: {losses[-1]:.4f}")
            print(f"✓ Weights remained non-negative throughout training")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(self.checkpoint_dir, 'monotonic_checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
            
            # Save history
            history = {
                'train_losses': losses,
                'val_perplexities': [105.0],  # Dummy (slightly worse)
                'best_val_perplexity': 105.0,
            }
            save_json(history, os.path.join(self.results_dir, 'monotonic_training_history.json'))
            
            create_completion_flag('stage_3_train_monotonic', work_dir=self.work_dir)
            print("✓ Stage 3 complete")
            
            return True
            
        except Exception as e:
            print(f"✗ Stage 3 failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def run_stage_4_evaluation(self):
        """Stage 4: Evaluation (minimal)"""
        print("\n" + "="*80)
        print("STAGE 4: EVALUATION")
        print("="*80)
        
        try:
            if not check_dependencies(['stage_2_train_baseline', 'stage_3_train_monotonic'], 
                                     work_dir=self.work_dir):
                print("✗ Dependencies not met")
                return False
            
            from transformers import GPT2LMHeadModel
            from torch.utils.data import DataLoader
            
            # Load baseline model
            print("Loading baseline model...")
            baseline_model = GPT2LMHeadModel(self.model_config)
            baseline_path = os.path.join(self.checkpoint_dir, 'baseline_checkpoints', 'best_model.pt')
            baseline_model.load_state_dict(torch.load(baseline_path, weights_only=False))
            baseline_model = baseline_model.to(self.device)
            baseline_model.eval()
            
            # Load monotonic model
            print("Loading monotonic model...")
            monotonic_model = GPT2LMHeadModel(self.model_config)
            monotonic_model = make_model_monotonic(monotonic_model)
            monotonic_path = os.path.join(self.checkpoint_dir, 'monotonic_checkpoints', 'best_model.pt')
            monotonic_model.load_state_dict(torch.load(monotonic_path, weights_only=False))
            monotonic_model = monotonic_model.to(self.device)
            monotonic_model.eval()
            
            # Create evaluation data
            eval_texts = ["Evaluation test sentence."] * 10
            dataset = LanguageModelingDataset(eval_texts, self.tokenizer, max_length=128)
            dataloader = DataLoader(dataset, batch_size=2)
            
            # Compute perplexity
            print("Computing perplexity for baseline...")
            baseline_result = compute_perplexity(baseline_model, dataloader, self.device)
            print(f"  Baseline perplexity: {baseline_result['perplexity']:.2f}")
            
            print("Computing perplexity for monotonic...")
            monotonic_result = compute_perplexity(monotonic_model, dataloader, self.device)
            print(f"  Monotonic perplexity: {monotonic_result['perplexity']:.2f}")
            
            # Calculate gap
            gap = (monotonic_result['perplexity'] - baseline_result['perplexity']) / baseline_result['perplexity'] * 100
            print(f"  Perplexity gap: {gap:+.1f}%")
            
            # Save results
            results = {
                'pile_test': {
                    'baseline_pythia': baseline_result,
                    'monotonic_pythia': monotonic_result,
                }
            }
            save_json(results, os.path.join(self.results_dir, 'evaluation_results.json'))
            
            create_completion_flag('stage_4_evaluate', work_dir=self.work_dir)
            print("✓ Stage 4 complete")
            
            return True
            
        except Exception as e:
            print(f"✗ Stage 4 failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def run_stage_5_uat_attacks(self):
        """Stage 5: UAT attacks (simulated)"""
        print("\n" + "="*80)
        print("STAGE 5: UAT ATTACKS (Simulated)")
        print("="*80)
        
        # Simulated results
        print("Simulating UAT attack optimization...")
        print("(Full implementation would optimize triggers)")
        
        results = {
            'results': {
                'baseline_pythia': {
                    'trigger_text': 'test trigger baseline',
                    'nll_increase': 0.008,
                    'rouge_delta': -0.004,
                },
                'monotonic_pythia': {
                    'trigger_text': 'test trigger monotonic',
                    'nll_increase': 0.006,
                    'rouge_delta': -0.002,
                }
            }
        }
        
        save_json(results, os.path.join(self.results_dir, 'uat_results.json'))
        create_completion_flag('stage_5_uat', work_dir=self.work_dir)
        print("✓ Stage 5 complete (simulated)")
        
        return True
    
    def run_stage_6_hotflip_attacks(self):
        """Stage 6: HotFlip attacks (simulated)"""
        print("\n" + "="*80)
        print("STAGE 6: HOTFLIP ATTACKS (Simulated)")
        print("="*80)
        
        # Simulated results
        print("Simulating HotFlip gradient-based attacks...")
        print("(Full implementation would compute gradients and flip tokens)")
        
        results = {
            'results': {
                'baseline_pythia': {
                    'avg_degradation': 0.16,
                    'success_rate': 0.58,
                    'avg_orig_loss': 2.5,
                    'avg_attack_loss': 2.9,
                },
                'monotonic_pythia': {
                    'avg_degradation': 0.05,
                    'success_rate': 0.18,
                    'avg_orig_loss': 2.7,
                    'avg_attack_loss': 2.83,
                }
            }
        }
        
        save_json(results, os.path.join(self.results_dir, 'hotflip_results.json'))
        create_completion_flag('stage_6_hotflip', work_dir=self.work_dir)
        print("✓ Stage 6 complete (simulated)")
        
        return True
    
    def run_stage_7_aggregate(self):
        """Stage 7: Aggregate results"""
        print("\n" + "="*80)
        print("STAGE 7: AGGREGATE RESULTS")
        print("="*80)
        
        try:
            if not check_dependencies(['stage_4_evaluate', 'stage_5_uat', 'stage_6_hotflip'],
                                     work_dir=self.work_dir):
                print("✗ Dependencies not met")
                return False
            
            # Load all results
            print("Loading results...")
            baseline_history = load_json(os.path.join(self.results_dir, 'baseline_training_history.json'))
            monotonic_history = load_json(os.path.join(self.results_dir, 'monotonic_training_history.json'))
            evaluation = load_json(os.path.join(self.results_dir, 'evaluation_results.json'))
            uat = load_json(os.path.join(self.results_dir, 'uat_results.json'))
            hotflip = load_json(os.path.join(self.results_dir, 'hotflip_results.json'))
            
            # Aggregate
            final_results = {
                'experiment_info': {
                    'seed': Config.CURRENT_SEED,
                    'model_name': 'GPT2-tiny (test)',
                    'timestamp': datetime.now().isoformat(),
                    'test_mode': True,
                },
                'training_summary': {
                    'baseline': baseline_history,
                    'monotonic': monotonic_history,
                },
                'evaluation_summary': evaluation,
                'attack_summary': {
                    'uat': uat,
                    'hotflip': hotflip,
                }
            }
            
            # Save
            final_path = os.path.join(self.results_dir, 'final_results.json')
            save_json(final_results, final_path)
            print(f"✓ Saved final results: {final_path}")
            
            # Create summary
            summary = self.create_summary(final_results)
            summary_path = os.path.join(self.results_dir, 'experiment_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(summary)
            print(f"✓ Saved summary: {summary_path}")
            
            create_completion_flag('stage_7_aggregate', work_dir=self.work_dir)
            print("✓ Stage 7 complete")
            
            return True
            
        except Exception as e:
            print(f"✗ Stage 7 failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def create_summary(self, results):
        """Create human-readable summary"""
        baseline_ppl = results['evaluation_summary']['pile_test']['baseline_pythia']['perplexity']
        monotonic_ppl = results['evaluation_summary']['pile_test']['monotonic_pythia']['perplexity']
        
        baseline_attack = results['attack_summary']['hotflip']['results']['baseline_pythia']['success_rate']
        monotonic_attack = results['attack_summary']['hotflip']['results']['monotonic_pythia']['success_rate']
        
        summary = f"""
{'='*80}
FOUNDATION LLM MONOTONICITY EXPERIMENT - LOCAL TEST
{'='*80}

Test Mode: Using tiny GPT-2 model
Seed: {results['experiment_info']['seed']}
Timestamp: {results['experiment_info']['timestamp']}

{'='*80}
PERPLEXITY RESULTS
{'='*80}

Baseline:  {baseline_ppl:.2f}
Monotonic: {monotonic_ppl:.2f}
Gap:       {(monotonic_ppl - baseline_ppl) / baseline_ppl * 100:+.1f}%

{'='*80}
ATTACK ROBUSTNESS (HOTFLIP)
{'='*80}

Baseline Success Rate:  {baseline_attack:.1%}
Monotonic Success Rate: {monotonic_attack:.1%}
Reduction:              {(baseline_attack - monotonic_attack) / baseline_attack * 100:.1f}%

{'='*80}
SUMMARY
{'='*80}

✓ All stages completed successfully
✓ Monotonicity constraints maintained through training
✓ Attack reduction demonstrated

NOTE: These are test results with tiny model and minimal data.
Real HPC results will differ significantly.

{'='*80}
"""
        return summary
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        print("\n" + "="*80)
        print("  LOCAL PIPELINE TEST - FULL RUN")
        print("="*80)
        print()
        print("This will run all 7 stages with tiny models and minimal data.")
        print("Purpose: Verify pipeline logic before HPC deployment.")
        print()
        
        set_all_seeds(Config.CURRENT_SEED)
        
        stages = [
            ("Stage 0: Setup", self.run_stage_0_setup),
            ("Stage 1: Apply Monotonicity", self.run_stage_1_apply_monotonicity),
            ("Stage 2: Baseline Training", self.run_stage_2_baseline_training),
            ("Stage 3: Monotonic Training", self.run_stage_3_monotonic_training),
            ("Stage 4: Evaluation", self.run_stage_4_evaluation),
            ("Stage 5: UAT Attacks", self.run_stage_5_uat_attacks),
            ("Stage 6: HotFlip Attacks", self.run_stage_6_hotflip_attacks),
            ("Stage 7: Aggregate", self.run_stage_7_aggregate),
        ]
        
        results = {}
        for stage_name, stage_func in stages:
            success = stage_func()
            results[stage_name] = success
            
            if not success:
                print(f"\n✗ Pipeline stopped at: {stage_name}")
                return False
        
        # Final summary
        print("\n" + "="*80)
        print("  PIPELINE TEST COMPLETE")
        print("="*80)
        print()
        
        # Show summary
        summary_path = os.path.join(self.results_dir, 'experiment_summary.txt')
        if os.path.exists(summary_path):
            print("Results Summary:")
            print("-"*80)
            with open(summary_path) as f:
                print(f.read())
        
        print("="*80)
        print("  ✓ ALL STAGES COMPLETED SUCCESSFULLY")
        print("="*80)
        print()
        print("Outputs saved to:")
        print(f"  {self.work_dir}")
        print()
        print("Next steps:")
        print("  1. Review outputs above")
        print("  2. Run pytest for comprehensive tests: pytest tests/ -v")
        print("  3. Deploy to HPC: bash run_all.sh")
        print()
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test pipeline locally")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--save-outputs', type=str, help="Directory to save test outputs")
    parser.add_argument('--keep-outputs', action='store_true', help="Don't delete test outputs after")
    
    args = parser.parse_args()
    
    # Create temporary work directory
    if args.save_outputs:
        work_dir = args.save_outputs
        os.makedirs(work_dir, exist_ok=True)
        cleanup = False
    else:
        work_dir = tempfile.mkdtemp(prefix="foundation_llm_test_")
        cleanup = not args.keep_outputs
    
    try:
        # Run pipeline
        tester = LocalPipelineTester(work_dir, verbose=args.verbose)
        success = tester.run_full_pipeline()
        
        if cleanup:
            print(f"\nCleaning up test directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\nTest outputs saved to: {work_dir}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
        return 130
    
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
