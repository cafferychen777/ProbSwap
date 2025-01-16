import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MarkLLM"))

from probswap.attack import ProbSwapAttack
from probswap.models import ModelWrapper
from probswap.utils import calculate_metrics, format_results
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

def run_basic_attack_experiment(
    text_samples,
    target_model_name="gpt2",
    substitute_model_name="gpt2-medium",
    prob_thresholds=[0.05, 0.1, 0.15],
    output_dir="experiment_results"
):
    """Run basic attack experiment without watermark."""
    
    results = []
    
    # Initialize models
    print(f"Loading models: {target_model_name} and {substitute_model_name}")
    target_model = ModelWrapper(target_model_name)
    substitute_model = ModelWrapper(substitute_model_name)
    
    for threshold in prob_thresholds:
        print(f"\nTesting with probability threshold: {threshold}")
        
        attack = ProbSwapAttack(
            target_model=target_model.model,
            target_tokenizer=target_model.tokenizer,
            substitute_model=substitute_model.model,
            substitute_tokenizer=substitute_model.tokenizer,
            prob_threshold=threshold,
            top_k_substitutes=5
        )
        
        threshold_results = []
        
        for text in tqdm(text_samples, desc="Processing samples"):
            # Apply attack
            modified_text, modifications = attack.attack(text)
            
            # Calculate metrics
            metrics = calculate_metrics(text, modified_text, modifications)
            
            threshold_results.append({
                "original_text": text,
                "modified_text": modified_text,
                "modifications": modifications,
                "metrics": metrics
            })
        
        # Calculate average metrics for this threshold
        avg_metrics = {
            "threshold": threshold,
            "avg_modifications": np.mean([r["metrics"]["num_modifications"] for r in threshold_results]),
            "avg_prob_increase": np.mean([r["metrics"]["avg_prob_increase"] for r in threshold_results]),
            "avg_bleu": np.mean([r["metrics"]["bleu_score"] for r in threshold_results])
        }
        
        results.append({
            "threshold": threshold,
            "detailed_results": threshold_results,
            "average_metrics": avg_metrics
        })
        
        print(f"\nResults for threshold {threshold}:")
        print(f"Average modifications: {avg_metrics['avg_modifications']:.2f}")
        print(f"Average probability increase: {avg_metrics['avg_prob_increase']:.3f}")
        print(f"Average BLEU score: {avg_metrics['avg_bleu']:.3f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"attack_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results

def main():
    # Test samples with varying complexity and length
    test_samples = [
        """Artificial intelligence has revolutionized various aspects of modern technology,
        enabling unprecedented advances in automation and decision-making processes.""",
        
        """The integration of machine learning algorithms in healthcare has significantly
        improved diagnostic accuracy and treatment planning, leading to better patient outcomes.""",
        
        """Natural language processing systems have made remarkable progress in understanding
        and generating human-like text, transforming how we interact with computers.""",
        
        """The ethical implications of AI development require careful consideration,
        particularly regarding privacy, bias, and the potential impact on employment."""
    ]
    
    # Create results directory
    os.makedirs("experiment_results", exist_ok=True)
    
    # Run experiments
    print("Starting ProbSwap attack experiments...")
    results = run_basic_attack_experiment(
        text_samples=test_samples,
        prob_thresholds=[0.05, 0.1, 0.15],
        output_dir="experiment_results"
    )
    
    # Print summary
    print("\nExperiment Summary:")
    print("=" * 50)
    for result in results:
        threshold = result["threshold"]
        metrics = result["average_metrics"]
        print(f"\nThreshold: {threshold}")
        print(f"Average modifications per text: {metrics['avg_modifications']:.2f}")
        print(f"Average probability increase: {metrics['avg_prob_increase']:.3f}")
        print(f"Average BLEU score: {metrics['avg_bleu']:.3f}")

if __name__ == "__main__":
    main()
