import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probswap.attack import ProbSwapAttack
from probswap.models import ModelWrapper
from probswap.utils import calculate_metrics
import json
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

class ProbSwapEvaluator:
    def __init__(
        self,
        target_model_name: str,
        substitute_model_name: str,
        prob_thresholds: List[float] = [0.05, 0.1, 0.15],
        top_k_values: List[int] = [3, 5, 10]
    ):
        """Initialize evaluator with different parameter configurations."""
        self.target_model = ModelWrapper(target_model_name)
        self.substitute_model = ModelWrapper(substitute_model_name)
        self.prob_thresholds = prob_thresholds
        self.top_k_values = top_k_values
        
    def evaluate_single_config(
        self,
        texts: List[str],
        prob_threshold: float,
        top_k: int
    ) -> Dict[str, float]:
        """Evaluate attack with specific configuration."""
        attack = ProbSwapAttack(
            target_model=self.target_model.model,
            target_tokenizer=self.target_model.tokenizer,
            substitute_model=self.substitute_model.model,
            substitute_tokenizer=self.substitute_model.tokenizer,
            prob_threshold=prob_threshold,
            top_k_substitutes=top_k
        )
        
        metrics_list = []
        for text in tqdm(texts, desc=f"Evaluating (thresh={prob_threshold}, top_k={top_k})"):
            modified_text, modifications = attack.attack(text)
            metrics = calculate_metrics(text, modified_text, modifications)
            metrics_list.append(metrics)
        
        # Average metrics across all texts
        avg_metrics = {
            "prob_threshold": prob_threshold,
            "top_k": top_k,
            "avg_num_modifications": np.mean([m["num_modifications"] for m in metrics_list]),
            "avg_prob_increase": np.mean([m["avg_prob_increase"] for m in metrics_list]),
            "avg_bleu_score": np.mean([m["bleu_score"] for m in metrics_list])
        }
        
        return avg_metrics
    
    def evaluate_all_configs(
        self,
        texts: List[str],
        output_file: str = "evaluation_results.json"
    ) -> List[Dict[str, float]]:
        """Evaluate attack with all parameter configurations."""
        all_results = []
        
        for prob_threshold in self.prob_thresholds:
            for top_k in self.top_k_values:
                results = self.evaluate_single_config(texts, prob_threshold, top_k)
                all_results.append(results)
                
        # Save results
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        return all_results

def main():
    # Example usage
    target_model_name = "gpt2"  # Replace with your watermarked model
    substitute_model_name = "gpt2-medium"
    
    # Sample texts (in practice, these would be your watermarked texts)
    sample_texts = [
        """The artificial intelligence revolution has transformed various aspects of our 
        daily lives, from virtual assistants to autonomous vehicles.""",
        
        """Machine learning algorithms continue to improve, enabling more sophisticated 
        applications in fields like healthcare and finance.""",
        
        """Natural language processing has made significant strides, allowing computers 
        to better understand and generate human language."""
    ]
    
    evaluator = ProbSwapEvaluator(
        target_model_name=target_model_name,
        substitute_model_name=substitute_model_name
    )
    
    results = evaluator.evaluate_all_configs(
        texts=sample_texts,
        output_file="evaluation_results.json"
    )
    
    # Print summary
    print("\nEvaluation Results Summary:")
    print("=" * 50)
    for result in results:
        print(f"\nConfiguration:")
        print(f"  Probability Threshold: {result['prob_threshold']}")
        print(f"  Top-k Substitutes: {result['top_k']}")
        print(f"Results:")
        print(f"  Average # of Modifications: {result['avg_num_modifications']:.2f}")
        print(f"  Average Probability Increase: {result['avg_prob_increase']:.3f}")
        print(f"  Average BLEU Score: {result['avg_bleu_score']:.3f}")

if __name__ == "__main__":
    main()
