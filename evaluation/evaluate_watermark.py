import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probswap.attack import ProbSwapAttack
from probswap.models import ModelWrapper
from probswap.markllm_integration import MarkLLMWrapper
from probswap.utils import calculate_metrics
import json
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import torch

class WatermarkEvaluator:
    def __init__(
        self,
        base_model_name: str = "gpt2",
        substitute_model_name: str = "gpt2-medium",
        watermark_config: Dict[str, Any] = None
    ):
        """
        Initialize evaluator for watermark attack assessment.
        
        Args:
            base_model_name: Name of the base model to watermark
            substitute_model_name: Model to use for substitutions
            watermark_config: Configuration for MarkLLM watermarking
        """
        if watermark_config is None:
            watermark_config = {
                "gamma": 0.5,  # Watermark strength
                "delta": 2.0,  # Watermark spread
                "seeding_scheme": "hash"  # Seeding method
            }
            
        # Initialize watermarked model
        self.watermarked_model = MarkLLMWrapper.from_markllm(
            base_model_name,
            watermark_config
        )
        
        # Initialize substitute model
        self.substitute_model = ModelWrapper(substitute_model_name)
        
        # Initialize attack
        self.attack = ProbSwapAttack(
            target_model=self.watermarked_model.model,
            target_tokenizer=self.watermarked_model.tokenizer,
            substitute_model=self.substitute_model.model,
            substitute_tokenizer=self.substitute_model.tokenizer,
            prob_threshold=0.1,
            top_k_substitutes=5
        )
        
    def generate_test_samples(
        self,
        prompts: List[str],
        max_length: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate watermarked samples for testing."""
        samples = []
        for prompt in tqdm(prompts, desc="Generating watermarked samples"):
            text, watermark_info = self.watermarked_model.generate_watermarked_text(
                prompt,
                max_length=max_length
            )
            samples.append({
                "prompt": prompt,
                "text": text,
                "watermark_info": watermark_info
            })
        return samples
    
    def evaluate_attack(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate attack effectiveness on watermarked samples."""
        results = []
        
        for sample in tqdm(samples, desc="Evaluating attacks"):
            # Apply attack
            modified_text, modifications = self.attack.attack(sample["text"])
            
            # Verify watermark before and after attack
            original_verification = self.watermarked_model.verify_watermark(
                sample["text"]
            )
            modified_verification = self.watermarked_model.verify_watermark(
                modified_text
            )
            
            # Calculate other metrics
            metrics = calculate_metrics(
                sample["text"],
                modified_text,
                modifications
            )
            
            results.append({
                "original_text": sample["text"],
                "modified_text": modified_text,
                "num_modifications": len(modifications),
                "original_watermark_score": original_verification["score"],
                "modified_watermark_score": modified_verification["score"],
                "watermark_reduction": original_verification["score"] - modified_verification["score"],
                "bleu_score": metrics["bleu_score"],
                "avg_prob_increase": metrics["avg_prob_increase"]
            })
            
        return results

def main():
    # Example prompts
    test_prompts = [
        "Explain the concept of artificial intelligence.",
        "What are the main challenges in machine learning?",
        "Describe the impact of technology on society.",
    ]
    
    # Initialize evaluator
    evaluator = WatermarkEvaluator()
    
    # Generate test samples
    print("Generating test samples...")
    samples = evaluator.generate_test_samples(test_prompts)
    
    # Evaluate attack
    print("\nEvaluating attack effectiveness...")
    results = evaluator.evaluate_attack(samples)
    
    # Save detailed results
    with open("watermark_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    print("\nAttack Evaluation Summary:")
    print("=" * 50)
    
    avg_metrics = {
        "watermark_reduction": np.mean([r["watermark_reduction"] for r in results]),
        "bleu_score": np.mean([r["bleu_score"] for r in results]),
        "num_modifications": np.mean([r["num_modifications"] for r in results]),
        "avg_prob_increase": np.mean([r["avg_prob_increase"] for r in results])
    }
    
    print(f"\nAverage Watermark Score Reduction: {avg_metrics['watermark_reduction']:.3f}")
    print(f"Average BLEU Score: {avg_metrics['bleu_score']:.3f}")
    print(f"Average Number of Modifications: {avg_metrics['num_modifications']:.2f}")
    print(f"Average Probability Increase: {avg_metrics['avg_prob_increase']:.3f}")

if __name__ == "__main__":
    main()
