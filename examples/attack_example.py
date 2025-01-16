import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probswap.attack import ProbSwapAttack
from probswap.models import ModelWrapper
from probswap.utils import calculate_metrics, format_results
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Initialize models
    # You can replace these with your preferred models
    target_model_name = "gpt2"  # This would be your watermarked model
    substitute_model_name = "gpt2-medium"  # Model for finding replacements
    
    print("Loading models...")
    target_model = ModelWrapper(target_model_name)
    substitute_model = ModelWrapper(substitute_model_name)
    
    # Initialize attack
    attack = ProbSwapAttack(
        target_model=target_model.model,
        target_tokenizer=target_model.tokenizer,
        substitute_model=substitute_model.model,
        substitute_tokenizer=substitute_model.tokenizer,
        prob_threshold=0.1,  # Adjust this threshold as needed
        top_k_substitutes=5
    )
    
    # Example watermarked text
    # In practice, this would come from your watermarking method
    watermarked_text = """
    The artificial intelligence revolution has transformed various aspects of our 
    daily lives, from virtual assistants to autonomous vehicles. These technological 
    advancements continue to push the boundaries of what machines can accomplish.
    """
    
    print("\nOriginal text:")
    print(watermarked_text)
    
    # Apply attack
    print("\nApplying ProbSwap attack...")
    modified_text, modifications = attack.attack(watermarked_text)
    
    # Calculate metrics
    metrics = calculate_metrics(watermarked_text, modified_text, modifications)
    
    # Display results
    print("\n" + format_results(watermarked_text, modified_text, modifications, metrics))

if __name__ == "__main__":
    main()
