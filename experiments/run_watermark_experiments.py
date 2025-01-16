import sys
import os
import logging
import asyncio

# Add project root and MarkLLM to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "MarkLLM"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from MarkLLM.watermark.kgw import KGW, KGWConfig
from MarkLLM.utils.transformers_config import TransformersConfig
from probswap.attack import ProbSwapAttack
from probswap.models import ModelWrapper
from probswap.claude_wrapper import ClaudeWrapper
from probswap.utils import calculate_metrics
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
import nltk

# Download required NLTK data
nltk.download('punkt')

# Setup logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('watermark_experiments')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, f'experiment_{timestamp}.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def setup_watermark(model_name="facebook/opt-1.3b", device="cuda" if torch.cuda.is_available() else "cpu", logger=None):
    """Setup watermarked model using MarkLLM KGW."""
    if logger:
        logger.info(f"Setting up watermarked model: {model_name}")
    
    # Load model and tokenizer
    if logger:
        logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if logger:
        logger.info("Model and tokenizer loaded successfully")
    
    # Setup transformers config
    if logger:
        logger.info("Setting up transformers config...")
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Setup KGW config
    if logger:
        logger.info("Setting up KGW watermark config...")
    watermark_config = {
        "algorithm_name": "KGW",
        "gamma": 0.5,        # Proportion of vocabulary to include in greenlist
        "delta": 2.0,        # Logit bias for greenlist tokens
        "hash_key": 15485863,# Prime number for hashing
        "prefix_length": 1,  # Length of prefix for computing hash
        "z_threshold": 4.0,  # Threshold for watermark detection
        "f_scheme": "time",  # Hashing scheme for prefix
        "window_scheme": "left" # Window scheme for greenlist
    }
    
    # Initialize KGW watermarker
    if logger:
        logger.info("Initializing KGW watermarker...")
    watermarker = KGW(watermark_config, transformers_config)
    if logger:
        logger.info("Watermarker initialized successfully")
    
    return watermarker, model, tokenizer

def run_watermark_experiment(
    prompts,
    watermarked_model_name="facebook/opt-1.3b",
    substitute_type="claude",  # "local" or "claude"
    substitute_model_name="gpt2-medium",  # only used if substitute_type is "local"
    prob_thresholds=[0.05, 0.1, 0.15],
    output_dir="experiment_results",
    max_length=200
):
    """Run experiment with watermarked text."""
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting watermark experiment")
    logger.info(f"Model configuration: {watermarked_model_name} (target) and {substitute_type} (substitute)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    results = []
    
    # Setup watermarked model
    watermarker, target_model, target_tokenizer = setup_watermark(watermarked_model_name, device, logger)
    
    # Initialize substitute model
    if substitute_type == "claude":
        logger.info("Initializing Claude API as substitute model")
        substitute_model = ClaudeWrapper(target_tokenizer=target_tokenizer)
    else:
        logger.info(f"Loading substitute model: {substitute_model_name}")
        substitute_model = ModelWrapper(substitute_model_name)
    logger.info("Substitute model initialized successfully")
    
    # Generate watermarked samples
    logger.info("Generating watermarked samples...")
    watermarked_samples = []
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Processing prompt {i}/{len(prompts)}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        # Generate watermarked text with max_length
        inputs = target_tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        ).to(device)
        
        with torch.no_grad():
            outputs = target_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=target_tokenizer.eos_token_id,
                eos_token_id=target_tokenizer.eos_token_id
            )
            watermarked_text = target_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        watermark_result = watermarker.detect_watermark(watermarked_text)
        
        logger.info(f"Generated text length: {len(watermarked_text)}")
        logger.info(f"Original watermark score: {watermark_result['score']:.3f}")
        logger.info(f"Is watermarked: {watermark_result['is_watermarked']}")
        
        watermarked_samples.append({
            "prompt": prompt,
            "text": watermarked_text,
            "original_score": watermark_result["score"],
            "original_is_watermarked": watermark_result["is_watermarked"]
        })
    
    # Test different probability thresholds
    for threshold in prob_thresholds:
        logger.info(f"\nTesting with probability threshold: {threshold}")
        
        attack = ProbSwapAttack(
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            substitute_model=substitute_model,  # Pass the wrapper directly
            prob_threshold=threshold,
            top_k_substitutes=5
        )
        
        threshold_results = []
        
        for i, sample in enumerate(watermarked_samples, 1):
            logger.info(f"Processing sample {i}/{len(watermarked_samples)} with threshold {threshold}")
            
            # Apply attack (now using asyncio)
            loop = asyncio.get_event_loop()
            modified_text, modifications = loop.run_until_complete(attack.attack(sample["text"]))
            
            logger.info(f"Number of modifications: {len(modifications)}")
            
            # Verify watermark after attack
            modified_result = watermarker.detect_watermark(modified_text)
            logger.info(f"Modified watermark score: {modified_result['score']:.3f}")
            logger.info(f"Still watermarked: {modified_result['is_watermarked']}")
            
            # Calculate other metrics
            metrics = calculate_metrics(sample["text"], modified_text, modifications)
            logger.info(f"BLEU score: {metrics['bleu_score']:.3f}")
            logger.info(f"Average probability increase: {metrics['avg_prob_increase']:.3f}")
            
            result = {
                "prompt": sample["prompt"],
                "original_text": sample["text"],
                "modified_text": modified_text,
                "original_watermark_score": sample["original_score"],
                "original_is_watermarked": sample["original_is_watermarked"],
                "modified_watermark_score": modified_result["score"],
                "modified_is_watermarked": modified_result["is_watermarked"],
                "watermark_reduction": sample["original_score"] - modified_result["score"],
                "num_modifications": len(modifications),
                "bleu_score": metrics["bleu_score"],
                "avg_prob_increase": metrics["avg_prob_increase"]
            }
            
            threshold_results.append(result)
        
        # Calculate average metrics for this threshold
        avg_metrics = {
            "threshold": threshold,
            "avg_watermark_reduction": np.mean([r["watermark_reduction"] for r in threshold_results]),
            "avg_modifications": np.mean([r["num_modifications"] for r in threshold_results]),
            "avg_bleu": np.mean([r["bleu_score"] for r in threshold_results]),
            "avg_prob_increase": np.mean([r["avg_prob_increase"] for r in threshold_results]),
            "watermark_removal_rate": np.mean([not r["modified_is_watermarked"] for r in threshold_results])
        }
        
        results.append({
            "threshold": threshold,
            "detailed_results": threshold_results,
            "average_metrics": avg_metrics
        })
        
        logger.info(f"\nResults for threshold {threshold}:")
        logger.info(f"Average watermark reduction: {avg_metrics['avg_watermark_reduction']:.3f}")
        logger.info(f"Average modifications: {avg_metrics['avg_modifications']:.2f}")
        logger.info(f"Average BLEU score: {avg_metrics['avg_bleu']:.3f}")
        logger.info(f"Average probability increase: {avg_metrics['avg_prob_increase']:.3f}")
        logger.info(f"Watermark removal rate: {avg_metrics['watermark_removal_rate']:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"watermark_attack_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    return results

def main():
    """Main entry point."""
    prompts = [
        "Explain what is artificial intelligence in one paragraph.",
        "What are the main challenges in modern robotics?",
        "Describe the process of photosynthesis briefly."
    ]
    
    # Run experiment
    run_watermark_experiment(
        prompts=prompts,
        watermarked_model_name="facebook/opt-1.3b",
        substitute_type="claude",
        substitute_model_name="gpt2-medium",
        prob_thresholds=[0.05],
        max_length=200
    )

if __name__ == "__main__":
    main()
