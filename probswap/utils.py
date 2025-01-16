import torch
from typing import List, Dict, Any
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

def calculate_metrics(
    original_text: str,
    modified_text: str,
    modifications: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate metrics to evaluate the attack.
    
    Args:
        original_text: Original watermarked text
        modified_text: Text after ProbSwap attack
        modifications: List of modifications made
        
    Returns:
        Dictionary containing metrics:
        - num_modifications: Number of tokens modified
        - avg_prob_increase: Average increase in token probability
        - bleu_score: BLEU score between original and modified text
    """
    # Calculate average probability increase
    prob_increases = [
        mod["new_prob"] - mod["original_prob"]
        for mod in modifications
    ]
    avg_prob_increase = np.mean(prob_increases) if prob_increases else 0.0
    
    # Calculate BLEU score
    original_tokens = word_tokenize(original_text)
    modified_tokens = word_tokenize(modified_text)
    bleu_score = sentence_bleu([original_tokens], modified_tokens)
    
    return {
        "num_modifications": len(modifications),
        "avg_prob_increase": avg_prob_increase,
        "bleu_score": bleu_score
    }

def format_results(
    original_text: str,
    modified_text: str,
    modifications: List[Dict[str, Any]],
    metrics: Dict[str, float]
) -> str:
    """Format attack results for display."""
    result = "ProbSwap Attack Results\n"
    result += "=" * 50 + "\n\n"
    
    result += "Original Text:\n"
    result += original_text + "\n\n"
    
    result += "Modified Text:\n"
    result += modified_text + "\n\n"
    
    result += "Modifications:\n"
    for mod in modifications:
        result += f"Position {mod['position']}: "
        result += f"'{mod['original']}' ({mod['original_prob']:.3f}) -> "
        result += f"'{mod['replacement']}' ({mod['new_prob']:.3f})\n"
    result += "\n"
    
    result += "Metrics:\n"
    result += f"Number of modifications: {metrics['num_modifications']}\n"
    result += f"Average probability increase: {metrics['avg_prob_increase']:.3f}\n"
    result += f"BLEU score: {metrics['bleu_score']:.3f}\n"
    
    return result
