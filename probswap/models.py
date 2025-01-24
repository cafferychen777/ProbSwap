from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, Tuple, List

class ModelWrapper:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize model wrapper for easier interaction with LLMs.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        
    def get_token_probabilities(
        self,
        text: str,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get probability distribution for each position in the text.
        
        Args:
            text: Input text
            temperature: Sampling temperature
            
        Returns:
            Tensor of probability distributions
        """
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits / temperature
            probs = torch.softmax(logits, dim=-1)
        return probs
    
    def generate_alternatives(
        self,
        text: str,
        position: int,
        num_alternatives: int = 5,
        temperature: float = 1.0
    ) -> List[Tuple[str, float]]:
        """
        Generate alternative tokens for a specific position.
        
        Args:
            text: Input text
            position: Position to generate alternatives for
            num_alternatives: Number of alternatives to generate
            temperature: Sampling temperature
            
        Returns:
            List of (token, probability) tuples
        """
        probs = self.get_token_probabilities(text, temperature)[0, position]
        top_k = torch.topk(probs, num_alternatives)
        
        alternatives = []
        for prob, idx in zip(top_k.values, top_k.indices):
            token = self.tokenizer.decode([idx])
            alternatives.append((token, prob.item()))
            
        return alternatives
