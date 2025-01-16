import torch
from typing import List, Tuple, Dict, Optional, Union
from .models import ModelWrapper
from .claude_wrapper import ClaudeWrapper
import asyncio
import logging

logger = logging.getLogger(__name__)

class ProbSwapAttack:
    """ProbSwap attack implementation."""
    
    def __init__(
        self,
        target_model: torch.nn.Module,
        target_tokenizer,
        substitute_model: Union[ModelWrapper, ClaudeWrapper],
        prob_threshold: float = 0.1,
        top_k_substitutes: int = 5,
        batch_size: int = 32
    ):
        """Initialize ProbSwap attack.
        
        Args:
            target_model: The model that generated the watermarked text
            target_tokenizer: Tokenizer for the target model
            substitute_model: Model to use for finding substitutes (local model or Claude)
            prob_threshold: Probability threshold for token replacement
            top_k_substitutes: Number of substitute candidates to consider
            batch_size: Batch size for processing tokens
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.substitute_model = substitute_model
        self.prob_threshold = prob_threshold
        self.top_k_substitutes = top_k_substitutes
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_model.to(self.device)
        
        # Check if using Claude API
        self.using_claude = isinstance(substitute_model, ClaudeWrapper)
        logger.info(f"Initialized ProbSwap attack with {'Claude API' if self.using_claude else 'local model'}")

    def get_target_probabilities(self, text: str, position: int) -> Dict[str, float]:
        """Get probability distribution from target model at position."""
        inputs = self.target_tokenizer(text[:position], return_tensors="pt")
        with torch.no_grad():
            outputs = self.target_model(**inputs)
            probs = torch.softmax(outputs.logits[0, -1], dim=0)
            
        # Convert to dictionary
        token_probs = {}
        for token_id in range(len(self.target_tokenizer)):
            prob = probs[token_id].item()
            if prob > 0.01:  # Filter very low probabilities
                token = self.target_tokenizer.decode([token_id])
                token_probs[token] = prob
                
        return token_probs

    async def get_substitutes(self, text: str, position: int) -> List[Tuple[str, float]]:
        """Get substitute candidates for position."""
        if self.using_claude:
            return await self.substitute_model.get_substitutes(text, position, self.top_k_substitutes)
        else:
            # Use local model
            inputs = self.substitute_model.tokenizer(text[:position], return_tensors="pt")
            with torch.no_grad():
                outputs = self.substitute_model.model(**inputs)
                probs = torch.softmax(outputs.logits[0, -1], dim=0)
                
            # Get top-k substitutes
            values, indices = torch.topk(probs, self.top_k_substitutes)
            substitutes = []
            for value, index in zip(values, indices):
                token = self.substitute_model.tokenizer.decode([index])
                substitutes.append((token, value.item()))
            return substitutes

    def get_token_probabilities(self, tokens: List[int]) -> List[float]:
        """Get probabilities for each token from target model."""
        with torch.no_grad():
            inputs = torch.tensor([tokens]).to(self.device)
            outputs = self.target_model(inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            token_probs = [probs[i, token].item() for i, token in enumerate(tokens)]
        return token_probs

    async def attack(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Apply ProbSwap attack to watermarked text.
        
        Args:
            text: Input text to attack
            
        Returns:
            Tuple of (modified text, list of modifications)
        """
        tokens = self.target_tokenizer.encode(text)
        token_probs = self.get_token_probabilities(tokens)
        
        # Find tokens with low probabilities
        low_prob_indices = [i for i, p in enumerate(token_probs) if p < self.prob_threshold]
        logger.info(f"Found {len(low_prob_indices)} tokens below probability threshold {self.prob_threshold}")
        
        if not low_prob_indices:
            logger.info("No tokens found below probability threshold, returning original text")
            return text, []
            
        # Get substitutes for all low probability tokens
        logger.info("Getting substitutes from model...")
        substitutes = await self.substitute_model.get_substitutes(
            text=text,
            token_indices=low_prob_indices,
            top_k=self.top_k_substitutes
        )
        
        # Track modifications
        modifications = []
        modified_tokens = tokens.copy()
        
        # Apply substitutions
        for idx, token_substitutes in zip(low_prob_indices, substitutes):
            if not token_substitutes:
                logger.debug(f"No substitutes found for token at position {idx}")
                continue
                
            # Find best substitute
            best_substitute = None
            best_prob = token_probs[idx]  # Initialize with current token's probability
            orig_token = self.target_tokenizer.decode([tokens[idx]])
            
            logger.debug(f"Processing token '{orig_token}' at position {idx} with probability {token_probs[idx]:.3f}")
            
            for substitute in token_substitutes:
                # Skip empty substitutes or if substitute is the same as original
                if not substitute or substitute == orig_token:
                    continue
                    
                # Get probability of substitute
                temp_tokens = modified_tokens.copy()
                temp_tokens[idx] = self.target_tokenizer.encode(substitute)[0]
                prob = self.get_token_probabilities([temp_tokens[idx]])[0]
                
                logger.debug(f"  Substitute '{substitute}' has probability {prob:.3f}")
                
                if prob > best_prob:
                    best_prob = prob
                    best_substitute = substitute
            
            # Apply best substitute if found
            if best_substitute and best_prob > token_probs[idx]:
                modified_tokens[idx] = self.target_tokenizer.encode(best_substitute)[0]
                logger.info(f"Replaced token '{orig_token}' with '{best_substitute}' at position {idx} "
                          f"(probability: {token_probs[idx]:.3f} -> {best_prob:.3f})")
                modifications.append({
                    "index": idx,
                    "original": orig_token,
                    "substitute": best_substitute,
                    "original_prob": token_probs[idx],
                    "new_prob": best_prob
                })
        
        modified_text = self.target_tokenizer.decode(modified_tokens)
        logger.info(f"Made {len(modifications)} substitutions")
        return modified_text, modifications
