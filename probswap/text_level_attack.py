"""
Text-level ProbSwap attack implementation.
"""
from typing import List, Tuple, Dict, Optional, Union
import torch
from probswap.models import ModelWrapper
from probswap.claude_wrapper import ClaudeWrapper
from probswap.deepseek_wrapper import DeepSeekWrapper
import logging
import jieba
import re
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class TextLevelProbSwap:
    """Text-level substitution attack using ProbSwap technique."""
    
    def __init__(
        self,
        model: ModelWrapper,
        llm_model: str = "deepseek",
        threshold: float = 0.1,
        top_k_substitutes: int = 5,
        batch_size: int = 32,
        language: str = 'auto'  # 'en', 'zh', or 'auto'
    ):
        """Initialize text-level attack.
        
        Args:
            model: Model wrapper for getting token probabilities
            llm_model: Which LLM to use for word substitutions ("deepseek" or "claude")
            threshold: Probability threshold for token replacement
            top_k_substitutes: Number of substitute candidates to consider
            batch_size: Batch size for processing tokens
            language: Text language ('en', 'zh', or 'auto' for auto-detection)
        """
        self.model = model
        self.threshold = threshold
        self.llm_model = llm_model.lower()
        if self.llm_model == "deepseek":
            self.llm = DeepSeekWrapper()
        elif self.llm_model == "claude":
            self.llm = ClaudeWrapper()
        else:
            raise ValueError(f"Unknown LLM model: {llm_model}. Must be 'deepseek' or 'claude'")
        self.top_k_substitutes = top_k_substitutes
        self.batch_size = batch_size
        self.language = language
        self.device = model.device
        
    def detect_language(self, text: str) -> str:
        """Detect text language.
        
        Args:
            text: Input text
            
        Returns:
            'en' for English, 'zh' for Chinese
        """
        # Simple detection based on character ranges
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        return 'zh' if chinese_chars > english_chars else 'en'
    
    def get_token_char_position(self, text: str, position: int) -> Tuple[int, int]:
        """Get character position for a token position.
        
        Args:
            text: Input text
            position: Token position
            
        Returns:
            Tuple of (start_pos, end_pos) in the original text
        """
        # First get the character position from token position
        tokens = self.model.tokenizer.encode(text)
        char_positions = []
        current_pos = 0
        
        # Map each token to its character position
        for i, token in enumerate(tokens):
            token_text = self.model.tokenizer.decode([token])
            token_len = len(token_text.lstrip())  # Remove leading spaces
            if token_text.startswith(' '):
                current_pos += 1
            char_positions.append((current_pos, current_pos + token_len))
            current_pos += token_len
        
        if position >= len(char_positions):
            return (len(text), len(text))
            
        return char_positions[position]
    
    def _find_word_boundaries(self, text: str, token_position: int) -> Tuple[int, int]:
        """Find the word boundaries for a given token position.
        
        Args:
            text: Input text
            token_position: Position of the token in the tokenized text
            
        Returns:
            Tuple of (start_pos, end_pos) in the original text
        """
        # First get the character position from token position
        tokens = self.model.tokenizer.encode(text)
        char_positions = []
        current_pos = 0
        
        # Build mapping from tokens to character positions
        for i, token in enumerate(tokens):
            token_text = self.model.tokenizer.decode([token])
            token_len = len(token_text.lstrip())  # Remove leading spaces
            if token_text.startswith(' '):
                current_pos += 1
            char_positions.append((current_pos, current_pos + token_len))
            current_pos += token_len
            
        # Get character position for our target token
        if token_position >= len(char_positions):
            raise ValueError(f"Token position {token_position} out of range")
        char_start, char_end = char_positions[token_position]
        
        # Detect language if set to auto
        lang = self.detect_language(text) if self.language == 'auto' else self.language
        
        if lang == 'zh':
            # For Chinese, use jieba to find word boundaries
            words = list(jieba.cut(text))
            current_pos = 0
            for word in words:
                word_end = current_pos + len(word)
                if current_pos <= char_start < word_end:
                    return current_pos, word_end
                current_pos = word_end
        else:
            # For English, use NLTK's word_tokenize
            words = word_tokenize(text)
            current_pos = 0
            for word in words:
                # Account for spaces between words
                if current_pos > 0:
                    current_pos += 1
                word_end = current_pos + len(word)
                if current_pos <= char_start < word_end:
                    return current_pos, word_end
                current_pos = word_end
                
        # Fallback: return the token's character positions
        return char_start, char_end
    
    def get_token_probabilities(self, tokens: List[int]) -> List[float]:
        """Get probabilities for each token from target model."""
        with torch.no_grad():
            inputs = torch.tensor([tokens], device=self.device)
            outputs = self.model.model(inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            # Get probability for each token at its position
            token_probs = []
            for i, token in enumerate(tokens):
                if i < len(logits):
                    token_probs.append(probs[i, token].item())
                else:
                    token_probs.append(0.0)  # Default probability for out-of-range tokens
        return token_probs
    
    async def get_word_substitutes(self, word: str, context: str) -> List[str]:
        """Get substitute words from LLM.
        
        Args:
            word: Word to find substitutes for
            context: Context where the word appears
            
        Returns:
            List of substitute words
        """
        # Construct prompt
        prompt = f"""Given the word "{word}" in the context "{context}", suggest alternative words or short phrases that could replace it while preserving the meaning and fluency.
Return only the substitutes, one per line, without numbering or explanation."""

        # Get suggestions from LLM
        if self.llm_model == "deepseek":
            response = await self.llm.get_completion(prompt)
        else:
            response = await self.llm._call_api(prompt)
        
        # Parse response to get substitutes
        substitutes = [
            line.strip() for line in response.split('\n')
            if line.strip() and not line.strip().isdigit()
        ]
        
        return substitutes[:self.top_k_substitutes]
    
    def _filter_substitutes(self, substitutes: List[str], original_word: str) -> List[str]:
        """Filter out invalid substitutes.
        
        Args:
            substitutes: List of substitute words
            original_word: Original word being replaced
            
        Returns:
            Filtered list of substitutes
        """
        filtered = []
        seen = set()
        
        for sub in substitutes:
            # Skip empty strings
            if not sub.strip():
                continue
                
            # Skip duplicates
            if sub in seen:
                continue
                
            # Skip if too similar to original
            if sub == original_word:
                continue
                
            # Skip if too long
            if len(sub.split()) > 3:  # Limit phrase length
                continue
                
            # Skip if contains original word
            if original_word in sub.split():
                continue
                
            filtered.append(sub)
            seen.add(sub)
            
            # Limit number of substitutes
            if len(filtered) >= 5:
                break
                
        return filtered
    
    async def attack(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply text-level ProbSwap attack to watermarked text.
        
        Args:
            text: Input text to attack
            
        Returns:
            Tuple of (modified text, list of modifications)
        """
        tokens = self.model.tokenizer.encode(text)
        token_probs = self.get_token_probabilities(tokens)
        
        # Find tokens with low probabilities
        low_prob_indices = [i for i, p in enumerate(token_probs) if p < 0.1]
        logger.info(f"Found {len(low_prob_indices)} tokens below probability threshold 0.1")
        
        if not low_prob_indices:
            logger.info("No tokens found below probability threshold, returning original text")
            return text, []
        
        # Track modifications
        modifications = []
        modified_text = text
        
        # Process each low probability position
        for idx in low_prob_indices:
            # Find word boundaries
            word_start, word_end = self._find_word_boundaries(modified_text, idx)
            original_word = modified_text[word_start:word_end]
            
            # Get word substitutes
            before_context = modified_text[:word_start].strip()
            after_context = modified_text[word_end:].strip()
            substitutes = await self.get_word_substitutes(original_word, before_context + ' ' + after_context)
            if not substitutes:
                continue
                
            # Filter substitutes
            substitutes = self._filter_substitutes(substitutes, original_word)
            if not substitutes:
                continue
                
            # Try each substitute
            best_substitute = None
            best_prob = token_probs[idx]
            best_substitute_prob = 0.0
            
            for substitute in substitutes:
                # Create temporary text with this substitute
                temp_text = modified_text[:word_start] + substitute + modified_text[word_end:]
                temp_tokens = self.model.tokenizer.encode(temp_text)
                
                # Get probability for the modified position
                temp_probs = self.get_token_probabilities(temp_tokens)
                # Use the average probability of the tokens that replaced the original token
                if idx < len(temp_probs):
                    substitute_prob = sum(temp_probs[idx:min(idx+len(substitute.split()), len(temp_probs))]) / len(substitute.split())
                else:
                    continue
                
                if substitute_prob > best_prob:
                    best_prob = substitute_prob
                    best_substitute = substitute
                    best_substitute_prob = substitute_prob
            
            if best_substitute:
                # Record the modification
                modifications.append({
                    'original': original_word,
                    'substitute': best_substitute,
                    'original_prob': token_probs[idx],
                    'substitute_prob': best_substitute_prob
                })
                
                # Update text and probabilities
                modified_text = modified_text[:word_start] + best_substitute + modified_text[word_end:]
                logger.info(f"Replaced '{original_word}' with '{best_substitute}' at position {idx} "
                          f"(probability: {token_probs[idx]:.3f} -> {best_substitute_prob:.3f})")
        
        logger.info(f"Made {len(modifications)} text-level substitutions")
        return modified_text, modifications
