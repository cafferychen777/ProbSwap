from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from MarkLLM.watermark.kgw import KGW
from MarkLLM.utils.transformers_config import TransformersConfig

class MarkLLMWrapper:
    """Wrapper for MarkLLM watermarked models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        watermark_processor: KGW
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.watermark_processor = watermark_processor
        
    @classmethod
    def from_markllm(
        cls,
        model_name: str,
        watermark_config: Dict[str, Any]
    ) -> "MarkLLMWrapper":
        """
        Create a wrapper from MarkLLM components.
        
        Args:
            model_name: Name of the base model (e.g. "gpt2")
            watermark_config: Watermark configuration for KGW
            
        Returns:
            MarkLLMWrapper instance
        """
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create transformers config
        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize KGW watermark processor
        watermark_processor = KGW(watermark_config, transformers_config)
        
        return cls(model, tokenizer, watermark_processor)
    
    def generate_watermarked_text(
        self,
        prompt: str,
        max_length: int = 100,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate watermarked text using MarkLLM KGW.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, watermark_info)
        """
        # Generate watermarked text
        generated_text = self.watermark_processor.generate_watermarked_text(
            prompt,
            max_length=max_length,
            **kwargs
        )
        
        # Detect watermark
        watermark_info = self.detect_watermark(generated_text)
        
        return generated_text, watermark_info
    
    def detect_watermark(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Detect watermark in the given text using KGW.
        
        Args:
            text: Text to verify
            
        Returns:
            Dictionary containing detection metrics
        """
        # Detect watermark
        detection_result = self.watermark_processor.detect_watermark(text)
        
        return detection_result
