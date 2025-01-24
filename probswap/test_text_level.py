"""
Test word-level ProbSwap attack
"""
import asyncio
import logging
import torch
from probswap.models import ModelWrapper
from probswap.text_level_attack import TextLevelProbSwap

# Set up logging
logging.basicConfig(level=logging.INFO)

async def test_text_level_attack():
    """Test text-level attack."""
    # Initialize attack
    model = ModelWrapper()
    attack = TextLevelProbSwap(model)
    
    # Test cases
    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "I love programming and artificial intelligence.",
        "The weather is beautiful today in the park.",
        "She reads many books in the university library.",
        "The scientists conducted groundbreaking research."
    ]
    
    for text in test_cases:
        print(f"\nTesting text: {text}")
        
        # Print token information
        tokens = model.tokenizer.encode(text)
        for i, token in enumerate(tokens):
            if i < len(text):
                start, end = attack.get_token_char_position(text, i)
                print(f"Token {i}: '{text[start:end]}' (positions {start}-{end})")
        
        # Apply attack
        modified_text, modifications = await attack.attack(text)
        
        # Print modifications
        print("\nModifications:")
        for mod in modifications:
            print(f"- Replaced '{mod['original']}' with '{mod['substitute']}' "
                  f"(probability: {mod['original_prob']:.3f} -> {mod['substitute_prob']:.3f})")
        
        print(f"Final text: {modified_text}\n")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_text_level_attack())
