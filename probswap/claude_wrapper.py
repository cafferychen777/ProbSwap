"""
Claude API wrapper for ProbSwap attack.
"""
from typing import Dict, List, Tuple, Optional
import anthropic
from anthropic import Anthropic
import os
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class ClaudeWrapper:
    """Wrapper for Claude API to use as substitute model in ProbSwap attack."""
    
    def __init__(self, model_name: str = "claude-3-haiku-20240122", target_tokenizer=None):
        """Initialize Claude API client.
        
        Args:
            model_name: The Claude model to use. Default is claude-3-haiku-20240122.
            target_tokenizer: Tokenizer from the target model for correct token handling
        """
        self.model_name = model_name
        self.target_tokenizer = target_tokenizer
        self.client = Anthropic(
            # API key should be set in environment variable ANTHROPIC_API_KEY
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        logger.info(f"Initialized Claude wrapper with model: {model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_next_token_probabilities(self, text: str, temperature: float = 0.0) -> Dict[str, float]:
        """Get probability distribution for the next token after the given text.
        
        Args:
            text: The input text context
            temperature: Sampling temperature (0.0 for deterministic output)
            
        Returns:
            Dictionary mapping tokens to their probabilities
        """
        try:
            # Claude-3 uses system prompt to set behavior
            system_prompt = """You are a helpful assistant. For the given text, you will:
1. Analyze what word or token would naturally come next
2. Return a JSON object with the top 5 most likely next tokens and their probabilities
3. Make sure probabilities sum to 1.0
4. Focus on maintaining natural and fluent text flow"""

            message = f"Given this text: '{text}'\nWhat are the top 5 most likely next tokens and their probabilities? Return only a JSON object."

            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=100,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON object in the response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError(f"No valid JSON found in response: {content}")
                
            json_str = content[start:end]
            probabilities = json.loads(json_str)
            
            # Normalize probabilities if needed
            total = sum(float(p) for p in probabilities.values())
            if total != 1.0:
                probabilities = {k: float(v)/total for k, v in probabilities.items()}
                
            logger.debug(f"Got token probabilities: {probabilities}")
            return probabilities
            
        except Exception as e:
            logger.error(f"Error getting token probabilities: {str(e)}")
            raise

    async def _call_api(self, prompt: str) -> str:
        """Call Claude API with the given prompt.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            Claude's response text
        """
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            raise

    async def get_substitutes(
        self, 
        text: str, 
        token_indices: List[int],
        top_k: int = 5
    ) -> List[List[str]]:
        """Get substitute tokens for the given positions in text.
        
        Args:
            text: Input text
            token_indices: List of token indices to get substitutes for
            top_k: Number of substitutes to return per position
            
        Returns:
            List of lists of substitute tokens, one list per position
        """
        if self.target_tokenizer is None:
            raise ValueError("Target tokenizer not provided")
            
        substitutes = []
        tokens = self.target_tokenizer.encode(text)
        
        # Process positions in batches
        batch_size = 5  # Process 5 positions at a time to avoid rate limits
        for i in range(0, len(token_indices), batch_size):
            batch_indices = token_indices[i:i + batch_size]
            
            # Build prompt for this batch
            prompt = "For each position, I will show you a token in its context. Please suggest natural alternatives that could replace it while maintaining the meaning and fluency of the text. Return the suggestions in a clear format, one per line with a number prefix.\n\n"
            
            for idx in batch_indices:
                # Get token at this position
                token = self.target_tokenizer.decode([tokens[idx]])
                # Get context (decode tokens to get proper text)
                pre_context = self.target_tokenizer.decode(tokens[max(0,idx-20):idx])
                post_context = self.target_tokenizer.decode(tokens[idx+1:min(len(tokens),idx+21)])
                
                prompt += f"Position {idx}:\n"
                prompt += f"Token to replace: '{token}'\n"
                prompt += f"Context: {pre_context}[{token}]{post_context}\n"
                prompt += f"Please suggest {top_k} natural alternatives that would fit well in this context, one per line with a number prefix.\n\n"
            
            # Call Claude API
            response = await self._call_api(prompt)
            
            # Parse response for each position in batch
            batch_substitutes = self._parse_substitutes_response(response, len(batch_indices), top_k)
            substitutes.extend(batch_substitutes)
            
            # Add small delay between batches
            if i + batch_size < len(token_indices):
                await asyncio.sleep(1)
        
        return substitutes
        
    def _parse_substitutes_response(self, response: str, num_positions: int, top_k: int) -> List[List[str]]:
        """Parse Claude's response to extract substitutes for each position."""
        substitutes = []
        current_position = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If we see a number or "Alternative" or "-", it's probably a substitute
            if (line[0].isdigit() and '. ' in line) or line.startswith('Alternative') or line.startswith('-'):
                # Extract the actual word/phrase
                parts = line.split('. ', 1) if '. ' in line else line.split(' ', 1)
                if len(parts) > 1:
                    word = parts[1].strip().strip('"\'')
                    if word and len(current_position) < top_k:
                        current_position.append(word)
                        
            # If we see "Position" or have enough substitutes, start new position
            if line.startswith('Position') or len(current_position) == top_k:
                if current_position:
                    substitutes.append(current_position)
                    current_position = []
                    
        # Add final position if any
        if current_position:
            substitutes.append(current_position)
            
        # Pad any positions that didn't get enough substitutes
        while len(substitutes) < num_positions:
            substitutes.append([])
            
        # Make sure each position has exactly top_k substitutes
        for i in range(len(substitutes)):
            if len(substitutes[i]) < top_k:
                # Pad with empty strings if we don't have enough substitutes
                substitutes[i].extend([''] * (top_k - len(substitutes[i])))
            elif len(substitutes[i]) > top_k:
                # Truncate if we have too many
                substitutes[i] = substitutes[i][:top_k]
        
        return substitutes

    async def batch_get_substitutes(self, text: str, positions: List[int], top_k: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """Get substitutes for multiple positions in parallel.
        
        Args:
            text: The full text
            positions: List of positions to get substitutes for
            top_k: Number of substitutes per position
            
        Returns:
            Dictionary mapping positions to their substitutes
        """
        tasks = []
        for pos in positions:
            task = self.get_substitutes(text, [pos], top_k)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return dict(zip(positions, results))

    def create_prompt_for_natural_substitutes(self, text: str, position: int) -> str:
        """Create a prompt for getting more natural substitutes.
        
        This method creates a more sophisticated prompt that considers context
        and natural language flow.
        """
        # Get some context around the target position
        context_window = 100  # characters
        start = max(0, position - context_window)
        end = min(len(text), position + context_window)
        
        context = text[start:end]
        target_position_in_context = position - start
        
        prompt = f"""Given this text segment (| marks the position of interest):
{context[:target_position_in_context]}|{context[target_position_in_context:]}

What would be natural alternative words or phrases that could replace the text right after the | position?
Consider:
1. Grammar and syntax
2. Context and meaning
3. Style consistency
4. Natural flow

Return a JSON object with the top 5 alternatives and their confidence scores (0-1)."""
        
        return prompt
