from typing import List, Dict, Any
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekWrapper:
    """Wrapper for DeepSeek API"""
    def __init__(self, model: str = "deepseek-chat"):
        """Initialize DeepSeek wrapper with API key and model"""
        # Try to get API key from environment variable first
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # If not found in environment, try to read from config file
        if not self.api_key:
            config_path = os.path.expanduser("~/.probswap/config")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    for line in f:
                        if line.startswith("DEEPSEEK_API_KEY="):
                            self.api_key = line.strip().split("=")[1]
                            break
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment or config file")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        logger.info(f"Initialized DeepSeek wrapper with model: {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_completion(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Get completion from DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting completion from DeepSeek API: {e}")
            raise
