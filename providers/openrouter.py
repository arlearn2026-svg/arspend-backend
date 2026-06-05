import os
import logging
from openai import AsyncOpenAI
from .base import AIProvider

logger = logging.getLogger(__name__)

class OpenRouterProvider(AIProvider):
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not set – OpenRouter unavailable")
            self.client = None
            return
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    async def generate(self, prompt: str) -> str | None:
        if not self.client:
            return None
        try:
            completion = await self.client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",  # ← free model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            return None