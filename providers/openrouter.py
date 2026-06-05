import os
from openai import AsyncOpenAI
from .base import AIProvider

class OpenRouterProvider(AIProvider):
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
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
                model="google/gemini-2.0-flash-lite-preview-02-05:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return completion.choices[0].message.content
        except Exception:
            return None