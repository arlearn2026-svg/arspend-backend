import os
from groq import AsyncGroq
from .base import AIProvider

class GroqProvider(AIProvider):
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.client = None
            return
        self.client = AsyncGroq(api_key=api_key)

    async def generate(self, prompt: str) -> str | None:
        if not self.client:
            return None
        try:
            chat = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.2,
            )
            return chat.choices[0].message.content
        except Exception:
            return None