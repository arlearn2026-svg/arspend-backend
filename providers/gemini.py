import os
import asyncio
import google.generativeai as genai
from .base import AIProvider

class GeminiFlashLiteProvider(AIProvider):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.model = None
            return
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash-lite")

    async def generate(self, prompt: str) -> str | None:
        if not self.model:
            return None
        try:
            # Run sync generate_content in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            return response.text
        except Exception:
            return None

class GeminiFlashProvider(AIProvider):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.model = None
            return
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def generate(self, prompt: str) -> str | None:
        if not self.model:
            return None
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            return response.text
        except Exception:
            return None