import os
import google.generativeai as genai
from .base import AIProvider

class GeminiFlashLiteProvider(AIProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash-lite")

    async def generate(self, prompt: str) -> str | None:
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception:
            return None

class GeminiFlashProvider(AIProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def generate(self, prompt: str) -> str | None:
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception:
            return None