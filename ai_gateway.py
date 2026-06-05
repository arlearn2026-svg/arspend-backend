from providers.gemini import GeminiFlashLiteProvider, GeminiFlashProvider
from providers.groq_provider import GroqProvider
from providers.openrouter import OpenRouterProvider
from providers.cloudflare import CloudflareProvider
from providers.local import LocalFallbackProvider
from providers.base import AIProvider

class AIGateway:
    def __init__(self):
        self.providers: list[AIProvider] = [
            GeminiFlashLiteProvider(),
            GeminiFlashProvider(),
            GroqProvider(),
            OpenRouterProvider(),
            LocalFallbackProvider(),
        ]

    async def query(self, prompt: str) -> str:
        for provider in self.providers:
            result = await provider.generate(prompt)
            if result:
                return result
        return "Unable to process request."