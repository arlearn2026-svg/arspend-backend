import logging

from providers.groq_provider import GroqProvider
from providers.openrouter import OpenRouterProvider

from providers.local import LocalFallbackProvider
from providers.base import AIProvider

logger = logging.getLogger(__name__)

class AIGateway:
    def __init__(self):
        self.providers = []
        provider_classes = [
            GroqProvider,
            OpenRouterProvider,
            LocalFallbackProvider,
        ]
        for cls in provider_classes:
            try:
                provider = cls()
                self.providers.append(provider)
            except Exception as e:
                logger.warning(f"Failed to initialize {cls.__name__}: {e}")
        # Ensure local fallback is always present
        if not self.providers:
            self.providers.append(LocalFallbackProvider())

    async def query(self, prompt: str) -> str:
        for provider in self.providers:
            try:
                result = await provider.generate(prompt)
                if result:
                    return result
            except Exception:
                continue
        return "Unable to process request."