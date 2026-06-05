import logging

import logging
from providers.groq_provider import GroqProvider
from providers.openrouter import OpenRouterProvider
from providers.local import LocalFallbackProvider
from providers.base import AIProvider

logger = logging.getLogger(__name__)

class AIGateway:
    def __init__(self):
        self.providers = []
        provider_classes = [GroqProvider, OpenRouterProvider, LocalFallbackProvider]
        for cls in provider_classes:
            try:
                provider = cls()
                self.providers.append(provider)
                logger.info(f"Initialized provider: {cls.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize {cls.__name__}: {e}")
        if not self.providers:
            self.providers.append(LocalFallbackProvider())

    async def query(self, prompt: str) -> str:
        for provider in self.providers:
            try:
                result = await provider.generate(prompt)
                if result:
                    logger.info(f"Successful response from {provider.__class__.__name__}")
                    return result
                else:
                    logger.warning(f"Provider {provider.__class__.__name__} returned None")
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} raised: {e}")
        return "Unable to process request."