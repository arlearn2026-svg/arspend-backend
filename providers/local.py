from .base import AIProvider

class LocalFallbackProvider(AIProvider):
    async def generate(self, prompt: str) -> str | None:
        if "categorize" in prompt.lower():
            return "Other"
        return "We are unable to generate insights at this moment. Please try again later."