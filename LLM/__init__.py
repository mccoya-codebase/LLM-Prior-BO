"""
LLM client abstraction for prior generation.

Provides unified interface for OpenAI and Gemini API calls.
"""

from .base_client import BaseLLMClient, parse_json_recommendations
from .gemini_client import GeminiClient


def get_llm_client(provider: str, api_key: str = None, model: str = None):
    """
    Factory function to get the appropriate LLM client.

    Args:
        provider: 'openai' or 'gemini'
        api_key: API key (if None, reads from environment)
        model: Model name (if None, uses default for provider)

    Returns:
        LLM client instance
    """
    if provider.lower() == "gemini":
        return GeminiClient(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose 'openai' or 'gemini'.")


__all__ = [
    "BaseLLMClient",
    "GeminiClient",
    "get_llm_client",
    "parse_json_recommendations",
]
