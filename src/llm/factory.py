"""
LLM Provider Factory.

Creates the appropriate LLM provider based on configuration.
"""

import logging
from typing import Literal

from .base import LLMProvider
from .openai_client import OpenAIProvider
from .google_client import GoogleProvider

logger = logging.getLogger("twitter_sentiment.llm.factory")


def create_llm_provider(
    provider: Literal["openai", "google"],
    openai_api_key: str = "",
    openai_model: str = "gpt-4o",
    google_api_key: str = "",
    google_model: str = "gemini-1.5-pro",
) -> LLMProvider:
    """
    Create an LLM provider based on the specified type.

    This factory function allows easy swapping between different LLM providers
    by simply changing the provider parameter in configuration.

    Args:
        provider: Which provider to use ("openai" or "google").
        openai_api_key: OpenAI API key (required if provider is "openai").
        openai_model: OpenAI model to use.
        google_api_key: Google API key (required if provider is "google").
        google_model: Google model to use.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider is not supported or not properly configured.
    """
    logger.info(f"Creating LLM provider: {provider}")

    if provider == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        return OpenAIProvider(api_key=openai_api_key, model=openai_model)

    elif provider == "google":
        if not google_api_key:
            raise ValueError("Google API key is required when using Google provider")
        return GoogleProvider(api_key=google_api_key, model=google_model)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
