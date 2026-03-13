"""
LLM Provider Factory.

Creates the Gemini LLM provider based on configuration.
"""

import logging

from .base import LLMProvider
from .google_client import GoogleProvider

logger = logging.getLogger("jafar.llm.factory")


def create_llm_provider(
    google_api_key: str = "",
    google_model: str = "gemini-2.0-flash",
) -> LLMProvider:
    """
    Create the Gemini LLM provider.

    Args:
        google_api_key: Google API key.
        google_model: Google model to use.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If API key is not provided.
    """
    if not google_api_key:
        raise ValueError("Google API key is required")

    logger.info(f"Creating LLM provider: Google ({google_model})")
    return GoogleProvider(api_key=google_api_key, model=google_model)
