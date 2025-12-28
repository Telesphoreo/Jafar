"""
LLM Provider Interface Module.

Provides a unified interface for interacting with different LLM providers
(OpenAI, Google Generative AI) with easy swapping capability.
"""

from .base import LLMProvider, LLMResponse
from .openai_client import OpenAIProvider
from .google_client import GoogleProvider
from .factory import create_llm_provider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "GoogleProvider",
    "create_llm_provider",
]
