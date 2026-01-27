"""
Abstract base class for LLM providers.

Defines the interface that all LLM providers must implement,
allowing easy swapping between different AI services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    model: str
    usage: dict[str, int] | None = None
    tool_calls: list[Any] | None = None
    raw_response: Any = None
    raw_content: Any = None  # For Google: preserves Content object with thought_signature

    @property
    def token_count(self) -> int:
        """Return total tokens used if available."""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this interface to add support for new LLM services.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model being used."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/question (string).
            messages: List of conversation messages (alternate to prompt).
            system_prompt: Optional system prompt to set context.
            temperature: Creativity setting (0.0-1.0).
            max_tokens: Maximum tokens in response.
            tools: Optional list of tool definitions (schema depends on provider).

        Returns:
            LLMResponse containing the generated content.
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured with API keys."""
        pass
