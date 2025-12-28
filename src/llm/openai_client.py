"""
OpenAI LLM Provider Implementation.

Provides integration with OpenAI's API (GPT-4o, etc.) for sentiment analysis.
"""

import logging
from typing import Any

from openai import AsyncOpenAI

from .base import LLMProvider, LLMResponse

logger = logging.getLogger("twitter_sentiment.llm.openai")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model to use (default: gpt-4o).
        """
        self._api_key = api_key
        self._model = model
        self._client: AsyncOpenAI | None = None

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model_name(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """
        Generate a response using OpenAI's API.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt to set context.
            temperature: Creativity setting (0.0-1.0).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse containing the generated content.
        """
        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Sending request to OpenAI ({self._model})")

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content or ""
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            logger.debug(f"OpenAI response received, tokens used: {usage}")

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
