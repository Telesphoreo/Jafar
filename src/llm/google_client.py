"""
Google Generative AI (Gemini) LLM Provider Implementation.

Uses the new google-genai package (replaces deprecated google-generativeai).
"""

import logging
from typing import Any

from google import genai
from google.genai import types

from .base import LLMProvider, LLMResponse

logger = logging.getLogger("twitter_sentiment.llm.google")


class GoogleProvider(LLMProvider):
    """Google Generative AI provider implementation."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize the Google Generative AI provider.

        Args:
            api_key: Google API key.
            model: Model to use (default: gemini-2.0-flash).
        """
        self._api_key = api_key
        self._model = model
        self._client = None

        if api_key:
            self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "Google"

    @property
    def model_name(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return self._client is not None and bool(self._api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """
        Generate a response using Google's Generative AI API.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt to set context.
            temperature: Creativity setting (0.0-1.0).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse containing the generated content.
        """
        logger.debug(f"Sending request to Google ({self._model})")

        try:
            # Build contents with system instruction if provided
            contents = prompt

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_prompt if system_prompt else None,
            )

            # Use async generation
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            content = response.text if response.text else ""

            # Extract usage metadata if available
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            logger.debug(f"Google response received, tokens used: {usage}")

            return LLMResponse(
                content=content,
                model=self._model,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise
