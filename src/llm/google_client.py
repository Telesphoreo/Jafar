"""
Google Generative AI (Gemini) LLM Provider Implementation.

Provides integration with Google's Generative AI API for sentiment analysis.
"""

import logging
from typing import Any

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from .base import LLMProvider, LLMResponse

logger = logging.getLogger("twitter_sentiment.llm.google")


class GoogleProvider(LLMProvider):
    """Google Generative AI provider implementation."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        """
        Initialize the Google Generative AI provider.

        Args:
            api_key: Google API key.
            model: Model to use (default: gemini-1.5-pro).
        """
        self._api_key = api_key
        self._model = model
        self._configured = False

        if api_key:
            genai.configure(api_key=api_key)
            self._configured = True

    @property
    def provider_name(self) -> str:
        return "Google"

    @property
    def model_name(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return self._configured and bool(self._api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """
        Generate a response using Google's Generative AI API.

        Note: Google's API is synchronous, but we wrap it for async compatibility.

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
            # Combine system prompt with user prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            model = genai.GenerativeModel(
                model_name=self._model,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            # Google's SDK is synchronous, run in default executor
            response: GenerateContentResponse = await model.generate_content_async(
                full_prompt
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
