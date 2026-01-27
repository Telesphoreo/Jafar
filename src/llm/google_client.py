"""
Google Generative AI (Gemini) LLM Provider Implementation.

Uses the new google-genai package (replaces deprecated google-generativeai).
"""

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types

from .base import LLMProvider, LLMResponse

logger = logging.getLogger("jafar.llm.google")

# Retry configuration for transient errors
MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 60.0  # seconds


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    error_str = str(error).lower()
    # Check for common transient error patterns
    retryable_patterns = [
        "503",
        "overloaded",
        "unavailable",
        "resource exhausted",
        "rate limit",
        "quota exceeded",
        "timeout",
        "connection",
        "temporarily",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


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
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """
        Generate a response using Google's Generative AI API.

        Args:
            prompt: The user prompt/question (string).
            messages: List of conversation messages (alternate to prompt).
            system_prompt: Optional system prompt to set context.
            temperature: Creativity setting (0.0-1.0).
            max_tokens: Maximum tokens in response.
            tools: Optional list of tool definitions.

        Returns:
            LLMResponse containing the generated content.
        """
        logger.debug(f"Sending request to Google ({self._model})")

        try:
            # Build contents
            contents = []
            
            # If prompt provided, use it (simplest case)
            if prompt and not messages:
                contents = prompt
            
            # If messages provided, convert to Google format
            elif messages:
                # Collect function responses to batch them after the model message
                pending_function_responses = []

                for msg in messages:
                    if msg["role"] == "system" and not system_prompt:
                        system_prompt = msg["content"]
                        continue

                    role = msg["role"]
                    content = msg["content"]

                    if role == "user":
                        # Flush any pending function responses first
                        if pending_function_responses:
                            contents.append({
                                "role": "user",
                                "parts": pending_function_responses
                            })
                            pending_function_responses = []
                        contents.append({"role": "user", "parts": [{"text": str(content)}]})

                    elif role == "assistant":
                        # Flush any pending function responses first
                        if pending_function_responses:
                            contents.append({
                                "role": "user",
                                "parts": pending_function_responses
                            })
                            pending_function_responses = []

                        # Build model message parts
                        parts = []

                        # Add text content if present (may be empty when model only calls tools)
                        if content:
                            parts.append({"text": str(content)})

                        # Add function calls if present
                        if "tool_calls" in msg and msg["tool_calls"]:
                            for tc in msg["tool_calls"]:
                                import json
                                args = tc.function.arguments
                                if isinstance(args, str):
                                    args = json.loads(args) if args else {}
                                fc_part = {
                                    "function_call": {
                                        "name": tc.function.name,
                                        "args": args
                                    }
                                }
                                # Include thought_signature if present (required for thinking models)
                                if hasattr(tc, "thought_signature") and tc.thought_signature:
                                    fc_part["function_call"]["thought_signature"] = tc.thought_signature
                                parts.append(fc_part)

                        # Only add if we have parts
                        if parts:
                            contents.append({"role": "model", "parts": parts})

                    elif role == "tool":
                        # Accumulate function responses - they must be in a single "user" message
                        func_name = msg.get("name", "unknown")
                        pending_function_responses.append({
                            "function_response": {
                                "name": func_name,
                                "response": {"result": str(content)}
                            }
                        })

                # Flush any remaining function responses
                if pending_function_responses:
                    contents.append({
                        "role": "user",
                        "parts": pending_function_responses
                    })

            # Hand tools
            config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "system_instruction": system_prompt if system_prompt else None,
            }
            
            if tools:
                # Transform OpenAI-style tools to Google GenAI format
                google_tools = []
                function_declarations = []
                
                for tool in tools:
                    if tool.get("type") == "function":
                        func_def = tool.get("function", {})
                        # Ensure parameters are present even if empty, as Google might strictly require valid schema
                        if "parameters" not in func_def:
                            func_def["parameters"] = {"type": "object", "properties": {}}
                        function_declarations.append(func_def)
                
                if function_declarations:
                    # Google GenAI expects a list of Tool objects (or dicts), 
                    # where one Tool can contain multiple function declarations.
                    google_tools.append({"function_declarations": function_declarations})
                    config_kwargs["tools"] = google_tools

            config = types.GenerateContentConfig(**config_kwargs)

            # Use async generation with retry logic for transient errors
            response = None
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=self._model,
                        contents=contents,
                        config=config,
                    )
                    break  # Success, exit retry loop

                except Exception as e:
                    last_error = e

                    if not _is_retryable_error(e):
                        # Non-retryable error, raise immediately
                        logger.error(f"Google API error (non-retryable): {e}")
                        raise

                    if attempt < MAX_RETRIES - 1:
                        # Calculate delay with exponential backoff + jitter
                        delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                        logger.warning(
                            f"Google API transient error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Final attempt failed
                        logger.error(
                            f"Google API error after {MAX_RETRIES} attempts: {e}"
                        )
                        raise

            if response is None:
                raise last_error or RuntimeError("No response received from Google API")

            content = response.text if response.text else ""
            
            # Extract tool calls if present
            tool_calls = None
            if response.function_calls:
                 # Standardize to resemble OpenAI's format for easier consumption
                 from types import SimpleNamespace
                 import json
                 import uuid

                 tool_calls = []

                 # Build a map of function call names to their thought_signatures from raw parts
                 # Some models return thought_signature at the Part level, not the FunctionCall level
                 thought_sig_map = {}
                 if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                     for part in response.candidates[0].content.parts:
                         if hasattr(part, "function_call") and part.function_call:
                             fc_part = part.function_call
                             sig = getattr(part, "thought_signature", None) or getattr(fc_part, "thought_signature", None)
                             if sig and hasattr(fc_part, "name"):
                                 thought_sig_map[fc_part.name] = sig

                 for fc in response.function_calls:
                     # parsed arguments are usually a dict in google-genai, but main.py expects a JSON string
                     args_str = json.dumps(fc.args) if fc.args else "{}"

                     function_obj = SimpleNamespace(
                         name=fc.name,
                         arguments=args_str
                     )

                     # Generate unique ID to avoid collisions when same function is called multiple times
                     call_id = f"call_{fc.name}_{uuid.uuid4().hex[:8]}"

                     # Capture thought_signature if present (required for thinking models)
                     # Check both the function call object and our map from raw parts
                     thought_sig = getattr(fc, "thought_signature", None) or thought_sig_map.get(fc.name)

                     tool_calls.append(SimpleNamespace(
                         function=function_obj,
                         id=call_id,
                         type="function",
                         thought_signature=thought_sig
                     ))

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
                tool_calls=tool_calls,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise
