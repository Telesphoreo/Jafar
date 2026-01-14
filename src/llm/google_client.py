"""
Google Generative AI (Gemini) LLM Provider Implementation.

Uses the new google-genai package (replaces deprecated google-generativeai).
"""

import logging
from typing import Any

from google import genai
from google.genai import types

from .base import LLMProvider, LLMResponse

logger = logging.getLogger("jafar.llm.google")


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
                # Extract system prompt if present in messages but not explicitly passed
                for msg in messages:
                    if msg["role"] == "system" and not system_prompt:
                        system_prompt = msg["content"]
                        # Don't add system message to contents for Google (it uses config)
                        continue
                        
                    role = msg["role"]
                    content = msg["content"]
                    
                    # Map generic roles to Google roles
                    if role == "user":
                        google_role = "user"
                    elif role == "assistant":
                        google_role = "model"
                    elif role == "tool":
                        google_role = "function"
                    else:
                        google_role = "user" # Fallback
                        
                    # Handle tool outputs
                    if role == "tool":
                        # For simplicity in this adaptation, pass as simple string if needed
                        # But Google ideally wants specific FunctionResponse parts.
                        # Given we are refactoring mainly for OpenAI compatibility first, and Google SDK is complex,
                        # we'll use text injection for tool outputs in this specific "string-based" flow if needed,
                        # OR valid types.Part object.
                        # For now, let's treat tool output as user (or function) text message.
                        contents.append({"role": "user", "parts": [{"text": f"Tool Output: {str(content)}"}]})
                    else:
                        contents.append({"role": google_role, "parts": [{"text": str(content)}]})

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

            # Use async generation
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            content = response.text if response.text else ""
            
            # Extract tool calls if present
            tool_calls = None
            if response.function_calls:
                 # Standardize to resemble OpenAI's format for easier consumption
                 from types import SimpleNamespace
                 import json
                 
                 tool_calls = []
                 for fc in response.function_calls:
                     # parsed arguments are usually a dict in google-genai, but main.py expects a JSON string
                     args_str = json.dumps(fc.args) if fc.args else "{}"
                     
                     function_obj = SimpleNamespace(
                         name=fc.name,
                         arguments=args_str
                     )
                     
                     tool_calls.append(SimpleNamespace(
                         function=function_obj,
                         id="call_" + fc.name, # Google doesn't provide call IDs in the same way, synthesize one
                         type="function"
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
