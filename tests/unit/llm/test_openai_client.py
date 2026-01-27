"""
Unit tests for src/llm/openai_client.py

Tests OpenAI provider with mocked AsyncOpenAI client.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.base import LLMResponse


class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    def test_init(self):
        """Test provider initialization."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")

        assert provider._api_key == "test-key"
        assert provider._model == "gpt-4o"
        assert provider._client is None  # Lazy loaded

    def test_provider_name(self):
        """Test provider name property."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        assert provider.provider_name == "OpenAI"

    def test_model_name(self):
        """Test model name property."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        assert provider.model_name == "gpt-4o-mini"

    def test_is_configured_true(self):
        """Test is_configured returns True with valid key."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        assert provider.is_configured() is True

    def test_is_configured_false(self):
        """Test is_configured returns False without key."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="")
        assert provider.is_configured() is False

    def test_get_client_creates_client(self, mock_openai):
        """Test that _get_client creates AsyncOpenAI client."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        client = provider._get_client()

        assert client is not None
        mock_openai.assert_called_once_with(api_key="test-key")

    def test_get_client_reuses_client(self, mock_openai):
        """Test that _get_client reuses existing client."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        client1 = provider._get_client()
        client2 = provider._get_client()

        assert client1 is client2
        mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_prompt(self, mock_openai):
        """Test generating response with simple prompt."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        response = await provider.generate(
            prompt="What is 2+2?",
            temperature=0.5,
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)
        assert "test response" in response.content.lower()
        assert response.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_with_messages(self, mock_openai):
        """Test generating response with message list."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        response = await provider.generate(messages=messages)

        assert isinstance(response, LLMResponse)
        # Verify messages were passed correctly
        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 3

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, mock_openai):
        """Test generating response with system prompt."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        response = await provider.generate(
            prompt="Analyze this.",
            system_prompt="You are a helpful assistant.",
        )

        # Verify system prompt was prepended
        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_system_prompt_not_duplicated(self, mock_openai):
        """Test that system prompt isn't duplicated if already in messages."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        messages = [
            {"role": "system", "content": "Existing system prompt."},
            {"role": "user", "content": "Hello"},
        ]

        await provider.generate(
            messages=messages,
            system_prompt="New system prompt",  # Should be ignored
        )

        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        call_messages = call_kwargs["messages"]

        # Should only have one system message (the original)
        system_messages = [m for m in call_messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "Existing system prompt."

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, mock_openai):
        """Test generating response with tool definitions."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        await provider.generate(
            prompt="What's the weather in NYC?",
            tools=tools,
        )

        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self, mock_openai):
        """Test that tool calls are extracted from response."""
        from src.llm.openai_client import OpenAIProvider
        from types import SimpleNamespace

        # Setup mock with tool calls
        mock_tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"location": "NYC"}',
            ),
            id="call_123",
            type="function",
        )

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(api_key="test-key")
        response = await provider.generate(prompt="What's the weather?")

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == "get_weather"

    @pytest.mark.asyncio
    async def test_generate_usage_tracking(self, mock_openai):
        """Test that usage metadata is tracked."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        response = await provider.generate(prompt="Test")

        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 50
        assert response.usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_generate_empty_prompt_creates_empty_messages(self, mock_openai):
        """Test that empty prompt works correctly."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        # Call with no prompt and no messages
        await provider.generate()

        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == []

    @pytest.mark.asyncio
    async def test_generate_raw_response_preserved(self, mock_openai):
        """Test that raw response is preserved."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        response = await provider.generate(prompt="Test")

        assert response.raw_response is not None

    @pytest.mark.asyncio
    async def test_generate_handles_api_error(self, mock_openai):
        """Test that API errors are propagated."""
        from src.llm.openai_client import OpenAIProvider

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        provider = OpenAIProvider(api_key="test-key")

        with pytest.raises(Exception, match="API Error"):
            await provider.generate(prompt="Test")


class TestOpenAIProviderModelSettings:
    """Tests for model and parameter settings."""

    @pytest.mark.asyncio
    async def test_temperature_passed(self, mock_openai):
        """Test that temperature is passed correctly."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        await provider.generate(prompt="Test", temperature=0.9)

        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_max_tokens_passed(self, mock_openai):
        """Test that max_tokens is passed correctly."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        await provider.generate(prompt="Test", max_tokens=500)

        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_model_passed(self, mock_openai):
        """Test that model is passed correctly."""
        from src.llm.openai_client import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        await provider.generate(prompt="Test")

        mock_client = mock_openai.return_value
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
