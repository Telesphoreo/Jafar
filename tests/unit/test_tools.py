"""
Unit tests for src/tools.py

Tests tool registry and tool execution with mocked dependencies.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools import ToolRegistry
from src.temporal_analyzer import TrendTimeline


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_init_empty(self):
        """Test registry initialization with no dependencies."""
        registry = ToolRegistry()

        assert registry.fact_checker is None
        assert registry.memory is None
        assert registry.temporal_analyzer is None
        assert registry.trend_timelines == {}

    def test_init_with_dependencies(self):
        """Test registry initialization with dependencies."""
        mock_fact_checker = MagicMock()
        mock_memory = MagicMock()
        mock_temporal = MagicMock()

        registry = ToolRegistry(
            fact_checker=mock_fact_checker,
            memory=mock_memory,
            temporal_analyzer=mock_temporal,
            trend_timelines={"$NVDA": MagicMock()},
        )

        assert registry.fact_checker is mock_fact_checker
        assert registry.memory is mock_memory
        assert registry.temporal_analyzer is mock_temporal
        assert "$NVDA" in registry.trend_timelines

    def test_get_definitions_minimal(self):
        """Test getting tool definitions with no dependencies (only weather available)."""
        registry = ToolRegistry(enable_web_search=False)
        definitions = registry.get_definitions()

        # Weather tool is always available
        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "get_weather_forecast"

    def test_get_definitions_with_web_search(self):
        """Test that web search tool is included when available."""
        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock()}):
            with patch("duckduckgo_search.DDGS") as mock_ddgs_class:
                mock_ddgs_class.return_value = MagicMock()

                # Create registry with web search enabled
                registry = ToolRegistry(enable_web_search=True)
                # Force enable since mock may not trigger correctly in __init__
                registry.enable_web_search = True
                registry.ddgs = MagicMock()

                definitions = registry.get_definitions()

                tool_names = [t["function"]["name"] for t in definitions]
                assert "search_web" in tool_names

    def test_get_definitions_with_fact_checker(self, mock_yfinance):
        """Test that market data tool is included with fact checker."""
        from src.fact_checker import MarketFactChecker

        fact_checker = MarketFactChecker()
        registry = ToolRegistry(fact_checker=fact_checker, enable_web_search=False)

        definitions = registry.get_definitions()

        tool_names = [t["function"]["name"] for t in definitions]
        assert "get_market_data" in tool_names

    def test_get_definitions_with_memory(self):
        """Test that historical parallels tool is included with memory."""
        mock_memory = MagicMock()
        registry = ToolRegistry(memory=mock_memory, enable_web_search=False)

        definitions = registry.get_definitions()

        tool_names = [t["function"]["name"] for t in definitions]
        assert "search_historical_parallels" in tool_names

    def test_get_definitions_with_temporal(self):
        """Test that trend timeline tool is included with temporal analyzer."""
        mock_temporal = MagicMock()
        mock_timelines = {"$NVDA": MagicMock()}

        registry = ToolRegistry(
            temporal_analyzer=mock_temporal,
            trend_timelines=mock_timelines,
            enable_web_search=False,
        )

        definitions = registry.get_definitions()

        tool_names = [t["function"]["name"] for t in definitions]
        assert "get_trend_timeline" in tool_names

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error message."""
        registry = ToolRegistry(enable_web_search=False)
        result = await registry.execute("unknown_tool", {})

        assert "not found" in result.lower() or "missing" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_web_search(self):
        """Test executing web search tool."""
        # Create registry and manually set up web search
        registry = ToolRegistry(enable_web_search=False)
        registry.enable_web_search = True

        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "body": "Body 1", "href": "https://example.com"}
        ]
        registry.ddgs = mock_ddgs

        result = await registry.execute("search_web", {"query": "test query"})

        assert "Web Search Results" in result
        assert "test query" in result

    @pytest.mark.asyncio
    async def test_execute_web_search_disabled(self):
        """Test web search when disabled."""
        registry = ToolRegistry(enable_web_search=False)
        result = await registry.execute("search_web", {"query": "test"})

        assert "not enabled" in result.lower() or "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_get_market_data(self, mock_yfinance):
        """Test executing market data tool."""
        from src.fact_checker import MarketFactChecker

        fact_checker = MarketFactChecker()

        # Mock the fetch to return empty (tested separately)
        with patch.object(fact_checker, "fetch_market_data", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {}

            registry = ToolRegistry(fact_checker=fact_checker, enable_web_search=False)
            result = await registry.execute("get_market_data", {"symbols": ["NVDA"]})

            # Should call fetch_market_data
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_get_market_data_no_symbols(self, mock_yfinance):
        """Test market data tool with no symbols."""
        from src.fact_checker import MarketFactChecker

        fact_checker = MarketFactChecker()
        registry = ToolRegistry(fact_checker=fact_checker, enable_web_search=False)

        result = await registry.execute("get_market_data", {"symbols": []})

        assert "No symbols" in result

    @pytest.mark.asyncio
    async def test_execute_historical_parallels(self):
        """Test executing historical parallels tool."""
        mock_memory = MagicMock()
        mock_memory.find_parallels = AsyncMock(return_value=[])
        mock_memory.format_parallels_for_llm = AsyncMock(return_value="No parallels found")

        registry = ToolRegistry(memory=mock_memory, enable_web_search=False)
        result = await registry.execute(
            "search_historical_parallels",
            {"query": "silver shortage"},
        )

        mock_memory.find_parallels.assert_called_once()
        assert "No" in result or "parallel" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_get_trend_timeline(self):
        """Test executing trend timeline tool."""
        mock_temporal = MagicMock()
        mock_temporal.format_context_for_llm.return_value = "## Temporal Context\n$NVDA: Day 3"

        mock_timeline = TrendTimeline(
            term="$NVDA",
            term_normalized="nvda",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=50,
            engagement_today=10000.0,
            consecutive_days=3,
        )

        registry = ToolRegistry(
            temporal_analyzer=mock_temporal,
            trend_timelines={"$NVDA": mock_timeline},
            enable_web_search=False,
        )

        result = await registry.execute("get_trend_timeline", {"trend": "$NVDA"})

        assert "Temporal Context" in result
        mock_temporal.format_context_for_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_get_trend_timeline_not_found(self):
        """Test trend timeline for unknown trend."""
        mock_temporal = MagicMock()

        registry = ToolRegistry(
            temporal_analyzer=mock_temporal,
            trend_timelines={},
            enable_web_search=False,
        )

        result = await registry.execute("get_trend_timeline", {"trend": "$UNKNOWN"})

        assert "not found" in result.lower() or "No timeline" in result

    @pytest.mark.asyncio
    async def test_execute_handles_errors(self):
        """Test that execute handles errors gracefully."""
        mock_memory = MagicMock()
        mock_memory.find_parallels = AsyncMock(side_effect=Exception("Test error"))

        registry = ToolRegistry(memory=mock_memory, enable_web_search=False)
        result = await registry.execute(
            "search_historical_parallels",
            {"query": "test"},
        )

        assert "Error" in result
        assert "Test error" in result


class TestToolDefinitionSchemas:
    """Tests for tool definition schemas."""

    def test_web_search_schema(self):
        """Test web search tool schema."""
        # Create registry and manually enable web search
        registry = ToolRegistry(enable_web_search=False)
        registry.enable_web_search = True
        registry.ddgs = MagicMock()

        definitions = registry.get_definitions()
        web_search = next(
            (t for t in definitions if t["function"]["name"] == "search_web"),
            None,
        )

        assert web_search is not None
        assert web_search["type"] == "function"
        assert "query" in web_search["function"]["parameters"]["properties"]
        assert "query" in web_search["function"]["parameters"]["required"]

    def test_market_data_schema(self, mock_yfinance):
        """Test market data tool schema."""
        from src.fact_checker import MarketFactChecker

        fact_checker = MarketFactChecker()
        registry = ToolRegistry(fact_checker=fact_checker, enable_web_search=False)

        definitions = registry.get_definitions()
        market_data = next(
            (t for t in definitions if t["function"]["name"] == "get_market_data"),
            None,
        )

        assert market_data is not None
        assert "symbols" in market_data["function"]["parameters"]["properties"]
        assert market_data["function"]["parameters"]["properties"]["symbols"]["type"] == "array"

    def test_historical_parallels_schema(self):
        """Test historical parallels tool schema."""
        mock_memory = MagicMock()
        registry = ToolRegistry(memory=mock_memory, enable_web_search=False)

        definitions = registry.get_definitions()
        parallels = next(
            (t for t in definitions if t["function"]["name"] == "search_historical_parallels"),
            None,
        )

        assert parallels is not None
        assert "query" in parallels["function"]["parameters"]["properties"]

    def test_trend_timeline_schema(self):
        """Test trend timeline tool schema."""
        mock_temporal = MagicMock()
        registry = ToolRegistry(
            temporal_analyzer=mock_temporal,
            trend_timelines={"$NVDA": MagicMock()},
            enable_web_search=False,
        )

        definitions = registry.get_definitions()
        timeline = next(
            (t for t in definitions if t["function"]["name"] == "get_trend_timeline"),
            None,
        )

        assert timeline is not None
        assert "trend" in timeline["function"]["parameters"]["properties"]

    def test_weather_forecast_schema(self):
        """Test weather forecast tool schema."""
        registry = ToolRegistry(enable_web_search=False)

        definitions = registry.get_definitions()
        weather = next(
            (t for t in definitions if t["function"]["name"] == "get_weather_forecast"),
            None,
        )

        assert weather is not None
        assert weather["type"] == "function"
        assert "cities" in weather["function"]["parameters"]["properties"]
        assert weather["function"]["parameters"]["properties"]["cities"]["type"] == "array"
        assert "cities" in weather["function"]["parameters"]["required"]


class TestWeatherTool:
    """Tests for weather forecast tool."""

    def test_weather_tool_always_available(self):
        """Test that weather tool is always included in definitions."""
        registry = ToolRegistry(enable_web_search=False)
        definitions = registry.get_definitions()

        tool_names = [t["function"]["name"] for t in definitions]
        assert "get_weather_forecast" in tool_names

    @pytest.mark.asyncio
    async def test_execute_weather_no_cities(self):
        """Test weather tool with no cities."""
        registry = ToolRegistry(enable_web_search=False)
        result = await registry.execute("get_weather_forecast", {"cities": []})

        assert "No cities" in result

    @pytest.mark.asyncio
    async def test_execute_weather_success(self):
        """Test weather tool with mocked API response."""
        registry = ToolRegistry(enable_web_search=False)

        # Mock the aiohttp session
        with patch("src.tools.aiohttp.ClientSession") as mock_session_class:
            # Create mock responses
            geo_data = {
                "results": [{
                    "name": "Houston",
                    "admin1": "Texas",
                    "country": "United States",
                    "latitude": 29.76,
                    "longitude": -95.36,
                }]
            }
            weather_data = {
                "current": {
                    "temperature_2m": 32,
                    "apparent_temperature": 25,
                    "weather_code": 75,
                    "wind_speed_10m": 15,
                    "wind_gusts_10m": 25,
                    "relative_humidity_2m": 85,
                },
                "daily": {
                    "time": ["2024-01-26", "2024-01-27"],
                    "weather_code": [75, 71],
                    "temperature_2m_max": [35, 40],
                    "temperature_2m_min": [28, 32],
                    "precipitation_sum": [2.5, 0.5],
                    "precipitation_probability_max": [90, 40],
                }
            }

            # Create async context manager mock for responses
            call_count = [0]

            class MockResponse:
                def __init__(self, data):
                    self.status = 200
                    self._data = data

                async def json(self):
                    return self._data

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            class MockSession:
                def get(self, url, params=None):
                    if "geocoding" in url:
                        return MockResponse(geo_data)
                    else:
                        return MockResponse(weather_data)

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            mock_session_class.return_value = MockSession()

            result = await registry.execute("get_weather_forecast", {"cities": ["Houston"]})

            assert "Weather Forecast" in result
            assert "Houston" in result
            assert "Texas" in result
            assert "Heavy snow" in result  # Weather code 75

    @pytest.mark.asyncio
    async def test_execute_weather_city_not_found(self):
        """Test weather tool with unknown city."""
        registry = ToolRegistry(enable_web_search=False)

        with patch("src.tools.aiohttp.ClientSession") as mock_session_class:
            class MockResponse:
                def __init__(self):
                    self.status = 200

                async def json(self):
                    return {"results": []}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            class MockSession:
                def get(self, url, params=None):
                    return MockResponse()

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            mock_session_class.return_value = MockSession()

            result = await registry.execute("get_weather_forecast", {"cities": ["FakeCity123"]})

            assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_weather_api_error(self):
        """Test weather tool handles API errors gracefully."""
        registry = ToolRegistry(enable_web_search=False)

        with patch("src.tools.aiohttp.ClientSession") as mock_session_class:
            class MockResponse:
                def __init__(self):
                    self.status = 500

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            class MockSession:
                def get(self, url, params=None):
                    return MockResponse()

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            mock_session_class.return_value = MockSession()

            result = await registry.execute("get_weather_forecast", {"cities": ["Houston"]})

            assert "Could not geocode" in result or "Could not fetch" in result
