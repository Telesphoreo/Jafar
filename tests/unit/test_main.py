"""
Unit tests for src/main.py

Tests the agentic analysis loop, particularly submit_report handling and fallback parsing.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.main import analyze_with_llm


class TestAnalyzeWithLLM:
    """Tests for the analyze_with_llm function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.provider_name = "TestProvider"
        llm.model_name = "test-model"
        llm.generate = AsyncMock()
        return llm

    @pytest.fixture
    def mock_trend_tweets(self):
        """Create mock trend tweets data."""
        mock_tweet = MagicMock()
        mock_tweet.text = "Silver is going crazy!"
        mock_tweet.engagement_score = 100
        mock_tweet.username = "testuser"
        mock_tweet.created_at = None
        return {"silver": [mock_tweet]}

    @pytest.mark.asyncio
    async def test_submit_report_extracts_structured_sections(self, mock_llm, mock_trend_tweets):
        """Test that submit_report extracts and formats structured sections correctly."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "Silver Surges on Supply Fears",
            "signal_strength": "medium",
            "assessment": "Silver is having a moment. The chatter is real.",
            "trends_observed": "- Silver price discussions up 300%\\n- Supply concerns mentioned",
            "fact_check": "Verified: Silver actually up 2.3% today.",
            "actionability": "monitor only",
            "actionability_reason": "Not enough volume yet to act.",
            "historical_parallel": "History rhymes: Similar to 2011 silver squeeze.",
            "bottom_line": "Worth watching but not worth acting on yet."
        }'''
        mock_tool_call.id = "call_123"

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 100
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert subject == "Silver Surges on Supply Fears"
        assert signal == "medium"
        assert is_notable is False
        # Check Title Case headers
        assert "**Assessment:**" in body
        assert "**Trends Observed:**" in body
        assert "**Fact Check:**" in body
        assert "**Actionability:** Monitor Only" in body
        assert "**Historical Parallel:**" in body
        assert "**Bottom Line:**" in body
        # Check content
        assert "Silver is having a moment" in body
        assert "Similar to 2011" in body

    @pytest.mark.asyncio
    async def test_submit_report_omits_empty_sections(self, mock_llm, mock_trend_tweets):
        """Test that empty sections are omitted from the body."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "Nothing Burger Today",
            "signal_strength": "none",
            "assessment": "Another quiet day. Nothing to see here.",
            "trends_observed": "- Generic market chatter\\n- No specific trends",
            "fact_check": "",
            "actionability": "not actionable",
            "actionability_reason": "No claims worth verifying.",
            "historical_parallel": "",
            "bottom_line": "Save your attention for another day."
        }'''
        mock_tool_call.id = "call_456"

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 50
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        # Empty sections should NOT appear
        assert "**Fact Check:**" not in body
        assert "**Historical Parallel:**" not in body
        # Non-empty sections should appear
        assert "**Assessment:**" in body
        assert "**Trends Observed:**" in body
        assert "**Actionability:**" in body
        assert "**Bottom Line:**" in body

    @pytest.mark.asyncio
    async def test_submit_report_high_signal_sets_notable(self, mock_llm, mock_trend_tweets):
        """Test that high signal strength sets is_notable to True."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "NVDA Shortage Getting Real",
            "signal_strength": "high",
            "assessment": "This is significant.",
            "trends_observed": "- NVDA shortage confirmed",
            "actionability": "warrants attention",
            "actionability_reason": "Multiple sources confirming.",
            "bottom_line": "Pay attention to this one."
        }'''
        mock_tool_call.id = "call_789"

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 50
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert signal == "high"
        assert is_notable is True

    @pytest.mark.asyncio
    async def test_submit_report_normalizes_signal_strength_case(self, mock_llm, mock_trend_tweets):
        """Test that signal_strength is normalized to lowercase."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "Test",
            "signal_strength": "MEDIUM",
            "assessment": "Content",
            "trends_observed": "- Test",
            "actionability": "monitor only",
            "actionability_reason": "Testing.",
            "bottom_line": "Test."
        }'''
        mock_tool_call.id = "call_case"

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 25
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert signal == "medium"  # Should be lowercase

    @pytest.mark.asyncio
    async def test_actionability_formatted_as_title_case(self, mock_llm, mock_trend_tweets):
        """Test that actionability value is formatted as Title Case."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "Test",
            "signal_strength": "low",
            "assessment": "Content",
            "trends_observed": "- Test",
            "actionability": "worth researching",
            "actionability_reason": "Interesting but unverified.",
            "bottom_line": "Test."
        }'''
        mock_tool_call.id = "call_title"

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 25
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert "**Actionability:** Worth Researching" in body

    @pytest.mark.asyncio
    async def test_fallback_parsing_when_no_tool_call(self, mock_llm, mock_trend_tweets):
        """Test fallback parsing when LLM returns text instead of calling submit_report."""
        mock_response = MagicMock()
        mock_response.content = """**SUBJECT LINE**: Nothing Burger Today

**SIGNAL STRENGTH**: LOW

**ASSESSMENT**:
Another quiet day in the markets.

**BOTTOM LINE**:
Save your attention for another day."""
        mock_response.tool_calls = None
        mock_response.token_count = 75
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert subject == "Nothing Burger Today"
        assert signal == "low"
        assert is_notable is False
        # Body should NOT contain the metadata lines
        assert "**SUBJECT LINE**" not in body
        assert "**SIGNAL STRENGTH**" not in body
        assert "ASSESSMENT" in body

    @pytest.mark.asyncio
    async def test_fallback_strips_metadata_from_body(self, mock_llm, mock_trend_tweets):
        """Test that fallback parsing strips SUBJECT LINE and SIGNAL STRENGTH from body."""
        mock_response = MagicMock()
        mock_response.content = """**SUBJECT LINE**: Test Subject

**SIGNAL STRENGTH**: MEDIUM

The actual body content starts here.

**ASSESSMENT**:
Some analysis."""
        mock_response.tool_calls = None
        mock_response.token_count = 60
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        # Verify metadata was stripped
        assert "**SUBJECT LINE**" not in body
        assert "**SIGNAL STRENGTH**" not in body
        # Actual content should remain
        assert "actual body content" in body
        assert "ASSESSMENT" in body

    @pytest.mark.asyncio
    async def test_fallback_detects_high_signal(self, mock_llm, mock_trend_tweets):
        """Test that fallback correctly detects HIGH signal strength."""
        mock_response = MagicMock()
        mock_response.content = """**SUBJECT LINE**: Big News

**SIGNAL STRENGTH**: HIGH

This is important."""
        mock_response.tool_calls = None
        mock_response.token_count = 40
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert signal == "high"
        assert is_notable is True

    @pytest.mark.asyncio
    async def test_fallback_detects_none_signal(self, mock_llm, mock_trend_tweets):
        """Test that fallback correctly detects NONE signal strength."""
        mock_response = MagicMock()
        mock_response.content = """**SUBJECT LINE**: Absolutely Nothing

**SIGNAL STRENGTH**: NONE

Nothing to see here."""
        mock_response.tool_calls = None
        mock_response.token_count = 30
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert signal == "none"
        assert is_notable is False

    @pytest.mark.asyncio
    async def test_fallback_handles_title_case(self, mock_llm, mock_trend_tweets):
        """Test that fallback parsing handles Title Case headers (not just UPPERCASE)."""
        mock_response = MagicMock()
        mock_response.content = """**Subject Line**: Title Case Subject

**Signal Strength**: Medium

**Assessment:**
This uses Title Case headers.

**Bottom Line:**
Should still parse correctly."""
        mock_response.tool_calls = None
        mock_response.token_count = 45
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        assert subject == "Title Case Subject"
        assert signal == "medium"
        # Metadata should be stripped even with Title Case
        assert "**Subject Line**" not in body
        assert "**Signal Strength**" not in body
        assert "Assessment" in body

    @pytest.mark.asyncio
    async def test_other_tools_executed_before_submit(self, mock_llm, mock_trend_tweets):
        """Test that other tools are executed and their output fed back before submit_report."""
        # First response: LLM calls get_market_data
        mock_market_tool_call = MagicMock()
        mock_market_tool_call.function.name = "get_market_data"
        mock_market_tool_call.function.arguments = '{"symbols": ["NVDA"]}'
        mock_market_tool_call.id = "call_market"

        first_response = MagicMock()
        first_response.content = "Let me check the market data."
        first_response.tool_calls = [mock_market_tool_call]
        first_response.token_count = 50
        first_response.raw_content = None

        # Second response: LLM calls submit_report after seeing tool output
        mock_submit_call = MagicMock()
        mock_submit_call.function.name = "submit_report"
        mock_submit_call.function.arguments = '''{
            "subject_line": "NVDA Verified at $500",
            "signal_strength": "low",
            "assessment": "Checked the data, nothing unusual.",
            "trends_observed": "- NVDA mentioned",
            "fact_check": "NVDA price verified at $500.",
            "actionability": "not actionable",
            "actionability_reason": "Normal trading day.",
            "bottom_line": "Move along, nothing to see here."
        }'''
        mock_submit_call.id = "call_submit"

        second_response = MagicMock()
        second_response.content = ""
        second_response.tool_calls = [mock_submit_call]
        second_response.token_count = 60

        mock_llm.generate.side_effect = [first_response, second_response]

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            mock_registry.execute = AsyncMock(return_value="NVDA: $500.00 (+0.5%)")
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

            # Verify the market data tool was executed
            mock_registry.execute.assert_called_once_with("get_market_data", {"symbols": ["NVDA"]})

        # Verify final output came from submit_report
        assert subject == "NVDA Verified at $500"
        assert "nothing unusual" in body
        assert tokens == 110  # 50 + 60

    @pytest.mark.asyncio
    async def test_submit_report_defaults_on_missing_fields(self, mock_llm, mock_trend_tweets):
        """Test that submit_report provides defaults for missing fields."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '{}'  # Empty - all fields missing
        mock_tool_call.id = "call_empty"

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 20
        mock_response.raw_content = None

        mock_llm.generate.return_value = mock_response

        with patch('src.main.ToolRegistry') as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry

            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        # Should have defaults
        assert subject == "Jafar Market Digest"
        assert signal == "low"
        assert is_notable is False
        assert body == "No analysis provided."
