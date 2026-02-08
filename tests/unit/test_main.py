"""
Unit tests for src/main.py

Tests the agentic analysis loop, submit_report handling, fallback parsing,
and the LLM trend filter with tool calling.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.main import analyze_with_llm, llm_filter_trends, _get_filter_tool_definition
from src.analyzer import DiscoveredTrend


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

    @pytest.mark.asyncio
    async def test_submit_report_handles_null_values(self, mock_llm, mock_trend_tweets):
        """Test that submit_report handles null/None values without crashing."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        # Simulate LLM sending null for optional fields
        mock_tool_call.function.arguments = '''{
            "subject_line": null,
            "signal_strength": null,
            "assessment": "Some assessment",
            "trends_observed": "- Trends",
            "fact_check": null,
            "actionability": null,
            "actionability_reason": null,
            "historical_parallel": null,
            "bottom_line": "Bottom line"
        }'''
        mock_tool_call.id = "call_null"

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

            # Should not raise an exception
            body, signal, is_notable, tokens, subject = await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
            )

        # Should use defaults for null values
        assert subject == "Jafar Market Digest"
        assert signal == "low"
        assert is_notable is False
        # Sections with null values should be omitted
        assert "**Fact Check:**" not in body
        assert "**Actionability:**" not in body
        assert "**Historical Parallel:**" not in body
        # Non-null sections should appear
        assert "**Assessment:**" in body
        assert "**Trends Observed:**" in body
        assert "**Bottom Line:**" in body


    @pytest.mark.asyncio
    async def test_submit_report_news_roundup_appears_first(self, mock_llm, mock_trend_tweets):
        """Test that news_roundup appears first in body_parts."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "Fed Holds Steady, Twitter Yawns",
            "signal_strength": "low",
            "news_roundup": "• Fed holds rates at 5.25% - expected, markets shrug\\n• NVIDIA earnings beat by 15%",
            "assessment": "Another quiet Twitter day.",
            "trends_observed": "• Generic market chatter",
            "actionability": "not actionable",
            "actionability_reason": "No unusual signals.",
            "bottom_line": "News was the story today, not Twitter."
        }'''
        mock_tool_call.id = "call_news"

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

        # News roundup should appear in the body
        assert "**News Roundup:**" in body
        # News roundup should come BEFORE assessment
        news_pos = body.index("**News Roundup:**")
        assessment_pos = body.index("**Assessment:**")
        assert news_pos < assessment_pos

    @pytest.mark.asyncio
    async def test_news_context_included_in_prompt(self, mock_llm, mock_trend_tweets):
        """Test that news_context is passed through to the LLM prompt."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "Test",
            "signal_strength": "low",
            "assessment": "Test",
            "trends_observed": "• Test",
            "actionability": "not actionable",
            "actionability_reason": "Test.",
            "bottom_line": "Test."
        }'''
        mock_tool_call.id = "call_ctx"

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

            await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
                news_context="## Today's Economic News Headlines\n1. Fed holds rates",
            )

        # Verify the news context was included in the prompt sent to LLM
        call_args = mock_llm.generate.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = messages[0]["content"] if messages else ""
        assert "Economic News Headlines" in user_msg

    @pytest.mark.asyncio
    async def test_twitter_mentions_included_in_prompt(self, mock_llm, mock_trend_tweets):
        """Test that twitter_mentions are included in the LLM prompt."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_report"
        mock_tool_call.function.arguments = '''{
            "subject_line": "Test",
            "signal_strength": "low",
            "assessment": "Test",
            "trends_observed": "• Test",
            "actionability": "not actionable",
            "actionability_reason": "Test.",
            "bottom_line": "Test."
        }'''
        mock_tool_call.id = "call_mentions"

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

            await analyze_with_llm(
                llm=mock_llm,
                trend_tweets=mock_trend_tweets,
                twitter_mentions=["Bitcoin", "Elon Musk", "Music"],
            )

        # Verify twitter mentions were included in the prompt
        call_args = mock_llm.generate.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = messages[0]["content"] if messages else ""
        assert "Twitter Mentions" in user_msg
        assert "Bitcoin" in user_msg


class TestSanitizeLLMOutput:
    """Tests for the sanitize_llm_output function."""

    def test_replaces_literal_backslash_n_with_newline(self):
        """Test that literal \\n strings become actual newlines."""
        from src.main import sanitize_llm_output

        text = "• First bullet\\n• Second bullet\\n• Third bullet"
        result = sanitize_llm_output(text)
        assert result == "• First bullet\n• Second bullet\n• Third bullet"

    def test_replaces_box_drawing_characters_with_spaces(self):
        """Test that box-drawing horizontal lines become spaces."""
        from src.main import sanitize_llm_output

        text = "•─Kevin─Warsh─nominated─for─Fed─Chair"
        result = sanitize_llm_output(text)
        assert "─" not in result
        assert "Kevin Warsh nominated for Fed Chair" in result

    def test_fixes_bullet_points_not_on_own_line(self):
        """Test that bullet points get their own line."""
        from src.main import sanitize_llm_output

        text = "Some text• Bullet one• Bullet two"
        result = sanitize_llm_output(text)
        assert "text\n•" in result
        assert "one\n•" in result

    def test_handles_empty_string(self):
        """Test that empty strings are handled gracefully."""
        from src.main import sanitize_llm_output

        assert sanitize_llm_output("") == ""
        assert sanitize_llm_output(None) is None

    def test_handles_already_clean_text(self):
        """Test that clean text passes through unchanged."""
        from src.main import sanitize_llm_output

        text = "• First bullet\n• Second bullet\n• Third bullet"
        result = sanitize_llm_output(text)
        assert result == text

    def test_cleans_multiple_spaces(self):
        """Test that multiple consecutive spaces are collapsed."""
        from src.main import sanitize_llm_output

        text = "Too    many   spaces   here"
        result = sanitize_llm_output(text)
        assert "  " not in result
        assert result == "Too many spaces here"

    def test_realistic_malformed_output(self):
        """Test with realistic malformed LLM output."""
        from src.main import sanitize_llm_output

        malformed = "• Verified: Kevin Warsh nominated for Fed Chair\\n• Exaggerated: Silver dropping 20%\\n• False: Dollar crashing"
        result = sanitize_llm_output(malformed)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "• Verified: Kevin Warsh nominated for Fed Chair"
        assert lines[1] == "• Exaggerated: Silver dropping 20%"
        assert lines[2] == "• False: Dollar crashing"


class TestLLMFilterTrends:
    """Tests for the llm_filter_trends function with tool calling."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.provider_name = "TestProvider"
        llm.model_name = "test-model"
        llm.generate = AsyncMock()
        return llm

    @pytest.fixture
    def sample_trend_candidates(self):
        """Create sample DiscoveredTrend objects with sample tweets."""
        return [
            DiscoveredTrend(
                term="Release",
                term_type="ngram",
                mention_count=15,
                unique_authors=12,
                total_engagement=5000,
                avg_engagement=333,
                sample_tweets=[
                    "The Epstein files release is finally happening today",
                    "Full document release expected within hours",
                    "This release will expose everything",
                ],
                financial_context_count=10,
            ),
            DiscoveredTrend(
                term="$NVDA",
                term_type="cashtag",
                mention_count=25,
                unique_authors=20,
                total_engagement=8000,
                avg_engagement=320,
                sample_tweets=[
                    "$NVDA earnings beat expectations",
                    "NVIDIA guidance raised for next quarter",
                ],
                financial_context_count=25,
            ),
            DiscoveredTrend(
                term="Music",
                term_type="ngram",
                mention_count=10,
                unique_authors=8,
                total_engagement=2000,
                avg_engagement=200,
                sample_tweets=[
                    "New Taylor Swift album dropped",
                    "Best music of 2026 so far",
                ],
                financial_context_count=2,
            ),
        ]

    def test_get_filter_tool_definition(self):
        """Test that filter tool definition has correct structure."""
        tool_def = _get_filter_tool_definition()

        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "submit_filter_result"

        params = tool_def["function"]["parameters"]
        assert "trends_to_keep" in params["properties"]
        assert "reasoning" in params["properties"]
        assert params["properties"]["trends_to_keep"]["type"] == "array"
        assert "trends_to_keep" in params["required"]
        assert "reasoning" in params["required"]

    @pytest.mark.asyncio
    async def test_filter_keeps_relevant_trends(self, mock_llm, sample_trend_candidates):
        """Test that filter keeps relevant trends via tool call."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_filter_result"
        mock_tool_call.function.arguments = '''{
            "trends_to_keep": ["Release", "$NVDA"],
            "reasoning": "Kept Release (Epstein files) and $NVDA (earnings). Rejected Music (entertainment)."
        }'''

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 50

        mock_llm.generate.return_value = mock_response

        result = await llm_filter_trends(mock_llm, sample_trend_candidates)

        assert result == ["Release", "$NVDA"]
        assert "Music" not in result

    @pytest.mark.asyncio
    async def test_filter_rejects_all_trends(self, mock_llm, sample_trend_candidates):
        """Test that filter can reject all trends (quiet day)."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_filter_result"
        mock_tool_call.function.arguments = '''{
            "trends_to_keep": [],
            "reasoning": "No significant market-moving trends identified."
        }'''

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 30

        mock_llm.generate.return_value = mock_response

        result = await llm_filter_trends(mock_llm, sample_trend_candidates)

        assert result == []

    @pytest.mark.asyncio
    async def test_filter_fallback_on_no_tool_call(self, mock_llm, sample_trend_candidates):
        """Test fallback returns all candidates when LLM doesn't use tool."""
        mock_response = MagicMock()
        mock_response.content = "I think Release and NVDA are interesting."
        mock_response.tool_calls = None
        mock_response.token_count = 20

        mock_llm.generate.return_value = mock_response

        result = await llm_filter_trends(mock_llm, sample_trend_candidates)

        # Should fall back to returning all candidate terms
        assert len(result) == 3
        assert "Release" in result
        assert "$NVDA" in result
        assert "Music" in result

    @pytest.mark.asyncio
    async def test_filter_fallback_on_exception(self, mock_llm, sample_trend_candidates):
        """Test fallback returns all candidates when exception occurs."""
        mock_llm.generate.side_effect = Exception("API Error")

        result = await llm_filter_trends(mock_llm, sample_trend_candidates)

        # Should fall back to returning all candidate terms
        assert len(result) == 3
        assert "Release" in result
        assert "$NVDA" in result
        assert "Music" in result

    @pytest.mark.asyncio
    async def test_filter_empty_candidates(self, mock_llm):
        """Test filter handles empty candidate list."""
        result = await llm_filter_trends(mock_llm, [])

        assert result == []
        # LLM should not be called for empty list
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_filter_passes_tools_to_llm(self, mock_llm, sample_trend_candidates):
        """Test that filter passes correct tool definition to LLM."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_filter_result"
        mock_tool_call.function.arguments = '{"trends_to_keep": ["Release"], "reasoning": "Test"}'

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 20

        mock_llm.generate.return_value = mock_response

        await llm_filter_trends(mock_llm, sample_trend_candidates)

        # Verify LLM was called with tools
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "submit_filter_result"

    @pytest.mark.asyncio
    async def test_filter_includes_sample_tweets_in_prompt(self, mock_llm, sample_trend_candidates):
        """Test that sample tweets are included in the prompt for context."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_filter_result"
        mock_tool_call.function.arguments = '{"trends_to_keep": ["Release"], "reasoning": "Test"}'

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 20

        mock_llm.generate.return_value = mock_response

        await llm_filter_trends(mock_llm, sample_trend_candidates)

        # Verify prompt contains sample tweets
        call_kwargs = mock_llm.generate.call_args.kwargs
        prompt = call_kwargs.get("prompt", "")
        assert "Epstein files release" in prompt
        assert "NVDA earnings" in prompt
        assert "Taylor Swift" in prompt

    @pytest.mark.asyncio
    async def test_filter_candidates_without_samples(self, mock_llm):
        """Test filter handles candidates without sample tweets."""
        candidates = [
            DiscoveredTrend(
                term="Silver",
                term_type="ngram",
                mention_count=10,
                unique_authors=8,
                total_engagement=3000,
                avg_engagement=300,
                sample_tweets=[],  # No samples
                financial_context_count=8,
            ),
        ]

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_filter_result"
        mock_tool_call.function.arguments = '{"trends_to_keep": ["Silver"], "reasoning": "Test"}'

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]
        mock_response.token_count = 20

        mock_llm.generate.return_value = mock_response

        result = await llm_filter_trends(mock_llm, candidates)

        assert result == ["Silver"]
