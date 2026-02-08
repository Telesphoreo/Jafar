"""
Unit tests for src/news.py

Tests news fetching and formatting with mocked DuckDuckGo API.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.news import NewsArticle, fetch_economic_news, format_news_for_llm


class TestFetchEconomicNews:
    """Tests for the fetch_economic_news function."""

    @pytest.mark.asyncio
    async def test_fetch_economic_news_success(self):
        """Test successful news fetching with mock DDGS."""
        mock_results = [
            {
                "title": "Fed Holds Rates Steady",
                "body": "The Federal Reserve kept interest rates unchanged.",
                "url": "https://example.com/fed-rates",
                "source": "Reuters",
                "date": "2024-01-15",
            },
            {
                "title": "Oil Prices Rise on OPEC Cuts",
                "body": "Crude oil jumped 3% after OPEC announced production cuts.",
                "url": "https://example.com/oil-opec",
                "source": "Bloomberg",
                "date": "2024-01-15",
            },
        ]

        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.return_value = mock_results
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["economy news"],
                max_results_per_query=5,
            )

        assert len(articles) == 2
        assert isinstance(articles[0], NewsArticle)
        assert articles[0].title == "Fed Holds Rates Steady"
        assert articles[0].source == "Reuters"
        assert articles[1].url == "https://example.com/oil-opec"

    @pytest.mark.asyncio
    async def test_fetch_economic_news_deduplication(self):
        """Test that duplicate URLs are removed across queries."""
        mock_results = [
            {
                "title": "Fed Holds Rates",
                "body": "Snippet 1",
                "url": "https://example.com/same-article",
                "source": "Reuters",
                "date": "2024-01-15",
            },
        ]

        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.return_value = mock_results
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["economy news", "fed news"],  # Two queries returning same URL
                max_results_per_query=5,
            )

        # Should deduplicate - same URL from both queries
        assert len(articles) == 1

    @pytest.mark.asyncio
    async def test_fetch_economic_news_failure(self):
        """Test graceful error handling when DDGS fails."""
        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.side_effect = Exception("Network error")
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["economy news"],
                max_results_per_query=5,
            )

        # Should return empty list, not raise
        assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_economic_news_empty(self):
        """Test handling of empty results."""
        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.return_value = []
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["economy news"],
                max_results_per_query=5,
            )

        assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_economic_news_multiple_queries(self):
        """Test fetching from multiple queries with different results."""
        results_q1 = [
            {
                "title": "Fed News",
                "body": "Fed snippet",
                "url": "https://example.com/fed",
                "source": "Reuters",
                "date": "2024-01-15",
            },
        ]
        results_q2 = [
            {
                "title": "Oil News",
                "body": "Oil snippet",
                "url": "https://example.com/oil",
                "source": "Bloomberg",
                "date": "2024-01-15",
            },
        ]

        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.side_effect = [results_q1, results_q2]
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["fed news", "oil news"],
                max_results_per_query=5,
            )

        assert len(articles) == 2
        assert articles[0].title == "Fed News"
        assert articles[1].title == "Oil News"


class TestFormatNewsForLLM:
    """Tests for the format_news_for_llm function."""

    def test_format_news_for_llm(self):
        """Test formatting articles for LLM consumption."""
        articles = [
            NewsArticle(
                title="Fed Holds Rates Steady",
                body="The Federal Reserve kept rates unchanged.",
                url="https://example.com/fed",
                source="Reuters",
                date="2024-01-15",
            ),
            NewsArticle(
                title="Oil Prices Rise",
                body="Crude jumped 3%.",
                url="https://example.com/oil",
                source="Bloomberg",
                date="2024-01-15",
            ),
        ]

        result = format_news_for_llm(articles)

        assert "Today's Economic News Headlines" in result
        assert "2 articles" in result
        assert "**Fed Holds Rates Steady**" in result
        assert "**Oil Prices Rise**" in result
        assert "Reuters" in result
        assert "Bloomberg" in result
        assert "Federal Reserve kept rates unchanged" in result

    def test_format_news_for_llm_empty(self):
        """Test formatting with no articles returns empty string."""
        result = format_news_for_llm([])
        assert result == ""

    def test_format_news_for_llm_missing_body(self):
        """Test formatting with articles that have no body."""
        articles = [
            NewsArticle(
                title="Breaking News",
                body="",
                url="https://example.com/breaking",
                source="AP",
                date="2024-01-15",
            ),
        ]

        result = format_news_for_llm(articles)

        assert "**Breaking News**" in result
        assert "AP" in result
