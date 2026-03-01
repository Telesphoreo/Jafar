"""
Unit tests for src/news.py

Tests news fetching and formatting with mocked DuckDuckGo API.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.news import NewsArticle, fetch_economic_news, format_news_for_llm, parse_article_age_hours


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
                "date": datetime.now().isoformat(),
            },
            {
                "title": "Oil Prices Rise on OPEC Cuts",
                "body": "Crude oil jumped 3% after OPEC announced production cuts.",
                "url": "https://example.com/oil-opec",
                "source": "Bloomberg",
                "date": datetime.now().isoformat(),
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
                "date": datetime.now().isoformat(),
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
                "date": datetime.now().isoformat(),
            },
        ]
        results_q2 = [
            {
                "title": "Oil News",
                "body": "Oil snippet",
                "url": "https://example.com/oil",
                "source": "Bloomberg",
                "date": datetime.now().isoformat(),
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
                date="2024-01-15",  # Static date OK for format tests
            ),
            NewsArticle(
                title="Oil Prices Rise",
                body="Crude jumped 3%.",
                url="https://example.com/oil",
                source="Bloomberg",
                date="2024-01-15",  # Static date OK for format tests
            ),
        ]

        result = format_news_for_llm(articles)

        assert "Today's Economic News Headlines" in result
        assert "2 fresh articles" in result
        assert "**Fed Holds Rates Steady**" in result
        assert "**Oil Prices Rise**" in result
        assert "Reuters" in result
        assert "Bloomberg" in result
        assert "Federal Reserve kept rates unchanged" in result

    def test_format_news_for_llm_empty(self):
        """Test formatting with no articles returns empty string."""
        result = format_news_for_llm([])
        assert result == ""

    def test_format_news_for_llm_fresh_header(self):
        """Test that formatted output uses fresh articles header."""
        articles = [
            NewsArticle(
                title="Test",
                body="Body",
                url="https://example.com/test",
                source="AP",
                date="2024-01-15",  # Static date OK for format tests
            ),
        ]
        result = format_news_for_llm(articles)
        assert "fresh articles" in result

    def test_format_news_for_llm_missing_body(self):
        """Test formatting with articles that have no body."""
        articles = [
            NewsArticle(
                title="Breaking News",
                body="",
                url="https://example.com/breaking",
                source="AP",
                date="2024-01-15",  # Static date OK for format tests
            ),
        ]

        result = format_news_for_llm(articles)

        assert "**Breaking News**" in result
        assert "AP" in result


class TestParseArticleAgeHours:
    """Tests for the parse_article_age_hours helper."""

    def test_iso_date_recent(self):
        """Test parsing a recent ISO date."""
        recent = (datetime.now() - timedelta(hours=1)).isoformat()
        age = parse_article_age_hours(recent)
        assert age is not None
        assert 0.5 < age < 1.5

    def test_iso_date_old(self):
        """Test parsing an old ISO date."""
        old = (datetime.now() - timedelta(hours=72)).isoformat()
        age = parse_article_age_hours(old)
        assert age is not None
        assert age > 70

    def test_empty_string(self):
        """Test that empty string returns None."""
        assert parse_article_age_hours("") is None

    def test_garbage_string(self):
        """Test that unparseable string returns None."""
        assert parse_article_age_hours("not a date at all xyz") is None

    def test_standard_date_format(self):
        """Test parsing a standard date format."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        age = parse_article_age_hours(yesterday)
        assert age is not None
        assert age > 20  # At least 20 hours old


class TestFetchEconomicNewsFiltering:
    """Tests for freshness and cross-run dedup in fetch_economic_news."""

    @pytest.mark.asyncio
    async def test_excludes_previously_reported_urls(self):
        """Test that exclude_urls filters out already-reported articles."""
        fresh_date = (datetime.now() - timedelta(hours=1)).isoformat()
        mock_results = [
            {"title": "Old Story", "body": "...", "url": "https://example.com/old", "source": "R", "date": fresh_date},
            {"title": "New Story", "body": "...", "url": "https://example.com/new", "source": "R", "date": fresh_date},
        ]
        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.return_value = mock_results
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["test"],
                exclude_urls={"https://example.com/old"},
            )
        assert len(articles) == 1
        assert articles[0].title == "New Story"

    @pytest.mark.asyncio
    async def test_filters_stale_articles(self):
        """Test that articles older than max_age_hours are filtered."""
        old_date = (datetime.now() - timedelta(hours=72)).isoformat()
        fresh_date = (datetime.now() - timedelta(hours=1)).isoformat()
        mock_results = [
            {"title": "Stale", "body": "...", "url": "https://example.com/stale", "source": "R", "date": old_date},
            {"title": "Fresh", "body": "...", "url": "https://example.com/fresh", "source": "R", "date": fresh_date},
        ]
        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.return_value = mock_results
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["test"],
                max_age_hours=48,
            )
        assert len(articles) == 1
        assert articles[0].title == "Fresh"

    @pytest.mark.asyncio
    async def test_unparseable_dates_pass_through(self):
        """Test that articles with unparseable dates are not filtered."""
        mock_results = [
            {"title": "No Date", "body": "...", "url": "https://example.com/1", "source": "R", "date": ""},
            {"title": "Bad Date", "body": "...", "url": "https://example.com/2", "source": "R", "date": "not-a-date-xyz"},
        ]
        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs.news.return_value = mock_results
            mock_ddgs_class.return_value = mock_ddgs

            articles = await fetch_economic_news(
                queries=["test"],
                max_age_hours=48,
            )
        # Both should pass through since dates are unparseable
        assert len(articles) == 2
