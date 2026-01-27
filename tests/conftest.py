"""
Shared pytest fixtures for Jafar tests.

This module provides:
- Session-scoped spaCy model (loaded once)
- In-memory SQLite databases
- Temporary checkpoint paths
- Mock external services (yfinance, LLM providers, twscrape)
- Sample data fixtures
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scraper import ScrapedTweet
from tests.fixtures import make_financial_tweets, make_sample_tweet, make_sample_tweets


# =============================================================================
# Core Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def spacy_nlp():
    """Session-scoped spaCy model - load once for all tests."""
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model 'en_core_web_sm' not installed")


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory that cleans up after test."""
    return tmp_path


@pytest.fixture
def temp_checkpoint_file(tmp_path) -> Path:
    """Provide a temporary checkpoint file path."""
    return tmp_path / "test_checkpoint.json"


@pytest.fixture
def temp_db_path(tmp_path) -> str:
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_history.db")


@pytest.fixture
def in_memory_db():
    """Provide an in-memory SQLite connection for fast tests."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_tweet() -> ScrapedTweet:
    """Provide a single sample tweet."""
    return make_sample_tweet()


@pytest.fixture
def sample_tweets() -> list[ScrapedTweet]:
    """Provide a list of sample tweets."""
    return make_sample_tweets(count=10)


@pytest.fixture
def financial_tweets() -> list[ScrapedTweet]:
    """Provide tweets with strong financial context."""
    return make_financial_tweets(count=5)


@pytest.fixture
def sample_trends() -> list[str]:
    """Provide a list of sample trend terms."""
    return ["$NVDA", "$AAPL", "Silver", "#inflation", "Fed"]


@pytest.fixture
def sample_config_yaml(tmp_path) -> Path:
    """Create a sample config.yaml file."""
    config_path = tmp_path / "config.yaml"
    config_content = """
llm:
  provider: openai
  openai_model: gpt-4o
  google_model: gemini-2.0-flash

scraping:
  broad_tweet_limit: 100
  specific_tweet_limit: 50
  top_trends_count: 10
  min_trend_mentions: 3
  min_trend_authors: 2

smtp:
  host: smtp.example.com
  port: 587
  use_tls: true

email:
  from: test@example.com
  from_name: Test System
  to:
    - recipient@example.com

memory:
  enabled: true
  store_type: chroma
  embedding_provider: openai

fact_checker:
  enabled: true
  cache_ttl_minutes: 5

temporal:
  consecutive_threshold: 3
  gap_threshold_days: 14

logging:
  level: DEBUG
"""
    config_path.write_text(config_content)
    return config_path


# =============================================================================
# Mock External Services
# =============================================================================


@pytest.fixture
def mock_yfinance():
    """Mock yfinance.Ticker to return predictable market data."""
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()

        # Mock history() to return a DataFrame-like object
        mock_history = MagicMock()
        mock_history.empty = False

        # Create mock rows with proper indexing
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "Close": 150.0,
            "Volume": 1000000,
        }.get(key, 0)

        # Mock iloc to return rows
        mock_history.iloc.__getitem__ = lambda idx: mock_row
        mock_history.__len__ = lambda self: 10

        mock_ticker.history.return_value = mock_history

        # Mock fast_info
        mock_fast_info = MagicMock()
        mock_fast_info.get = lambda key, default=None: {
            "averageVolume": 500000,
            "fiftyTwoWeekHigh": 160.0,
            "fiftyTwoWeekLow": 100.0,
            "marketCap": 1000000000,
        }.get(key, default)
        mock_ticker.fast_info = mock_fast_info

        mock_ticker_class.return_value = mock_ticker
        yield mock_ticker_class


@pytest.fixture
def mock_google_genai():
    """Mock google.genai.Client for Gemini API tests."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()

        # Create async mock for aio.models.generate_content
        mock_response = MagicMock()
        mock_response.text = "This is a test response from Gemini."
        mock_response.function_calls = None
        mock_response.candidates = [MagicMock(content=MagicMock())]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        )

        mock_aio = MagicMock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        mock_client_class.return_value = mock_client
        yield mock_client_class


@pytest.fixture
def mock_openai():
    """Mock AsyncOpenAI for OpenAI API tests."""
    # Patch at the module where it's imported, not where it's defined
    with patch("src.llm.openai_client.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()

        # Create mock response
        mock_message = MagicMock()
        mock_message.content = "This is a test response from OpenAI."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        yield mock_client_class


@pytest.fixture
def mock_twscrape_api():
    """Mock twscrape.API for Twitter scraping tests."""
    with patch("twscrape.API") as mock_api_class:
        mock_api = MagicMock()

        # Mock pool stats
        mock_api.pool.stats = AsyncMock(
            return_value={"active": 3, "total": 5, "locked": 2}
        )

        # Mock search to return async generator
        async def mock_search(*args, **kwargs):
            # Return some mock Tweet objects
            for i in range(5):
                mock_tweet = MagicMock()
                mock_tweet.id = 1234567890 + i
                mock_tweet.rawContent = f"Mock tweet #{i}"
                mock_tweet.user = MagicMock(username=f"user{i}", displayname=f"User {i}")
                mock_tweet.date = datetime.now()
                mock_tweet.likeCount = 100
                mock_tweet.retweetCount = 50
                mock_tweet.replyCount = 10
                mock_tweet.viewCount = 1000
                mock_tweet.lang = "en"
                mock_tweet.hashtags = ["test"]
                yield mock_tweet

        mock_api.search = mock_search
        mock_api_class.return_value = mock_api
        yield mock_api_class


@pytest.fixture
def mock_ddgs():
    """Mock DuckDuckGo search for web search tests."""
    with patch("ddgs.DDGS") as mock_ddgs_class:
        mock_ddgs = MagicMock()

        # Mock text search results
        mock_ddgs.text.return_value = [
            {
                "title": "Test Result 1",
                "body": "This is a test search result.",
                "href": "https://example.com/1",
            },
            {
                "title": "Test Result 2",
                "body": "Another test search result.",
                "href": "https://example.com/2",
            },
        ]

        mock_ddgs_class.return_value = mock_ddgs
        yield mock_ddgs_class


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("SMTP_USERNAME", "test@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "test-password")
    monkeypatch.setenv("TWITTER_USERNAME", "testuser")
    monkeypatch.setenv("TWITTER_PASSWORD", "testpass")


# =============================================================================
# History Database Fixtures
# =============================================================================


@pytest.fixture
def sample_history_db(temp_db_path):
    """Create a DigestHistory instance with sample data."""
    from src.history import DigestHistory

    history = DigestHistory(db_path=temp_db_path)

    # Add some sample historical data
    history.store_digest(
        trends=["$NVDA", "Silver", "Oil"],
        tweet_count=500,
        digest_text="Sample digest from yesterday.",
        signal_strength="medium",
        top_engagement=10000.0,
        notable=False,
        trend_details={
            "nvda": {"mentions": 50, "engagement": 5000},
            "silver": {"mentions": 30, "engagement": 3000},
        },
    )

    return history


# =============================================================================
# Diagnostic Fixtures
# =============================================================================


@pytest.fixture
def sample_diagnostics():
    """Create a sample RunDiagnostics object."""
    from src.diagnostics import RunDiagnostics

    return RunDiagnostics(
        run_id="20240101",
        start_time=datetime.now(),
        broad_topics_attempted=30,
        broad_topics_completed=28,
        broad_tweets_scraped=1500,
        trends_discovered=15,
        trends_filtered_by_llm=10,
        deep_dive_trends_attempted=10,
        deep_dive_trends_completed=10,
        deep_dive_tweets_scraped=500,
        twitter_accounts_total=5,
        twitter_accounts_active=4,
        twitter_accounts_rate_limited=1,
        llm_calls_made=3,
        llm_tokens_used=5000,
        fact_checks_performed=5,
        signal_strength="medium",
        notable=False,
        email_sent=True,
    )
