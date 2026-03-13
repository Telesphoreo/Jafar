"""
Unit tests for src/history.py

Tests async SQLAlchemy storage, historical digest retrieval, and signal strength calculation.
"""

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.history import DigestHistory, calculate_signal_strength
from src.models import Base, Digest, TrendHistory


@pytest.fixture
async def async_session_factory():
    """Create an in-memory SQLite async engine and session factory for tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def history(async_session_factory):
    """Provide a DigestHistory that uses the test session factory."""
    h = DigestHistory()
    # Patch get_session to return sessions from our test factory
    async def _get_session():
        return async_session_factory()
    with patch("src.history.get_session", side_effect=_get_session):
        yield h


@pytest.fixture
def patched_get_session(async_session_factory):
    """Patch get_session globally and return a DigestHistory instance."""
    async def _get_session():
        return async_session_factory()
    patcher = patch("src.history.get_session", side_effect=_get_session)
    patcher.start()
    yield DigestHistory()
    patcher.stop()


class TestDigestHistory:
    """Tests for DigestHistory class."""

    async def test_store_digest(self, patched_get_session):
        """Test storing a digest."""
        history = patched_get_session
        digest_id = await history.store_digest(
            trends=["$NVDA", "Silver", "Oil"],
            tweet_count=500,
            digest_text="Test digest content.",
            signal_strength="high",
            top_engagement=25000.0,
            notable=True,
        )

        assert digest_id is not None
        assert digest_id > 0

    async def test_store_digest_with_trend_details(self, patched_get_session):
        """Test storing digest with trend details."""
        history = patched_get_session
        trend_details = {
            "nvda": {"mentions": 100, "engagement": 15000},
            "silver": {"mentions": 50, "engagement": 8000},
        }

        digest_id = await history.store_digest(
            trends=["$NVDA", "Silver"],
            tweet_count=300,
            digest_text="Test digest.",
            signal_strength="medium",
            top_engagement=15000.0,
            trend_details=trend_details,
        )

        assert digest_id is not None

        # Verify trend history was stored
        trend_hist = await history.get_trend_history("nvda", days=30)
        assert len(trend_hist) == 1
        assert trend_hist[0]["mentions"] == 100

    async def test_get_recent_digests(self, patched_get_session):
        """Test retrieving recent digests."""
        history = patched_get_session

        for i in range(3):
            await history.store_digest(
                trends=[f"Trend{i}"],
                tweet_count=100 * (i + 1),
                digest_text=f"Digest {i}",
                signal_strength="low",
                top_engagement=1000.0 * (i + 1),
            )

        recent = await history.get_recent_digests(days=7)

        assert len(recent) == 3
        assert all(isinstance(d, dict) for d in recent)
        assert "trends" in recent[0]
        assert "tweet_count" in recent[0]
        assert "signal_strength" in recent[0]

    async def test_get_trend_history(self, patched_get_session):
        """Test retrieving trend history."""
        history = patched_get_session

        await history.store_digest(
            trends=["$AAPL"],
            tweet_count=200,
            digest_text="Apple trending.",
            signal_strength="medium",
            top_engagement=12000.0,
            trend_details={"aapl": {"mentions": 75, "engagement": 12000}},
        )

        trend_hist = await history.get_trend_history("aapl", days=30)

        assert len(trend_hist) == 1
        assert trend_hist[0]["mentions"] == 75
        assert trend_hist[0]["engagement"] == 12000

    async def test_get_trend_history_empty(self, patched_get_session):
        """Test trend history returns empty for unknown trends."""
        history = patched_get_session
        trend_hist = await history.get_trend_history("unknown_trend", days=30)
        assert trend_hist == []

    async def test_get_all_recent_trends(self, patched_get_session):
        """Test retrieving all recent trends aggregated."""
        history = patched_get_session

        await history.store_digest(
            trends=["$NVDA", "$AAPL"],
            tweet_count=400,
            digest_text="Tech stocks trending.",
            signal_strength="high",
            top_engagement=20000.0,
            trend_details={
                "nvda": {"mentions": 80, "engagement": 15000},
                "aapl": {"mentions": 60, "engagement": 10000},
            },
        )

        all_trends = await history.get_all_recent_trends(days=7)

        assert isinstance(all_trends, list)
        trend_terms = [t["trend_term"] for t in all_trends]
        assert "nvda" in trend_terms
        assert "aapl" in trend_terms

    async def test_get_baseline_stats(self, patched_get_session):
        """Test calculating baseline statistics."""
        history = patched_get_session

        for i in range(5):
            await history.store_digest(
                trends=[f"Trend{i}"],
                tweet_count=100 + i * 50,
                digest_text=f"Digest {i}",
                signal_strength="low" if i < 4 else "high",
                top_engagement=5000.0 + i * 1000,
                notable=i == 4,
            )

        stats = await history.get_baseline_stats(days=30)

        assert stats["total_digests"] == 5
        assert stats["avg_tweet_count"] > 0
        assert stats["avg_engagement"] > 0

    async def test_get_baseline_stats_empty(self, patched_get_session):
        """Test baseline stats with no data."""
        history = patched_get_session
        stats = await history.get_baseline_stats(days=30)

        assert stats["total_digests"] == 0
        assert stats["avg_engagement"] == 0

    async def test_get_previous_digest_summary(self, patched_get_session):
        """Test extracting summary from previous digest."""
        history = patched_get_session

        await history.store_digest(
            trends=["Gold"],
            tweet_count=100,
            digest_text="## Summary\nGold up 2%, Fed holds rates.\n\n**Assessment:**\nQuiet day.",
            signal_strength="low",
            top_engagement=5000.0,
        )

        summary = await history.get_previous_digest_summary(days_back=1)
        assert summary is not None
        assert "Gold up 2%" in summary

    async def test_get_previous_digest_summary_empty(self, patched_get_session):
        """Test empty summary when no digests exist."""
        history = patched_get_session
        summary = await history.get_previous_digest_summary(days_back=1)
        assert summary is None

    async def test_format_context_for_llm_empty(self, patched_get_session):
        """Test LLM context formatting with no data."""
        history = patched_get_session
        context = await history.format_context_for_llm(days=7)
        assert "first run" in context.lower() or "no historical data" in context.lower()

    async def test_format_context_for_llm_with_data(self, patched_get_session):
        """Test LLM context formatting with data."""
        history = patched_get_session

        await history.store_digest(
            trends=["$NVDA", "Silver"],
            tweet_count=500,
            digest_text="Market digest.",
            signal_strength="medium",
            top_engagement=15000.0,
            notable=True,
        )

        context = await history.format_context_for_llm(days=7)
        assert "Historical Context" in context


class TestCalculateSignalStrength:
    """Tests for calculate_signal_strength function."""

    def test_no_trends_returns_none(self):
        """Test that zero trends returns 'none' signal."""
        result = calculate_signal_strength(
            top_engagement=10000.0,
            trend_count=0,
            baseline_engagement=5000.0,
        )
        assert result == "none"

    def test_high_signal(self):
        """Test high signal detection."""
        result = calculate_signal_strength(
            top_engagement=50000.0,
            trend_count=5,
            baseline_engagement=5000.0,
        )
        assert result == "high"

    def test_medium_signal(self):
        """Test medium signal detection."""
        result = calculate_signal_strength(
            top_engagement=15000.0,
            trend_count=3,
            baseline_engagement=5000.0,
        )
        assert result == "medium"

    def test_low_signal(self):
        """Test low signal detection."""
        result = calculate_signal_strength(
            top_engagement=4000.0,
            trend_count=2,
            baseline_engagement=5000.0,
        )
        assert result == "low"

    def test_none_signal_below_threshold(self):
        """Test none signal for very low engagement."""
        result = calculate_signal_strength(
            top_engagement=2000.0,
            trend_count=1,
            baseline_engagement=5000.0,
        )
        assert result == "none"

    def test_handles_zero_baseline(self):
        """Test that zero baseline doesn't cause division error."""
        result = calculate_signal_strength(
            top_engagement=10000.0,
            trend_count=3,
            baseline_engagement=0,
        )
        assert result in ["high", "medium", "low", "none"]
