"""
Unit tests for src/history.py

Tests SQLite storage, historical digest retrieval, and signal strength calculation.
"""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from src.history import DigestHistory, HistoricalDigest, calculate_signal_strength


class TestHistoricalDigest:
    """Tests for HistoricalDigest dataclass."""

    def test_historical_digest_creation(self):
        """Test creating a HistoricalDigest."""
        digest = HistoricalDigest(
            id=1,
            run_date=datetime.now(),
            trends=["$NVDA", "Silver"],
            tweet_count=500,
            digest_text="Test digest content.",
            signal_strength="medium",
            top_engagement=10000.0,
            notable=True,
        )

        assert digest.id == 1
        assert len(digest.trends) == 2
        assert digest.tweet_count == 500
        assert digest.notable is True


class TestDigestHistory:
    """Tests for DigestHistory class."""

    def test_init_creates_tables(self, temp_db_path):
        """Test that initialization creates database tables."""
        history = DigestHistory(db_path=temp_db_path)

        # Verify tables exist
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        assert "digests" in tables
        assert "trend_history" in tables

    def test_store_digest(self, temp_db_path):
        """Test storing a digest."""
        history = DigestHistory(db_path=temp_db_path)

        digest_id = history.store_digest(
            trends=["$NVDA", "Silver", "Oil"],
            tweet_count=500,
            digest_text="Test digest content.",
            signal_strength="high",
            top_engagement=25000.0,
            notable=True,
        )

        assert digest_id is not None
        assert digest_id > 0

    def test_store_digest_with_trend_details(self, temp_db_path):
        """Test storing digest with trend details."""
        history = DigestHistory(db_path=temp_db_path)

        # Use lowercase keys since store_digest normalizes to lowercase
        trend_details = {
            "nvda": {"mentions": 100, "engagement": 15000},
            "silver": {"mentions": 50, "engagement": 8000},
        }

        digest_id = history.store_digest(
            trends=["$NVDA", "Silver"],
            tweet_count=300,
            digest_text="Test digest.",
            signal_strength="medium",
            top_engagement=15000.0,
            trend_details=trend_details,
        )

        # Verify trend history was stored - use longer lookback since dates may differ
        trend_hist = history.get_trend_history("nvda", days=30)
        assert len(trend_hist) == 1
        assert trend_hist[0]["mentions"] == 100

    def test_get_recent_digests(self, temp_db_path):
        """Test retrieving recent digests."""
        history = DigestHistory(db_path=temp_db_path)

        # Store multiple digests
        for i in range(3):
            history.store_digest(
                trends=[f"Trend{i}"],
                tweet_count=100 * (i + 1),
                digest_text=f"Digest {i}",
                signal_strength="low",
                top_engagement=1000.0 * (i + 1),
            )

        recent = history.get_recent_digests(days=7)

        assert len(recent) == 3
        assert all(isinstance(d, HistoricalDigest) for d in recent)

    def test_get_trend_history(self, temp_db_path):
        """Test retrieving trend history."""
        history = DigestHistory(db_path=temp_db_path)

        # Store digest with trend details
        history.store_digest(
            trends=["$AAPL"],
            tweet_count=200,
            digest_text="Apple trending.",
            signal_strength="medium",
            top_engagement=12000.0,
            trend_details={"aapl": {"mentions": 75, "engagement": 12000}},
        )

        trend_hist = history.get_trend_history("aapl", days=30)

        assert len(trend_hist) == 1
        assert trend_hist[0]["mentions"] == 75
        assert trend_hist[0]["engagement"] == 12000

    def test_get_trend_history_empty(self, temp_db_path):
        """Test trend history returns empty for unknown trends."""
        history = DigestHistory(db_path=temp_db_path)
        trend_hist = history.get_trend_history("unknown_trend", days=30)
        assert trend_hist == []

    def test_get_all_recent_trends(self, temp_db_path):
        """Test retrieving all recent trends."""
        history = DigestHistory(db_path=temp_db_path)

        history.store_digest(
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

        all_trends = history.get_all_recent_trends(days=7)

        assert "nvda" in all_trends
        assert "aapl" in all_trends
        assert len(all_trends["nvda"]) == 1

    def test_get_baseline_stats(self, temp_db_path):
        """Test calculating baseline statistics."""
        history = DigestHistory(db_path=temp_db_path)

        # Store several digests
        for i in range(5):
            history.store_digest(
                trends=[f"Trend{i}"],
                tweet_count=100 + i * 50,
                digest_text=f"Digest {i}",
                signal_strength="low" if i < 4 else "high",
                top_engagement=5000.0 + i * 1000,
                notable=i == 4,  # One notable day
            )

        stats = history.get_baseline_stats(days=30)

        assert stats["total_runs"] == 5
        assert stats["notable_days"] == 1
        assert stats["avg_tweets"] > 0
        assert stats["avg_top_engagement"] > 0

    def test_get_baseline_stats_empty(self, temp_db_path):
        """Test baseline stats with no data."""
        history = DigestHistory(db_path=temp_db_path)
        stats = history.get_baseline_stats(days=30)

        assert stats["total_runs"] == 0
        assert stats["avg_top_engagement"] == 0
        assert stats["notable_rate"] == 0

    def test_format_context_for_llm(self, temp_db_path):
        """Test LLM context formatting."""
        history = DigestHistory(db_path=temp_db_path)

        # First run - empty history
        context = history.format_context_for_llm(days=7)
        assert "first run" in context.lower() or "no historical data" in context.lower()

        # Store some data
        history.store_digest(
            trends=["$NVDA", "Silver"],
            tweet_count=500,
            digest_text="Market digest.",
            signal_strength="medium",
            top_engagement=15000.0,
            notable=True,
        )

        # Now should have context
        context = history.format_context_for_llm(days=7)
        assert "Historical Context" in context
        assert "NVDA" in context or "Silver" in context


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
            top_engagement=50000.0,  # 10x baseline
            trend_count=5,
            baseline_engagement=5000.0,
        )
        assert result == "high"

    def test_medium_signal(self):
        """Test medium signal detection."""
        result = calculate_signal_strength(
            top_engagement=15000.0,  # 3x baseline
            trend_count=3,
            baseline_engagement=5000.0,
        )
        assert result == "medium"

    def test_low_signal(self):
        """Test low signal detection."""
        result = calculate_signal_strength(
            top_engagement=4000.0,  # Below baseline
            trend_count=2,
            baseline_engagement=5000.0,
        )
        assert result == "low"

    def test_none_signal_below_threshold(self):
        """Test none signal for very low engagement."""
        result = calculate_signal_strength(
            top_engagement=2000.0,  # Well below baseline
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
        # Should handle gracefully
        assert result in ["high", "medium", "low", "none"]
