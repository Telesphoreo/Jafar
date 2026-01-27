"""
Unit tests for src/temporal_analyzer.py

Tests trend timeline detection, consecutive day tracking, and recurring theme detection.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.temporal_analyzer import TemporalTrendAnalyzer, TrendTimeline


class TestTrendTimeline:
    """Tests for TrendTimeline dataclass."""

    def test_new_trend(self):
        """Test detection of brand new trends."""
        timeline = TrendTimeline(
            term="$NVDA",
            term_normalized="nvda",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=50,
            engagement_today=10000.0,
            total_appearances=1,
            consecutive_days=0,
        )

        assert timeline.is_new is True
        assert timeline.is_continuing is False
        assert timeline.is_recurring is False

    def test_continuing_trend(self):
        """Test detection of continuing trends."""
        timeline = TrendTimeline(
            term="Silver",
            term_normalized="silver",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=100,
            engagement_today=20000.0,
            total_appearances=5,
            consecutive_days=3,
            last_seen_date=datetime.now() - timedelta(days=1),
            days_since_last=1,
        )

        assert timeline.is_new is False
        assert timeline.is_continuing is True
        assert timeline.is_recurring is False

    def test_recurring_trend(self):
        """Test detection of recurring trends."""
        timeline = TrendTimeline(
            term="Uranium",
            term_normalized="uranium",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=30,
            engagement_today=5000.0,
            total_appearances=10,
            consecutive_days=1,
            last_seen_date=datetime.now() - timedelta(days=30),
            days_since_last=30,
        )

        assert timeline.is_new is False
        assert timeline.is_continuing is False
        assert timeline.is_recurring is True

    def test_temporal_badge_new(self):
        """Test badge for new trends."""
        timeline = TrendTimeline(
            term="$NEW",
            term_normalized="new",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=10,
            engagement_today=1000.0,
            total_appearances=1,
        )

        assert timeline.temporal_badge == "New"

    def test_temporal_badge_consecutive_days(self):
        """Test badge for multi-day trends."""
        timeline = TrendTimeline(
            term="$TSLA",
            term_normalized="tsla",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=100,
            engagement_today=20000.0,
            total_appearances=5,
            consecutive_days=5,
        )

        assert "Day 5" in timeline.temporal_badge

    def test_temporal_badge_recurring(self):
        """Test badge for recurring trends."""
        timeline = TrendTimeline(
            term="Gold",
            term_normalized="gold",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=50,
            engagement_today=8000.0,
            total_appearances=20,
            consecutive_days=1,
            days_since_last=45,
        )

        assert "Last seen" in timeline.temporal_badge
        assert "ago" in timeline.temporal_badge

    def test_temporal_badge_recurring_long_gap(self):
        """Test badge for trends with very long gaps."""
        timeline = TrendTimeline(
            term="OldTrend",
            term_normalized="oldtrend",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=20,
            engagement_today=3000.0,
            total_appearances=5,
            consecutive_days=1,
            days_since_last=200,  # ~6 months
        )

        assert "mo ago" in timeline.temporal_badge

    def test_trend_velocity_emerging(self):
        """Test velocity detection for emerging trends."""
        timeline = TrendTimeline(
            term="$NEW",
            term_normalized="new",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=50,
            engagement_today=5000.0,
            total_appearances=1,
            previous_mentions=[],  # No history
        )

        assert timeline.trend_velocity == "emerging"

    def test_trend_velocity_accelerating(self):
        """Test velocity detection for accelerating trends."""
        timeline = TrendTimeline(
            term="$ACCEL",
            term_normalized="accel",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=100,  # 2x recent average
            engagement_today=20000.0,
            total_appearances=5,
            previous_mentions=[30, 40, 50],  # Average ~40
        )

        assert timeline.trend_velocity == "accelerating"

    def test_trend_velocity_declining(self):
        """Test velocity detection for declining trends."""
        timeline = TrendTimeline(
            term="$DECL",
            term_normalized="decl",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=20,  # Much lower than average
            engagement_today=2000.0,
            total_appearances=5,
            previous_mentions=[100, 90, 80],  # Average ~90
        )

        assert timeline.trend_velocity == "declining"

    def test_trend_velocity_stable(self):
        """Test velocity detection for stable trends."""
        timeline = TrendTimeline(
            term="$STBL",
            term_normalized="stbl",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=50,  # Similar to average
            engagement_today=5000.0,
            total_appearances=5,
            previous_mentions=[45, 50, 55],  # Average ~50
        )

        assert timeline.trend_velocity == "stable"

    def test_format_for_llm_new(self):
        """Test LLM formatting for new trends."""
        timeline = TrendTimeline(
            term="$NEW",
            term_normalized="new",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=10,
            engagement_today=1000.0,
            total_appearances=1,
        )

        formatted = timeline.format_for_llm()
        assert "New trend" in formatted
        assert "First appearance" in formatted

    def test_format_for_llm_continuing(self):
        """Test LLM formatting for continuing trends."""
        timeline = TrendTimeline(
            term="$CONT",
            term_normalized="cont",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=100,
            engagement_today=15000.0,
            total_appearances=5,
            consecutive_days=4,
            previous_mentions=[80, 90, 100],
        )

        formatted = timeline.format_for_llm()
        assert "Developing story" in formatted
        assert "Day 4" in formatted


class TestTemporalTrendAnalyzer:
    """Tests for TemporalTrendAnalyzer class."""

    @pytest.fixture
    def mock_history_db(self):
        """Create a mock history database."""
        mock_db = MagicMock()
        mock_db.get_trend_history.return_value = []
        return mock_db

    def test_init(self, mock_history_db):
        """Test analyzer initialization."""
        analyzer = TemporalTrendAnalyzer(
            history_db=mock_history_db,
            consecutive_threshold=3,
            gap_threshold_days=14,
        )

        assert analyzer.consecutive_threshold == 3
        assert analyzer.gap_threshold == 14

    def test_analyze_new_trend(self, mock_history_db):
        """Test analysis of a brand new trend."""
        mock_history_db.get_trend_history.return_value = []

        analyzer = TemporalTrendAnalyzer(history_db=mock_history_db)
        timeline = analyzer.analyze_trend_timeline(
            term="$NVDA",
            mentions_today=50,
            engagement_today=10000.0,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        assert timeline.is_new is True
        assert timeline.total_appearances == 1
        mock_history_db.get_trend_history.assert_called_once()

    def test_analyze_consecutive_trend(self, mock_history_db):
        """Test analysis of a trend with consecutive days."""
        today = datetime.now().date()
        mock_history_db.get_trend_history.return_value = [
            {"date": (today - timedelta(days=1)).isoformat(), "mentions": 40, "engagement": 8000},
            {"date": (today - timedelta(days=2)).isoformat(), "mentions": 35, "engagement": 7000},
            {"date": (today - timedelta(days=3)).isoformat(), "mentions": 30, "engagement": 6000},
        ]

        analyzer = TemporalTrendAnalyzer(history_db=mock_history_db)
        timeline = analyzer.analyze_trend_timeline(
            term="Silver",
            mentions_today=50,
            engagement_today=10000.0,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        assert timeline.is_new is False
        assert timeline.consecutive_days >= 3
        assert timeline.total_appearances == 4  # 3 historical + today

    def test_analyze_recurring_trend(self, mock_history_db):
        """Test analysis of a recurring trend with gap."""
        today = datetime.now().date()
        mock_history_db.get_trend_history.return_value = [
            {"date": (today - timedelta(days=30)).isoformat(), "mentions": 40, "engagement": 8000},
            {"date": (today - timedelta(days=31)).isoformat(), "mentions": 35, "engagement": 7000},
        ]

        analyzer = TemporalTrendAnalyzer(history_db=mock_history_db)
        timeline = analyzer.analyze_trend_timeline(
            term="Uranium",
            mentions_today=50,
            engagement_today=10000.0,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        assert timeline.is_new is False
        assert timeline.is_recurring is True
        assert timeline.days_since_last >= 30

    def test_analyze_all_trends(self, mock_history_db):
        """Test analyzing multiple trends at once."""
        mock_history_db.get_trend_history.return_value = []

        analyzer = TemporalTrendAnalyzer(history_db=mock_history_db)
        trend_details = {
            "$NVDA": {
                "mentions": 50,
                "engagement": 10000.0,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
            },
            "Silver": {
                "mentions": 30,
                "engagement": 5000.0,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
            },
        }

        timelines = analyzer.analyze_all_trends(trend_details)

        assert len(timelines) == 2
        assert "$NVDA" in timelines
        assert "Silver" in timelines

    def test_format_context_for_llm_empty(self, mock_history_db):
        """Test LLM context with no timelines."""
        analyzer = TemporalTrendAnalyzer(history_db=mock_history_db)
        context = analyzer.format_context_for_llm({})

        assert context == ""

    def test_format_context_for_llm_with_data(self, mock_history_db):
        """Test LLM context formatting with timelines."""
        mock_history_db.get_trend_history.return_value = []

        analyzer = TemporalTrendAnalyzer(history_db=mock_history_db)
        trend_details = {
            "$NVDA": {
                "mentions": 50,
                "engagement": 10000.0,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
            },
        }

        timelines = analyzer.analyze_all_trends(trend_details)
        context = analyzer.format_context_for_llm(timelines)

        assert "Temporal Context" in context
        assert "New Trends" in context

    def test_term_normalization(self, mock_history_db):
        """Test that terms are normalized correctly."""
        mock_history_db.get_trend_history.return_value = []

        analyzer = TemporalTrendAnalyzer(history_db=mock_history_db)
        timeline = analyzer.analyze_trend_timeline(
            term="$NVDA",
            mentions_today=50,
            engagement_today=10000.0,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        # Should strip $ prefix for matching
        assert timeline.term_normalized == "nvda"
        mock_history_db.get_trend_history.assert_called_with("nvda", days=180)
