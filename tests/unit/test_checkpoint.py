"""
Unit tests for src/checkpoint.py

Tests checkpoint persistence, serialization, and pipeline state management.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.checkpoint import CheckpointManager, PipelineState
from src.scraper import ScrapedTweet


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    def test_default_state(self):
        """Test default PipelineState values."""
        state = PipelineState(
            run_id="20240101",
            started_at="2024-01-01T12:00:00",
        )

        assert state.step1_complete is False
        assert state.step2_complete is False
        assert state.step3_complete is False
        assert state.step4_complete is False
        assert state.step5_complete is False
        assert state.step6_complete is False
        assert state.topics_completed == []
        assert state.topics_remaining == []
        assert state.broad_tweets == []
        assert state.trends == []

    def test_state_with_data(self):
        """Test PipelineState with populated data."""
        state = PipelineState(
            run_id="20240101",
            started_at="2024-01-01T12:00:00",
            step1_complete=True,
            topics_completed=["fintwit", "markets"],
            trends=["$NVDA", "Silver"],
            analysis="Test analysis content",
        )

        assert state.step1_complete is True
        assert len(state.topics_completed) == 2
        assert len(state.trends) == 2
        assert state.analysis == "Test analysis content"


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_init_creates_directory(self, temp_checkpoint_file):
        """Test that CheckpointManager creates parent directory."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        assert temp_checkpoint_file.parent.exists()

    def test_start_new_run(self, temp_checkpoint_file):
        """Test starting a new pipeline run."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        topics = ["fintwit", "markets", "trading"]

        state = manager.start_new_run(topics)

        assert state.run_id == datetime.now().strftime("%Y%m%d")
        assert state.topics_remaining == topics
        assert state.topics_completed == []
        assert temp_checkpoint_file.exists()

    def test_save_and_load(self, temp_checkpoint_file):
        """Test saving and loading checkpoint state."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        topics = ["fintwit", "markets"]

        manager.start_new_run(topics)
        manager.get_state().step1_complete = True
        manager.save()

        # Create new manager and load
        manager2 = CheckpointManager(str(temp_checkpoint_file))
        loaded_state = manager2.load()

        assert loaded_state is not None
        assert loaded_state.step1_complete is True
        assert loaded_state.topics_remaining == topics

    def test_load_returns_none_if_no_file(self, temp_checkpoint_file):
        """Test that load returns None when no checkpoint exists."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        assert manager.load() is None

    def test_serialize_tweet(self, temp_checkpoint_file, sample_tweet):
        """Test tweet serialization."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        serialized = manager._serialize_tweet(sample_tweet)

        assert serialized["id"] == sample_tweet.id
        assert serialized["text"] == sample_tweet.text
        assert serialized["username"] == sample_tweet.username
        assert serialized["likes"] == sample_tweet.likes

    def test_deserialize_tweet(self, temp_checkpoint_file, sample_tweet):
        """Test tweet deserialization."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        serialized = manager._serialize_tweet(sample_tweet)
        deserialized = manager._deserialize_tweet(serialized)

        assert deserialized.id == sample_tweet.id
        assert deserialized.text == sample_tweet.text
        assert deserialized.username == sample_tweet.username
        assert deserialized.likes == sample_tweet.likes

    def test_mark_topic_complete(self, temp_checkpoint_file, sample_tweets):
        """Test marking a topic as complete with tweets."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit", "markets"])

        manager.mark_topic_complete("fintwit", sample_tweets[:5])
        state = manager.get_state()

        assert "fintwit" in state.topics_completed
        assert "fintwit" not in state.topics_remaining
        assert len(state.broad_tweets) == 5

    def test_mark_topic_complete_empty_tweets_triggers_retry(
        self, temp_checkpoint_file
    ):
        """Test that empty tweets trigger retry mechanism."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])

        # First call with empty tweets - should trigger retry
        manager.mark_topic_complete("fintwit", [])
        state = manager.get_state()

        # Topic should still be remaining (not completed) due to retry
        assert "fintwit" in state.topics_remaining
        assert "fintwit" not in state.topics_completed
        assert state.retry_counts.get("fintwit") == 1

    def test_get_broad_tweets(self, temp_checkpoint_file, sample_tweets):
        """Test retrieving all broad tweets."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit", "markets"])

        manager.mark_topic_complete("fintwit", sample_tweets[:5])
        manager.mark_topic_complete("markets", sample_tweets[5:])

        tweets = manager.get_broad_tweets()
        assert len(tweets) == len(sample_tweets)
        assert all(isinstance(t, ScrapedTweet) for t in tweets)

    def test_save_trends(self, temp_checkpoint_file):
        """Test saving discovered trends."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])

        trends = ["$NVDA", "Silver", "#inflation"]
        manager.save_trends(trends)
        state = manager.get_state()

        assert state.trends == trends
        assert state.step2_complete is True

    def test_mark_trend_scraped(self, temp_checkpoint_file, sample_tweets):
        """Test marking a trend as scraped."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])
        manager.save_trends(["$NVDA"])

        manager.mark_trend_scraped("$NVDA", sample_tweets[:3])
        trend_tweets = manager.get_trend_tweets()

        assert "$NVDA" in trend_tweets
        assert len(trend_tweets["$NVDA"]) == 3

    def test_save_analysis(self, temp_checkpoint_file):
        """Test saving analysis results."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])

        manager.save_analysis(
            analysis="Test analysis content",
            signal_strength="high",
            is_notable=True,
            top_engagement=50000.0,
        )
        state = manager.get_state()

        assert state.analysis == "Test analysis content"
        assert state.signal_strength == "high"
        assert state.is_notable is True
        assert state.top_engagement == 50000.0
        assert state.step4_complete is True

    def test_complete_steps(self, temp_checkpoint_file):
        """Test completing various pipeline steps."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])

        manager.complete_step1()
        assert manager.get_state().step1_complete is True

        manager.complete_step3()
        assert manager.get_state().step3_complete is True

        manager.complete_step5()
        assert manager.get_state().step5_complete is True

        manager.complete_step6()
        assert manager.get_state().step6_complete is True

    def test_should_resume_same_day(self, temp_checkpoint_file):
        """Test resume detection for same-day incomplete run."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])
        manager.complete_step1()

        # New manager should detect resumable state
        manager2 = CheckpointManager(str(temp_checkpoint_file))
        assert manager2.should_resume() is True

    def test_should_not_resume_completed_run(self, temp_checkpoint_file):
        """Test that completed runs don't trigger resume."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])
        manager.complete_step6()  # Mark as fully complete

        manager2 = CheckpointManager(str(temp_checkpoint_file))
        assert manager2.should_resume() is False

    def test_clear(self, temp_checkpoint_file):
        """Test clearing checkpoint file."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])
        assert temp_checkpoint_file.exists()

        manager.clear()
        assert not temp_checkpoint_file.exists()

    def test_set_error(self, temp_checkpoint_file):
        """Test recording an error."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run(["fintwit"])

        manager.set_error("Test error message")
        state = manager.get_state()

        assert state.error == "Test error message"

    def test_get_state_raises_without_init(self, temp_checkpoint_file):
        """Test that get_state raises error without initialization."""
        manager = CheckpointManager(str(temp_checkpoint_file))

        with pytest.raises(RuntimeError, match="No active state"):
            manager.get_state()
