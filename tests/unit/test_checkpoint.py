"""
Unit tests for src/checkpoint.py

Tests checkpoint persistence and pipeline state management.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.checkpoint import CheckpointManager, PipelineState


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
        assert state.trends == []
        assert state.trends_completed == []

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

        state = manager.start_new_run()

        assert state.run_id == datetime.now().strftime("%Y%m%d")
        assert state.topics_completed == []
        assert state.trends_completed == []
        assert temp_checkpoint_file.exists()

    def test_save_and_load(self, temp_checkpoint_file):
        """Test saving and loading checkpoint state."""
        manager = CheckpointManager(str(temp_checkpoint_file))

        manager.start_new_run()
        manager.get_state().step1_complete = True
        manager.save()

        # Create new manager and load
        manager2 = CheckpointManager(str(temp_checkpoint_file))
        loaded_state = manager2.load()

        assert loaded_state is not None
        assert loaded_state.step1_complete is True

    def test_load_returns_none_if_no_file(self, temp_checkpoint_file):
        """Test that load returns None when no checkpoint exists."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        assert manager.load() is None

    def test_migrate_state_rejects_old_schema(self, temp_checkpoint_file):
        """Test that old checkpoint formats are rejected due to schema version mismatch."""
        manager = CheckpointManager(str(temp_checkpoint_file))

        # Write an old-format checkpoint without schema_version
        old_data = {
            "run_id": datetime.now().strftime("%Y%m%d"),
            "started_at": "2024-01-01T12:00:00",
            "step1_complete": True,
            "topics_completed": ["fintwit"],
            "topics_remaining": ["markets", "trading"],
            "broad_tweets": [{"id": 1, "text": "test"}],
        }
        with open(temp_checkpoint_file, "w") as f:
            json.dump(old_data, f)

        loaded = manager.load()
        assert loaded is None  # Old schema is discarded

    def test_migrate_state_strips_unknown_fields(self, temp_checkpoint_file):
        """Test that current-schema checkpoints with extra fields load correctly."""
        from src.checkpoint import SCHEMA_VERSION
        manager = CheckpointManager(str(temp_checkpoint_file))

        # Write a checkpoint with current schema but some unknown extra field
        data = {
            "run_id": datetime.now().strftime("%Y%m%d"),
            "started_at": "2024-01-01T12:00:00",
            "schema_version": SCHEMA_VERSION,
            "step1_complete": True,
            "topics_completed": ["fintwit"],
            "some_future_field": "should be stripped",
        }
        with open(temp_checkpoint_file, "w") as f:
            json.dump(data, f)

        loaded = manager.load()
        assert loaded is not None
        assert loaded.step1_complete is True
        assert loaded.topics_completed == ["fintwit"]

    def test_mark_topic_done(self, temp_checkpoint_file):
        """Test marking a topic as done."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()

        manager.mark_topic_done("fintwit")
        state = manager.get_state()

        assert "fintwit" in state.topics_completed

    def test_mark_topic_done_idempotent(self, temp_checkpoint_file):
        """Test that marking the same topic twice doesn't duplicate."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()

        manager.mark_topic_done("fintwit")
        manager.mark_topic_done("fintwit")
        state = manager.get_state()

        assert state.topics_completed.count("fintwit") == 1

    def test_save_trends(self, temp_checkpoint_file):
        """Test saving discovered trends."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()

        trends = ["$NVDA", "Silver", "#inflation"]
        manager.save_trends(trends)
        state = manager.get_state()

        assert state.trends == trends
        assert state.step2_complete is True

    def test_mark_trend_done(self, temp_checkpoint_file):
        """Test marking a trend as done."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()
        manager.save_trends(["$NVDA"])

        manager.mark_trend_done("$NVDA")
        state = manager.get_state()

        assert "$NVDA" in state.trends_completed

    def test_mark_trend_done_idempotent(self, temp_checkpoint_file):
        """Test that marking the same trend twice doesn't duplicate."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()

        manager.mark_trend_done("$NVDA")
        manager.mark_trend_done("$NVDA")
        state = manager.get_state()

        assert state.trends_completed.count("$NVDA") == 1

    def test_save_analysis(self, temp_checkpoint_file):
        """Test saving analysis results."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()

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
        manager.start_new_run()

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
        manager.start_new_run()
        manager.complete_step1()

        # New manager should detect resumable state
        manager2 = CheckpointManager(str(temp_checkpoint_file))
        assert manager2.should_resume() is True

    def test_should_not_resume_completed_run(self, temp_checkpoint_file):
        """Test that completed runs don't trigger resume."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()
        manager.complete_step6()  # Mark as fully complete

        manager2 = CheckpointManager(str(temp_checkpoint_file))
        assert manager2.should_resume() is False

    def test_clear(self, temp_checkpoint_file):
        """Test clearing checkpoint file."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()
        assert temp_checkpoint_file.exists()

        manager.clear()
        assert not temp_checkpoint_file.exists()

    def test_set_error(self, temp_checkpoint_file):
        """Test recording an error."""
        manager = CheckpointManager(str(temp_checkpoint_file))
        manager.start_new_run()

        manager.set_error("Test error message")
        state = manager.get_state()

        assert state.error == "Test error message"

    def test_get_state_raises_without_init(self, temp_checkpoint_file):
        """Test that get_state raises error without initialization."""
        manager = CheckpointManager(str(temp_checkpoint_file))

        with pytest.raises(RuntimeError, match="No active state"):
            manager.get_state()
