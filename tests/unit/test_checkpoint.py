"""
Unit tests for src/checkpoint.py

Tests database-backed checkpoint persistence, pipeline state management,
and advisory lock behavior.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.checkpoint import CheckpointManager, PipelineState
from src.models import Base, PipelineRun


@pytest.fixture
async def checkpoint_db():
    """Create an in-memory SQLite async engine and patch get_session/get_engine.

    SQLite doesn't support advisory locks, so we patch initialize()
    to skip the lock acquisition for unit tests.
    """
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_session():
        return factory()

    with (
        patch("src.checkpoint.get_session", side_effect=_get_session),
        patch("src.checkpoint.get_engine", return_value=engine),
    ):
        yield factory

    await engine.dispose()


@pytest.fixture
async def checkpoint(checkpoint_db):
    """Provide a CheckpointManager connected to the test DB.

    Skips advisory lock since SQLite doesn't support it.
    """
    mgr = CheckpointManager()
    # Mark the lock as "acquired" without actually calling pg_try_advisory_lock
    mgr._lock_conn = AsyncMock()
    yield mgr
    mgr._lock_conn = None


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

    async def test_start_new_run(self, checkpoint):
        """Test starting a new pipeline run."""
        state = await checkpoint.start_new_run()

        assert state.run_id == datetime.now().strftime("%Y%m%d")
        assert state.topics_completed == []
        assert state.trends_completed == []

    async def test_save_and_resume(self, checkpoint, checkpoint_db):
        """Test that state persists to DB and can be resumed."""
        await checkpoint.start_new_run()
        await checkpoint.complete_step1()

        # Create a fresh manager (simulates process restart)
        mgr2 = CheckpointManager()
        mgr2._lock_conn = AsyncMock()
        assert await mgr2.should_resume() is True
        assert mgr2.get_state().step1_complete is True

    async def test_should_resume_no_data(self, checkpoint):
        """Test that should_resume returns False when no checkpoint exists."""
        assert await checkpoint.should_resume() is False

    async def test_should_resume_completed_run(self, checkpoint, checkpoint_db):
        """Test that completed runs don't trigger resume."""
        await checkpoint.start_new_run()
        await checkpoint.clear()

        mgr2 = CheckpointManager()
        mgr2._lock_conn = AsyncMock()
        assert await mgr2.should_resume() is False

    async def test_mark_topic_done(self, checkpoint):
        """Test marking a topic as done."""
        await checkpoint.start_new_run()
        await checkpoint.mark_topic_done("fintwit")

        assert "fintwit" in checkpoint.get_state().topics_completed

    async def test_mark_topic_done_idempotent(self, checkpoint):
        """Test that marking the same topic twice doesn't duplicate."""
        await checkpoint.start_new_run()
        await checkpoint.mark_topic_done("fintwit")
        await checkpoint.mark_topic_done("fintwit")

        assert checkpoint.get_state().topics_completed.count("fintwit") == 1

    async def test_save_trends(self, checkpoint):
        """Test saving discovered trends."""
        await checkpoint.start_new_run()

        trends = ["$NVDA", "Silver", "#inflation"]
        await checkpoint.save_trends(trends)
        state = checkpoint.get_state()

        assert state.trends == trends
        assert state.step2_complete is True

    async def test_mark_trend_done(self, checkpoint):
        """Test marking a trend as done."""
        await checkpoint.start_new_run()
        await checkpoint.save_trends(["$NVDA"])
        await checkpoint.mark_trend_done("$NVDA")

        assert "$NVDA" in checkpoint.get_state().trends_completed

    async def test_mark_trend_done_idempotent(self, checkpoint):
        """Test that marking the same trend twice doesn't duplicate."""
        await checkpoint.start_new_run()
        await checkpoint.mark_trend_done("$NVDA")
        await checkpoint.mark_trend_done("$NVDA")

        assert checkpoint.get_state().trends_completed.count("$NVDA") == 1

    async def test_save_analysis(self, checkpoint):
        """Test saving analysis results."""
        await checkpoint.start_new_run()

        await checkpoint.save_analysis(
            analysis="Test analysis content",
            signal_strength="high",
            is_notable=True,
            top_engagement=50000.0,
        )
        state = checkpoint.get_state()

        assert state.analysis == "Test analysis content"
        assert state.signal_strength == "high"
        assert state.is_notable is True
        assert state.top_engagement == 50000.0
        assert state.step4_complete is True

    async def test_complete_steps(self, checkpoint):
        """Test completing various pipeline steps."""
        await checkpoint.start_new_run()

        await checkpoint.complete_step1()
        assert checkpoint.get_state().step1_complete is True

        await checkpoint.complete_step3()
        assert checkpoint.get_state().step3_complete is True

        await checkpoint.complete_step5()
        assert checkpoint.get_state().step5_complete is True

        await checkpoint.complete_step6()
        assert checkpoint.get_state().step6_complete is True

    async def test_clear_marks_completed(self, checkpoint, checkpoint_db):
        """Test that clear marks the run as completed in DB."""
        await checkpoint.start_new_run()
        run_id = checkpoint.get_state().run_id

        await checkpoint.clear()

        # Verify status in DB
        session = checkpoint_db()
        run = await session.get(PipelineRun, run_id)
        await session.close()

        assert run.status == "completed"

    async def test_set_error(self, checkpoint, checkpoint_db):
        """Test recording an error."""
        await checkpoint.start_new_run()

        await checkpoint.set_error("Test error message")

        assert checkpoint.get_state().error == "Test error message"

        # Verify in DB
        run_id = checkpoint.get_state().run_id
        session = checkpoint_db()
        run = await session.get(PipelineRun, run_id)
        await session.close()

        assert run.error == "Test error message"
        assert run.status == "failed"

    async def test_get_state_raises_without_init(self, checkpoint):
        """Test that get_state raises error without initialization."""
        with pytest.raises(RuntimeError, match="No active state"):
            checkpoint.get_state()

    async def test_run_id_property(self, checkpoint):
        """Test run_id property."""
        assert checkpoint.run_id is None
        await checkpoint.start_new_run()
        assert checkpoint.run_id == datetime.now().strftime("%Y%m%d")

    async def test_full_pipeline_lifecycle(self, checkpoint, checkpoint_db):
        """Test a complete pipeline run persisted to DB."""
        await checkpoint.start_new_run()

        await checkpoint.mark_topic_done("fintwit")
        await checkpoint.mark_topic_done("markets")
        await checkpoint.complete_step1()

        await checkpoint.save_trends(["$NVDA", "Silver"])

        await checkpoint.mark_trend_done("$NVDA")
        await checkpoint.mark_trend_done("Silver")
        await checkpoint.complete_step3()

        await checkpoint.save_analysis(
            analysis="Test analysis",
            signal_strength="medium",
            is_notable=False,
            top_engagement=12000.0,
        )

        await checkpoint.complete_step5()
        await checkpoint.complete_step6()
        await checkpoint.clear()

        # Verify final DB state
        run_id = datetime.now().strftime("%Y%m%d")
        session = checkpoint_db()
        run = await session.get(PipelineRun, run_id)
        await session.close()

        assert run.status == "completed"
        assert run.step1_complete is True
        assert run.step6_complete is True
        assert run.topics_completed == ["fintwit", "markets"]
        assert run.trends == ["$NVDA", "Silver"]

    async def test_start_new_run_overwrites_failed(self, checkpoint, checkpoint_db):
        """Test that starting a new run overwrites a failed run from same day."""
        await checkpoint.start_new_run()
        await checkpoint.complete_step1()
        await checkpoint.set_error("Something broke")

        # Start fresh — should overwrite
        await checkpoint.start_new_run()
        state = checkpoint.get_state()

        assert state.step1_complete is False
        assert state.error == ""
