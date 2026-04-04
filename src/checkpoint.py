"""
Database-Backed Checkpoint System for Pipeline State Persistence.

All pipeline state lives in PostgreSQL. Advisory locks prevent
concurrent runs. If the database is unreachable, the pipeline
exits immediately — there is no file-based fallback.
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from src.database import get_engine, get_session
from src.models import PipelineRun

logger = logging.getLogger("jafar.checkpoint")

# PostgreSQL advisory lock ID for pipeline exclusivity.
# 0x4A414641 = "JAFA" in ASCII = 1245793857
PIPELINE_LOCK_ID = 1245793857


@dataclass
class PipelineState:
    """In-memory representation of pipeline run state."""

    run_id: str
    started_at: str

    step1_complete: bool = False
    step2_complete: bool = False
    step3_complete: bool = False
    step4_complete: bool = False
    step5_complete: bool = False
    step6_complete: bool = False

    topics_completed: list[str] = field(default_factory=list)
    trends: list[str] = field(default_factory=list)
    trends_completed: list[str] = field(default_factory=list)

    analysis: str = ""
    signal_strength: str = ""
    is_notable: bool = False
    top_engagement: float = 0.0

    last_updated: str = ""
    error: str = ""
    diagnostics_json: dict = field(default_factory=dict)


class CheckpointManager:
    """Database-backed pipeline state manager with advisory locking.

    Usage:
        checkpoint = CheckpointManager()
        await checkpoint.initialize()     # acquires DB lock
        if await checkpoint.should_resume():
            state = checkpoint.get_state()
        else:
            state = await checkpoint.start_new_run()
        ...
        await checkpoint.close()          # releases lock
    """

    def __init__(self):
        self._state: Optional[PipelineState] = None
        self._lock_conn = None

    @property
    def run_id(self) -> Optional[str]:
        return self._state.run_id if self._state else None

    async def initialize(self) -> None:
        """Connect to DB and acquire advisory lock.

        Exits the process if:
        - Database engine is not initialized
        - Another pipeline instance holds the lock
        """
        try:
            engine = get_engine()
        except RuntimeError:
            logger.critical("Database not initialized — cannot start pipeline")
            sys.exit(1)

        # Hold a dedicated connection for the advisory lock.
        # The lock auto-releases if this connection drops (crash, OOM, etc.).
        self._lock_conn = await engine.connect()
        result = await self._lock_conn.execute(
            text("SELECT pg_try_advisory_lock(:lock_id)"),
            {"lock_id": PIPELINE_LOCK_ID},
        )
        locked = result.scalar()

        if not locked:
            await self._lock_conn.close()
            self._lock_conn = None
            logger.critical(
                "Another pipeline instance is already running (advisory lock held)"
            )
            sys.exit(1)

        logger.info("Pipeline advisory lock acquired")

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    async def should_resume(self) -> bool:
        """Check if there's a resumable run from today."""
        today = datetime.now().strftime("%Y%m%d")

        session = await get_session()
        try:
            run = await session.get(PipelineRun, today)
        finally:
            await session.close()

        if run is None:
            logger.info("No checkpoint found for today")
            return False

        if run.status == "completed":
            logger.info("Previous run completed successfully, starting fresh")
            return False

        # If the previous run still says "running" but we hold the advisory lock,
        # the old process must have died (SIGTERM/SIGKILL). Update status to reflect reality.
        if run.status == "running":
            logger.warning(
                f"Previous run {run.run_id} was still 'running' but lock was not held — "
                "marking as interrupted (previous process likely killed)"
            )
            session2 = await get_session()
            async with session2.begin():
                stale = await session2.get(PipelineRun, run.run_id)
                if stale:
                    stale.status = "interrupted"
                    stale.last_updated = datetime.now()
            await session2.close()
            run.status = "interrupted"

        self._state = self._to_state(run)
        logger.info(f"Found resumable checkpoint: {run.run_id} (status: {run.status})")
        return True

    async def start_new_run(self) -> PipelineState:
        """Start a fresh pipeline run, overwriting any prior state for today."""
        today = datetime.now().strftime("%Y%m%d")
        now = datetime.now()

        # If there's an existing "running" row for today, the previous process
        # must have died without cleaning up. Mark it as interrupted.
        session = await get_session()
        try:
            existing = await session.get(PipelineRun, today)
            if existing and existing.status == "running":
                logger.warning(
                    f"Existing run {today} was still 'running' — "
                    "marking as interrupted before starting new run"
                )
                async with session.begin():
                    existing.status = "interrupted"
                    existing.last_updated = now
        finally:
            await session.close()

        self._state = PipelineState(
            run_id=today,
            started_at=now.isoformat(),
            last_updated=now.isoformat(),
        )

        session = await get_session()
        async with session.begin():
            await session.merge(PipelineRun(
                run_id=today,
                started_at=now,
                status="running",
                step1_complete=False,
                step2_complete=False,
                step3_complete=False,
                step4_complete=False,
                step5_complete=False,
                step6_complete=False,
                topics_completed=[],
                trends=[],
                trends_completed=[],
                analysis="",
                signal_strength="",
                is_notable=False,
                top_engagement=0.0,
                error="",
            ))
        await session.close()

        logger.info(f"Started new pipeline run: {today}")
        return self._state

    def get_state(self) -> PipelineState:
        """Get current in-memory state."""
        if self._state is None:
            raise RuntimeError(
                "No active state. Call start_new_run() or should_resume() first."
            )
        return self._state

    # ------------------------------------------------------------------
    # Step completion methods
    # ------------------------------------------------------------------

    async def mark_topic_done(self, topic: str) -> None:
        """Mark a topic as scraped."""
        state = self.get_state()
        if topic not in state.topics_completed:
            state.topics_completed.append(topic)
        await self.save()
        logger.info(f"Topic complete: {topic}")

    async def complete_step1(self) -> None:
        """Mark step 1 (broad scraping) as complete."""
        self.get_state().step1_complete = True
        await self.save()
        logger.info("Step 1 (broad scraping) complete")

    async def save_trends(self, trends: list[str]) -> None:
        """Save discovered trends and mark step 2 complete."""
        state = self.get_state()
        state.trends = trends
        state.step2_complete = True
        await self.save()
        logger.info(f"Step 2 complete: {len(trends)} trends discovered")

    async def mark_trend_done(self, trend: str) -> None:
        """Mark a trend as scraped."""
        state = self.get_state()
        if trend not in state.trends_completed:
            state.trends_completed.append(trend)
        await self.save()
        logger.info(f"Trend complete: {trend}")

    async def complete_step3(self) -> None:
        """Mark step 3 (deep dive) as complete."""
        self.get_state().step3_complete = True
        await self.save()
        logger.info("Step 3 (deep dive) complete")

    async def save_analysis(
        self,
        analysis: str,
        signal_strength: str,
        is_notable: bool,
        top_engagement: float,
    ) -> None:
        """Save LLM analysis results and mark step 4 complete."""
        state = self.get_state()
        state.analysis = analysis
        state.signal_strength = signal_strength
        state.is_notable = is_notable
        state.top_engagement = top_engagement
        state.step4_complete = True
        await self.save()
        logger.info("Step 4 (analysis) complete")

    async def complete_step5(self) -> None:
        """Mark step 5 (email sent) as complete."""
        self.get_state().step5_complete = True
        await self.save()
        logger.info("Step 5 (email) complete")

    async def complete_step6(self) -> None:
        """Mark step 6 (history stored) as complete."""
        self.get_state().step6_complete = True
        await self.save()
        logger.info("Step 6 (history) complete — pipeline finished!")

    async def save_diagnostics(self, diagnostics) -> None:
        """Persist current diagnostics snapshot to the database."""
        from dataclasses import asdict
        state = self.get_state()
        d = asdict(diagnostics)
        # Convert datetimes to strings for JSON
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        state.diagnostics_json = d
        await self.save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save(self) -> None:
        """Persist current in-memory state to the database."""
        if self._state is None:
            return

        now = datetime.now()
        self._state.last_updated = now.isoformat()

        session = await get_session()
        async with session.begin():
            await session.merge(PipelineRun(
                run_id=self._state.run_id,
                started_at=(
                    self._state.started_at
                    if isinstance(self._state.started_at, datetime)
                    else datetime.fromisoformat(self._state.started_at)
                    if self._state.started_at
                    else datetime.now()
                ),
                status="running",
                step1_complete=self._state.step1_complete,
                step2_complete=self._state.step2_complete,
                step3_complete=self._state.step3_complete,
                step4_complete=self._state.step4_complete,
                step5_complete=self._state.step5_complete,
                step6_complete=self._state.step6_complete,
                topics_completed=self._state.topics_completed,
                trends=self._state.trends,
                trends_completed=self._state.trends_completed,
                analysis=self._state.analysis,
                signal_strength=self._state.signal_strength,
                is_notable=self._state.is_notable,
                top_engagement=self._state.top_engagement,
                error=self._state.error,
                diagnostics_json=self._state.diagnostics_json or None,
                last_updated=now,
            ))
        await session.close()

        logger.debug("Checkpoint saved to database")

    async def clear(self) -> None:
        """Mark the current run as completed in the database."""
        if self._state is None:
            return

        session = await get_session()
        async with session.begin():
            run = await session.get(PipelineRun, self._state.run_id)
            if run:
                run.status = "completed"
                run.last_updated = datetime.now()
        await session.close()

        self._state = None
        logger.info("Pipeline run marked as completed")

    async def set_error(self, error: str) -> None:
        """Record an error and mark the run as failed."""
        if self._state:
            self._state.error = error

            session = await get_session()
            async with session.begin():
                run = await session.get(PipelineRun, self._state.run_id)
                if run:
                    run.error = error
                    run.status = "failed"
                    run.last_updated = datetime.now()
            await session.close()

    async def close(self) -> None:
        """Mark incomplete runs as interrupted, then release the advisory lock."""
        # If the run is still "running" at close time, it didn't complete or
        # explicitly fail -- mark it as interrupted so the dashboard is accurate.
        if self._state:
            session = await get_session()
            try:
                async with session.begin():
                    run = await session.get(PipelineRun, self._state.run_id)
                    if run and run.status == "running":
                        run.status = "interrupted"
                        run.last_updated = datetime.now()
                        logger.info(
                            f"Run {self._state.run_id} still 'running' at close — "
                            "marked as interrupted"
                        )
            except Exception as e:
                logger.warning(f"Could not update run status on close: {e}")
            finally:
                await session.close()

        if self._lock_conn:
            await self._lock_conn.close()
            self._lock_conn = None
            logger.info("Pipeline advisory lock released")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_state(self, row: PipelineRun) -> PipelineState:
        """Convert an ORM PipelineRun to an in-memory PipelineState."""
        return PipelineState(
            run_id=row.run_id,
            started_at=row.started_at.isoformat() if row.started_at else "",
            step1_complete=row.step1_complete,
            step2_complete=row.step2_complete,
            step3_complete=row.step3_complete,
            step4_complete=row.step4_complete,
            step5_complete=row.step5_complete,
            step6_complete=row.step6_complete,
            topics_completed=row.topics_completed or [],
            trends=row.trends or [],
            trends_completed=row.trends_completed or [],
            analysis=row.analysis or "",
            signal_strength=row.signal_strength or "",
            is_notable=row.is_notable,
            top_engagement=row.top_engagement or 0.0,
            last_updated=(
                row.last_updated.isoformat() if row.last_updated else ""
            ),
            error=row.error or "",
            diagnostics_json=row.diagnostics_json or {},
        )
