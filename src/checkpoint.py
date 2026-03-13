"""
Checkpoint System for Pipeline State Persistence.

Saves progress after each major step so the pipeline can:
- Resume after interruption (Ctrl+C, crash, rate limits)
- Skip already-completed steps
- Track which topics/trends have been scraped (tweets live in the DB)
"""

import json
import logging
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("jafar.checkpoint")

# Runtime directory for checkpoint files
RUN_DIR = Path(".run")
CHECKPOINT_FILE = str(RUN_DIR / "pipeline_checkpoint.json")


SCHEMA_VERSION = 2  # Bump when step semantics change


@dataclass
class PipelineState:
    """Represents the current state of a pipeline run."""

    # Run identification
    run_id: str  # Date-based ID
    started_at: str
    schema_version: int = SCHEMA_VERSION

    # Step completion flags
    step1_complete: bool = False  # Broad scraping
    step2_complete: bool = False  # Trend analysis
    step3_complete: bool = False  # Deep dive scraping
    step4_complete: bool = False  # LLM analysis
    step5_complete: bool = False  # Email sent
    step6_complete: bool = False  # History stored

    # Step 1: Which topics have been scraped (tweets are in DB)
    topics_completed: list[str] = field(default_factory=list)

    # Step 2: Discovered trends
    trends: list[str] = field(default_factory=list)

    # Step 3: Which trends have been scraped (tweets are in DB)
    trends_completed: list[str] = field(default_factory=list)

    # Step 4: Analysis results
    analysis: str = ""
    signal_strength: str = ""
    is_notable: bool = False
    top_engagement: float = 0.0

    # Metadata
    last_updated: str = ""
    error: str = ""


class CheckpointManager:
    """Manages saving and loading pipeline state."""

    def __init__(self, checkpoint_file: str = CHECKPOINT_FILE):
        self.checkpoint_file = Path(checkpoint_file)
        self._state: Optional[PipelineState] = None

        # Ensure .run directory exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"CheckpointManager initialized: {checkpoint_file}")

    def _migrate_state(self, data: dict) -> dict | None:
        """Strip unknown fields from old checkpoint formats.

        Returns None if the checkpoint is from an incompatible schema version.
        """
        if data.get("schema_version", 0) < SCHEMA_VERSION:
            logger.warning(
                f"Checkpoint schema version {data.get('schema_version', 0)} "
                f"is older than current ({SCHEMA_VERSION}), discarding"
            )
            return None
        known = {f.name for f in fields(PipelineState)}
        return {k: v for k, v in data.items() if k in known}

    def start_new_run(self) -> PipelineState:
        """Start a fresh pipeline run."""
        today = datetime.now().strftime("%Y%m%d")

        self._state = PipelineState(
            run_id=today,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )

        self.save()
        logger.info(f"Started new pipeline run: {today}")
        return self._state

    def load(self) -> Optional[PipelineState]:
        """Load existing checkpoint if available."""
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found")
            return None

        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)

            data = self._migrate_state(data)
            if data is None:
                return None
            self._state = PipelineState(**data)
            logger.info(f"Loaded checkpoint from run: {self._state.run_id}")
            return self._state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def save(self) -> None:
        """Save current state to checkpoint file."""
        if self._state is None:
            return

        self._state.last_updated = datetime.now().isoformat()

        with open(self.checkpoint_file, "w") as f:
            json.dump(asdict(self._state), f, indent=2)

        logger.debug("Checkpoint saved")

    def should_resume(self) -> bool:
        """Check if there's a valid checkpoint to resume from."""
        state = self.load()
        if state is None:
            return False

        # Check if it's from today and not complete
        today = datetime.now().strftime("%Y%m%d")
        if state.run_id != today:
            logger.info(f"Checkpoint is from {state.run_id}, starting fresh")
            return False

        if state.step6_complete:
            logger.info("Previous run completed successfully, starting fresh")
            return False

        return True

    def get_state(self) -> PipelineState:
        """Get current state."""
        if self._state is None:
            raise RuntimeError("No active state. Call start_new_run() or load() first.")
        return self._state

    # Step 1: Broad scraping
    def mark_topic_done(self, topic: str) -> None:
        """Mark a topic as scraped (tweets are stored in DB)."""
        state = self.get_state()
        if topic not in state.topics_completed:
            state.topics_completed.append(topic)
        self.save()
        logger.info(f"Topic complete: {topic}")

    def complete_step1(self) -> None:
        """Mark step 1 as complete."""
        state = self.get_state()
        state.step1_complete = True
        self.save()
        logger.info("Step 1 (broad scraping) complete")

    # Step 2: Trend analysis
    def save_trends(self, trends: list[str]) -> None:
        """Save discovered trends."""
        state = self.get_state()
        state.trends = trends
        state.step2_complete = True
        self.save()
        logger.info(f"Step 2 complete: {len(trends)} trends discovered")

    # Step 3: Deep dive
    def mark_trend_done(self, trend: str) -> None:
        """Mark a trend as scraped (tweets are stored in DB)."""
        state = self.get_state()
        if trend not in state.trends_completed:
            state.trends_completed.append(trend)
        self.save()
        logger.info(f"Trend complete: {trend}")

    def complete_step3(self) -> None:
        """Mark step 3 as complete."""
        state = self.get_state()
        state.step3_complete = True
        self.save()
        logger.info("Step 3 (deep dive) complete")

    # Step 4: Analysis
    def save_analysis(
        self,
        analysis: str,
        signal_strength: str,
        is_notable: bool,
        top_engagement: float,
    ) -> None:
        """Save LLM analysis results."""
        state = self.get_state()
        state.analysis = analysis
        state.signal_strength = signal_strength
        state.is_notable = is_notable
        state.top_engagement = top_engagement
        state.step4_complete = True
        self.save()
        logger.info("Step 4 (analysis) complete")

    # Steps 5 & 6
    def complete_step5(self) -> None:
        """Mark email sent."""
        state = self.get_state()
        state.step5_complete = True
        self.save()
        logger.info("Step 5 (email) complete")

    def complete_step6(self) -> None:
        """Mark history stored."""
        state = self.get_state()
        state.step6_complete = True
        self.save()
        logger.info("Step 6 (history) complete - pipeline finished!")

    def clear(self) -> None:
        """Clear the checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self._state = None
        logger.info("Checkpoint cleared")

    def set_error(self, error: str) -> None:
        """Record an error in the checkpoint."""
        if self._state:
            self._state.error = error
            self.save()
