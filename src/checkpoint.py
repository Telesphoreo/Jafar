"""
Checkpoint System for Pipeline State Persistence.

Saves progress after each major step so the pipeline can:
- Resume after interruption (Ctrl+C, crash, rate limits)
- Skip already-completed steps
- Preserve collected tweets across runs
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .scraper import ScrapedTweet

logger = logging.getLogger("jafar.checkpoint")

CHECKPOINT_FILE = "pipeline_checkpoint.json"


@dataclass
class PipelineState:
    """Represents the current state of a pipeline run."""

    # Run identification
    run_id: str  # Date-based ID
    started_at: str

    # Step completion flags
    step1_complete: bool = False  # Broad scraping
    step2_complete: bool = False  # Trend analysis
    step3_complete: bool = False  # Deep dive scraping
    step4_complete: bool = False  # LLM analysis
    step5_complete: bool = False  # Email sent
    step6_complete: bool = False  # History stored

    # Step 1: Broad scraping progress
    topics_completed: list[str] = field(default_factory=list)
    topics_remaining: list[str] = field(default_factory=list)
    broad_tweets: list[dict] = field(default_factory=list)  # Serialized tweets

    # Step 2: Discovered trends
    trends: list[str] = field(default_factory=list)

    # Step 3: Deep dive tweets
    trend_tweets: dict[str, list[dict]] = field(default_factory=dict)

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
        logger.info(f"CheckpointManager initialized: {checkpoint_file}")

    def _serialize_tweet(self, tweet: ScrapedTweet) -> dict:
        """Convert a ScrapedTweet to a JSON-serializable dict."""
        return {
            "id": tweet.id,
            "text": tweet.text,
            "username": tweet.username,
            "display_name": tweet.display_name,
            "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
            "likes": tweet.likes,
            "retweets": tweet.retweets,
            "replies": tweet.replies,
            "views": tweet.views,
            "language": tweet.language,
            "hashtags": tweet.hashtags,
            "is_retweet": tweet.is_retweet,
        }

    def _deserialize_tweet(self, data: dict) -> ScrapedTweet:
        """Convert a dict back to a ScrapedTweet."""
        return ScrapedTweet(
            id=data["id"],
            text=data["text"],
            username=data["username"],
            display_name=data.get("display_name", "Unknown"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            likes=data["likes"],
            retweets=data["retweets"],
            replies=data["replies"],
            views=data.get("views"),
            language=data.get("language"),
            hashtags=data.get("hashtags", []),
            is_retweet=data["is_retweet"],
        )

    def start_new_run(self, topics: list[str]) -> PipelineState:
        """Start a fresh pipeline run."""
        today = datetime.now().strftime("%Y%m%d")

        self._state = PipelineState(
            run_id=today,
            started_at=datetime.now().isoformat(),
            topics_remaining=topics.copy(),
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
    def mark_topic_complete(self, topic: str, tweets: list[ScrapedTweet]) -> None:
        """Mark a topic as scraped and save its tweets."""
        state = self.get_state()

        if topic in state.topics_remaining:
            state.topics_remaining.remove(topic)
        if topic not in state.topics_completed:
            state.topics_completed.append(topic)

        # Add tweets to collection
        for tweet in tweets:
            state.broad_tweets.append(self._serialize_tweet(tweet))

        self.save()
        logger.info(f"Topic complete: {topic} ({len(tweets)} tweets)")

    def get_broad_tweets(self) -> list[ScrapedTweet]:
        """Get all collected broad tweets."""
        state = self.get_state()
        return [self._deserialize_tweet(t) for t in state.broad_tweets]

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
    def mark_trend_scraped(self, trend: str, tweets: list[ScrapedTweet]) -> None:
        """Save tweets for a specific trend."""
        state = self.get_state()
        state.trend_tweets[trend] = [self._serialize_tweet(t) for t in tweets]
        self.save()
        logger.info(f"Trend scraped: {trend} ({len(tweets)} tweets)")

    def get_trend_tweets(self) -> dict[str, list[ScrapedTweet]]:
        """Get all trend tweets."""
        state = self.get_state()
        return {
            trend: [self._deserialize_tweet(t) for t in tweets]
            for trend, tweets in state.trend_tweets.items()
        }

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
