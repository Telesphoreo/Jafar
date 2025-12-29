"""
Temporal Trend Tracking Module.

Detects trend continuity patterns over time:
- Consecutive days trending (developing stories)
- Gaps in trend activity (recurring themes)
- Historical parallels (similar past events)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("jafar.temporal")


@dataclass
class TrendTimeline:
    """Timeline data for a specific trend."""
    term: str
    term_normalized: str  # Lowercase, no symbols for matching

    # Current run data
    first_seen_today: datetime
    last_seen_today: datetime
    mentions_today: int
    engagement_today: float

    # Historical continuity
    consecutive_days: int = 0  # How many days in a row (including today)
    total_appearances: int = 1  # Total times seen in history
    last_seen_date: Optional[datetime] = None  # When was it last seen before today?
    days_since_last: Optional[int] = None  # Gap since last appearance

    # Historical context
    previous_mentions: list[int] = field(default_factory=list)  # Mention counts on previous days
    previous_engagements: list[float] = field(default_factory=list)  # Engagement on previous days

    @property
    def is_new(self) -> bool:
        """First time ever seeing this trend."""
        return self.total_appearances == 1

    @property
    def is_continuing(self) -> bool:
        """Trend is continuing from recent days (no significant gap)."""
        return self.consecutive_days >= 2

    @property
    def is_recurring(self) -> bool:
        """Trend reappeared after a significant gap."""
        return (
            not self.is_new
            and self.days_since_last is not None
            and self.days_since_last >= 14  # 2 week gap = new episode
        )

    @property
    def temporal_badge(self) -> str:
        """Generate display badge for trend timeline."""
        if self.is_new:
            return "New"
        elif self.consecutive_days >= 3:
            return f"Day {self.consecutive_days}"
        elif self.is_recurring:
            if self.days_since_last >= 180:  # 6+ months
                months = self.days_since_last // 30
                return f"Last seen {months}mo ago"
            else:
                return f"Last seen {self.days_since_last}d ago"
        elif self.consecutive_days == 2:
            return "Day 2"
        else:
            return ""

    @property
    def trend_velocity(self) -> str:
        """Describe the trend trajectory (growing/stable/declining)."""
        if not self.previous_mentions:
            return "emerging"

        # Compare today to recent average
        recent_avg = sum(self.previous_mentions[:3]) / len(self.previous_mentions[:3])

        if self.mentions_today > recent_avg * 1.5:
            return "accelerating"
        elif self.mentions_today < recent_avg * 0.5:
            return "declining"
        else:
            return "stable"

    def format_for_llm(self) -> str:
        """Format timeline data for LLM context."""
        lines = []

        if self.is_new:
            lines.append(f"New trend: First appearance in database")
        elif self.is_continuing:
            lines.append(f"Developing story: Day {self.consecutive_days} of consecutive trending")
            lines.append(f"Velocity: {self.trend_velocity}")
            if self.previous_mentions:
                mention_history = ", ".join(str(m) for m in self.previous_mentions[:5])
                lines.append(f"Recent mention history: [{mention_history}]")
        elif self.is_recurring:
            lines.append(f"Recurring theme: Last seen {self.days_since_last} days ago")
            lines.append(f"Previous appearances: {self.total_appearances} times")
            if self.last_seen_date:
                lines.append(f"Last active: {self.last_seen_date.strftime('%Y-%m-%d')}")
        else:
            lines.append(f"Seen {self.total_appearances} times in history")

        return " | ".join(lines)


class TemporalTrendAnalyzer:
    """
    Analyzes trend timelines to detect developing stories and recurring themes.

    Uses digest_history.db to track when trends appeared and identify patterns.
    """

    def __init__(
        self,
        history_db,  # DigestHistory instance
        consecutive_threshold: int = 3,
        gap_threshold_days: int = 14,
    ):
        """
        Initialize temporal analyzer.

        Args:
            history_db: DigestHistory instance for querying past data
            consecutive_threshold: Days to flag as "developing story"
            gap_threshold_days: Gap to consider a "new episode" vs continuation
        """
        self.history = history_db
        self.consecutive_threshold = consecutive_threshold
        self.gap_threshold = gap_threshold_days
        logger.info(
            f"TemporalTrendAnalyzer initialized "
            f"(consecutive_threshold={consecutive_threshold}, gap_threshold={gap_threshold_days}d)"
        )

    def analyze_trend_timeline(
        self,
        term: str,
        mentions_today: int,
        engagement_today: float,
        first_seen: datetime,
        last_seen: datetime,
    ) -> TrendTimeline:
        """
        Analyze timeline for a single trend.

        Args:
            term: The trend term (e.g., "$NVDA", "Silver", "Risk")
            mentions_today: Mention count in current run
            engagement_today: Total engagement in current run
            first_seen: First tweet timestamp today
            last_seen: Last tweet timestamp today

        Returns:
            TrendTimeline with historical context
        """
        # Normalize term for matching (remove symbols, lowercase)
        normalized = term.lower().strip('$#')

        # Query historical appearances
        history = self.history.get_trend_history(normalized, days=180)  # 6 months lookback

        timeline = TrendTimeline(
            term=term,
            term_normalized=normalized,
            first_seen_today=first_seen,
            last_seen_today=last_seen,
            mentions_today=mentions_today,
            engagement_today=engagement_today,
            total_appearances=len(history) + 1,  # +1 for today
        )

        if not history:
            # Brand new trend
            logger.debug(f"New trend detected: {term}")
            return timeline

        # Sort by date (most recent first)
        history.sort(key=lambda x: x['date'], reverse=True)

        # Detect consecutive days
        consecutive = 0
        today_date = datetime.now().date()
        check_date = today_date - timedelta(days=1)  # Start with yesterday

        for record in history:
            record_date = datetime.fromisoformat(record['date']).date()

            if record_date == check_date:
                consecutive += 1
                check_date -= timedelta(days=1)
            else:
                break  # Gap found, stop counting

        timeline.consecutive_days = consecutive + 1  # +1 for today

        # Find last appearance
        most_recent = history[0]
        most_recent_date = datetime.fromisoformat(most_recent['date'])
        timeline.last_seen_date = most_recent_date
        timeline.days_since_last = (today_date - most_recent_date.date()).days

        # Collect previous metrics for trajectory analysis
        timeline.previous_mentions = [h['mentions'] for h in history[:7]]  # Last week
        timeline.previous_engagements = [h['engagement'] for h in history[:7]]

        logger.debug(
            f"Timeline for {term}: consecutive={timeline.consecutive_days}, "
            f"gap={timeline.days_since_last}d, appearances={timeline.total_appearances}"
        )

        return timeline

    def analyze_all_trends(
        self,
        trend_details: dict[str, dict],
    ) -> dict[str, TrendTimeline]:
        """
        Analyze timelines for all discovered trends.

        Args:
            trend_details: Dict of {term: {mentions, engagement, first_seen, last_seen}}

        Returns:
            Dict of {term: TrendTimeline}
        """
        logger.info(f"Analyzing temporal patterns for {len(trend_details)} trends...")

        timelines = {}
        stats = {
            'new': 0,
            'continuing': 0,
            'recurring': 0,
        }

        for term, details in trend_details.items():
            timeline = self.analyze_trend_timeline(
                term=term,
                mentions_today=details['mentions'],
                engagement_today=details['engagement'],
                first_seen=details['first_seen'],
                last_seen=details['last_seen'],
            )
            timelines[term] = timeline

            # Track stats
            if timeline.is_new:
                stats['new'] += 1
            elif timeline.is_continuing:
                stats['continuing'] += 1
            elif timeline.is_recurring:
                stats['recurring'] += 1

        logger.info(
            f"Temporal analysis complete: "
            f"{stats['new']} new, "
            f"{stats['continuing']} continuing, "
            f"{stats['recurring']} recurring"
        )

        # Log notable timelines
        notable = [
            t for t in timelines.values()
            if t.consecutive_days >= self.consecutive_threshold or t.is_recurring
        ]
        if notable:
            logger.info("Notable temporal patterns:")
            for t in notable:
                logger.info(f"  {t.term}: {t.temporal_badge}")

        return timelines

    def format_context_for_llm(self, timelines: dict[str, TrendTimeline]) -> str:
        """
        Format temporal context as a string for LLM analysis.

        This gives the LLM insight into whether trends are developing or recurring.
        """
        if not timelines:
            return ""

        lines = ["## Temporal Context\n"]

        # Group by type
        new_trends = [t for t in timelines.values() if t.is_new]
        continuing = [t for t in timelines.values() if t.is_continuing]
        recurring = [t for t in timelines.values() if t.is_recurring]

        if continuing:
            lines.append("**Developing Stories (multi-day trends):**")
            for t in sorted(continuing, key=lambda x: x.consecutive_days, reverse=True):
                lines.append(f"- {t.term}: {t.format_for_llm()}")
            lines.append("")

        if recurring:
            lines.append("**Recurring Themes (gaps in activity):**")
            for t in sorted(recurring, key=lambda x: x.days_since_last or 0, reverse=True):
                lines.append(f"- {t.term}: {t.format_for_llm()}")
            lines.append("")

        if new_trends:
            lines.append(f"**New Trends:** {len(new_trends)} topics appearing for first time")
            lines.append("")

        return "\n".join(lines)
