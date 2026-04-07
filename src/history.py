# pylint: disable=not-callable
"""
Historical Digest Storage.

Stores past digests so the LLM can compare today's findings against
historical context. This enables:
- "Silver spiked last week too, but today is 10x the engagement"
- "This is the first time trucks have appeared in 30 days"
- "Today looks like a normal day, nothing unusual"
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from src.database import get_session
from src.models import Digest, TrendHistory

logger = logging.getLogger("jafar.history")


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types to serializable equivalents.

    Handles datetime, date, set, and other common types that json.dumps chokes on.
    Raises TypeError for truly unserializable objects rather than silently losing data.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_make_json_safe(v) for v in sorted(obj, key=str)]
    # Fail loud for types we don't know how to handle
    raise TypeError(f"Cannot serialize {type(obj).__name__} to JSON: {obj!r}")


class DigestHistory:
    """
    SQLAlchemy-backed storage for historical digests.

    Provides context for the LLM to make better judgments about
    whether today's trends are actually significant.
    """

    async def store_digest(
        self,
        trends: list[str],
        tweet_count: int,
        digest_text: str,
        signal_strength: str,
        top_engagement: float,
        notable: bool = False,
        trend_details: Optional[dict[str, dict]] = None,
    ) -> int:
        """
        Store a digest for future reference.

        Args:
            trends: List of trend terms discovered
            tweet_count: Total tweets analyzed
            digest_text: The full LLM-generated digest
            signal_strength: 'high', 'medium', 'low', or 'none'
            top_engagement: Highest engagement score seen
            notable: Whether this day was flagged as significant
            trend_details: Optional dict of {trend: {mentions, engagement}}

        Returns:
            The ID of the stored digest
        """
        # Sanitize JSON fields at the DB boundary — callers shouldn't
        # need to worry about datetime serialization
        safe_trends = _make_json_safe(trends)
        safe_trend_details = _make_json_safe(trend_details) if trend_details else None

        session = await get_session()
        async with session.begin():
            digest = Digest(
                date=date.today(),
                trends=safe_trends,
                tweet_count=tweet_count,
                digest_text=digest_text,
                signal_strength=signal_strength,
                top_engagement=top_engagement,
                notable=notable,
                trend_details=safe_trend_details,
            )
            session.add(digest)
            await session.flush()

            if trend_details:
                for term, details in trend_details.items():
                    mentions = details.get("mentions", 0)
                    engagement = details.get("engagement", 0)
                    stmt = insert(TrendHistory).values(
                        trend_term=term.lower(),
                        date=date.today(),
                        mentions=mentions,
                        engagement=engagement,
                    ).on_conflict_do_update(
                        index_elements=["trend_term", "date"],
                        set_={
                            "mentions": mentions,
                            "engagement": engagement,
                        },
                    )
                    await session.execute(stmt)

        logger.info(f"Stored digest #{digest.id} with {len(trends)} trends")
        return digest.id

    async def get_recent_digests(self, days: int = 7) -> list[dict]:
        """Get digests from the last N days."""
        cutoff = date.today() - timedelta(days=days)

        session = await get_session()
        async with session.begin():
            result = await session.execute(
                select(Digest)
                .where(Digest.date >= cutoff)
                .order_by(Digest.date.desc())
            )
            rows = result.scalars().all()

        return [
            {
                "date": row.date,
                "trends": row.trends or [],
                "tweet_count": row.tweet_count,
                "signal_strength": row.signal_strength,
                "top_engagement": row.top_engagement,
                "notable": row.notable,
                "trend_details": row.trend_details,
            }
            for row in rows
        ]

    async def get_trend_history(self, trend_term: str, days: int = 30) -> list[dict]:
        """
        Get historical data for a specific trend.

        Returns list of {date, mentions, engagement} for when this trend appeared.
        """
        cutoff = date.today() - timedelta(days=days)

        session = await get_session()
        async with session.begin():
            result = await session.execute(
                select(TrendHistory)
                .where(
                    TrendHistory.trend_term == trend_term.lower(),
                    TrendHistory.date >= cutoff,
                )
                .order_by(TrendHistory.date.desc())
            )
            rows = result.scalars().all()

        return [
            {
                "date": row.date,
                "mentions": row.mentions,
                "engagement": row.engagement,
            }
            for row in rows
        ]

    async def get_all_recent_trends(self, days: int = 7) -> list[dict]:
        """
        Get all trends that appeared in the last N days, aggregated.

        Returns list of dicts with: trend_term, total_mentions, total_engagement, appearances
        """
        cutoff = date.today() - timedelta(days=days)

        session = await get_session()
        async with session.begin():
            result = await session.execute(
                select(
                    TrendHistory.trend_term,
                    func.sum(TrendHistory.mentions).label("total_mentions"),
                    func.sum(TrendHistory.engagement).label("total_engagement"),
                    func.count().label("appearances"),
                )
                .where(TrendHistory.date >= cutoff)
                .group_by(TrendHistory.trend_term)
            )
            rows = result.all()

        return [
            {
                "trend_term": row.trend_term,
                "total_mentions": row.total_mentions or 0,
                "total_engagement": row.total_engagement or 0,
                "appearances": row.appearances,
            }
            for row in rows
        ]

    async def get_baseline_stats(self, days: int = 30) -> dict:
        """
        Calculate baseline statistics for comparison.

        Returns average engagement, typical trend count, etc.
        """
        cutoff = date.today() - timedelta(days=days)

        session = await get_session()
        async with session.begin():
            result = await session.execute(
                select(
                    func.count().label("total_digests"),
                    func.avg(Digest.tweet_count).label("avg_tweet_count"),
                    func.avg(Digest.top_engagement).label("avg_engagement"),
                    func.count()
                    .filter(Digest.notable.is_(True))
                    .label("notable_count"),
                )
                .where(Digest.date >= cutoff)
            )
            row = result.one()

        total = row.total_digests or 0
        if total == 0:
            return {
                "avg_tweet_count": 0,
                "avg_engagement": 0,
                "total_digests": 0,
                "notable_count": 0,
            }

        return {
            "avg_tweet_count": row.avg_tweet_count or 0,
            "avg_engagement": row.avg_engagement or 0,
            "total_digests": total,
            "notable_count": row.notable_count or 0,
        }

    async def get_previous_digest_summary(self, days_back: int = 1) -> Optional[str]:
        """Get a summary from the most recent digest."""
        cutoff = date.today() - timedelta(days=days_back)

        session = await get_session()
        async with session.begin():
            result = await session.execute(
                select(Digest.digest_text)
                .where(Digest.date >= cutoff)
                .order_by(Digest.date.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()

        if not row:
            return None

        text = row
        # Try to extract the summary section
        for marker in ("## Summary", "**Summary:**", "Summary:"):
            if marker in text:
                start = text.index(marker)
                next_section = text.find("\n\n**", start + len(marker))
                if next_section > 0:
                    return text[start:next_section].strip()
                return text[start:start + 500].strip()

        # Fall back to the first 500 characters
        return text[:500].strip() if text else None

    async def format_context_for_llm(self, days: int = 7) -> str:
        """
        Format recent history as context for the LLM.

        This helps the LLM understand what's normal vs unusual.
        """
        recent = await self.get_recent_digests(days)
        baseline = await self.get_baseline_stats(30)

        if not recent:
            return """
## Historical Context
This is the first run - no historical data available yet.
After a few days of data, the system will be able to compare against baseline.
"""

        lines = [
            "## Historical Context (last 7 days)",
            f"Runs in database: {baseline['total_digests']}",
            f"Average top engagement: {baseline['avg_engagement']:.0f}",
            f"Notable digests: {baseline['notable_count']}",
            "",
            "### Recent Digests:",
        ]

        for digest in recent[:5]:
            date_str = digest["date"].isoformat()
            notable_flag = " [NOTABLE]" if digest["notable"] else ""
            trends = digest["trends"] or []
            lines.append(
                f"- {date_str}: {', '.join(trends[:3])} "
                f"(signal: {digest['signal_strength']}, "
                f"engagement: {digest['top_engagement']:.0f}){notable_flag}"
            )

        return "\n".join(lines)


def calculate_signal_strength(
    top_engagement: float,
    trend_count: int,
    baseline_engagement: float = 10000,
) -> str:
    """
    Determine if today's signal is strong, medium, low, or none.

    This helps calibrate expectations - most days should be 'low' or 'medium'.
    """
    if trend_count == 0:
        return "none"

    # Compare against baseline
    engagement_ratio = top_engagement / max(baseline_engagement, 1)

    if engagement_ratio > 5 and trend_count >= 3:
        return "high"  # Genuinely unusual day
    elif engagement_ratio > 2 or trend_count >= 5:
        return "medium"  # Something worth noting
    elif engagement_ratio > 0.5:
        return "low"  # Normal market chatter
    else:
        return "none"  # Below-average day
