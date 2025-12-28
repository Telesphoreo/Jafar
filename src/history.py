"""
Historical Digest Storage.

Stores past digests so the LLM can compare today's findings against
historical context. This enables:
- "Silver spiked last week too, but today is 10x the engagement"
- "This is the first time trucks have appeared in 30 days"
- "Today looks like a normal day, nothing unusual"
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("jafar.history")


@dataclass
class HistoricalDigest:
    """A stored digest from a previous run."""
    id: int
    run_date: datetime
    trends: list[str]
    tweet_count: int
    digest_text: str
    signal_strength: str  # 'high', 'medium', 'low', 'none'
    top_engagement: float
    notable: bool  # Was this day flagged as notable?


class DigestHistory:
    """
    SQLite-backed storage for historical digests.

    Provides context for the LLM to make better judgments about
    whether today's trends are actually significant.
    """

    def __init__(self, db_path: str = "digest_history.db"):
        self.db_path = db_path
        self._init_db()
        logger.info(f"DigestHistory initialized with database: {db_path}")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS digests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TIMESTAMP NOT NULL,
                    trends TEXT NOT NULL,
                    tweet_count INTEGER NOT NULL,
                    digest_text TEXT NOT NULL,
                    signal_strength TEXT NOT NULL,
                    top_engagement REAL NOT NULL,
                    notable INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for date-based queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_digests_run_date
                ON digests(run_date)
            """)

            # Table for tracking individual trend appearances over time
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trend_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trend_term TEXT NOT NULL,
                    run_date TIMESTAMP NOT NULL,
                    mention_count INTEGER NOT NULL,
                    engagement_score REAL NOT NULL,
                    UNIQUE(trend_term, run_date)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trend_history_term
                ON trend_history(trend_term)
            """)

            conn.commit()

    def store_digest(
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO digests
                (run_date, trends, tweet_count, digest_text, signal_strength, top_engagement, notable)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    json.dumps(trends),
                    tweet_count,
                    digest_text,
                    signal_strength,
                    top_engagement,
                    1 if notable else 0,
                )
            )
            digest_id = cursor.lastrowid

            # Store individual trend history if provided
            if trend_details:
                for term, details in trend_details.items():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO trend_history
                        (trend_term, run_date, mention_count, engagement_score)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            term.lower(),
                            datetime.now().date().isoformat(),
                            details.get('mentions', 0),
                            details.get('engagement', 0),
                        )
                    )

            conn.commit()
            logger.info(f"Stored digest #{digest_id} with {len(trends)} trends")
            return digest_id

    def get_recent_digests(self, days: int = 7) -> list[HistoricalDigest]:
        """Get digests from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM digests
                WHERE run_date >= ?
                ORDER BY run_date DESC
                """,
                (cutoff.isoformat(),)
            ).fetchall()

        return [
            HistoricalDigest(
                id=row['id'],
                run_date=datetime.fromisoformat(row['run_date']),
                trends=json.loads(row['trends']),
                tweet_count=row['tweet_count'],
                digest_text=row['digest_text'],
                signal_strength=row['signal_strength'],
                top_engagement=row['top_engagement'],
                notable=bool(row['notable']),
            )
            for row in rows
        ]

    def get_trend_history(self, trend_term: str, days: int = 30) -> list[dict]:
        """
        Get historical data for a specific trend.

        Returns list of {date, mentions, engagement} for when this trend appeared.
        """
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT run_date, mention_count, engagement_score
                FROM trend_history
                WHERE trend_term = ? AND run_date >= ?
                ORDER BY run_date DESC
                """,
                (trend_term.lower(), cutoff.date().isoformat())
            ).fetchall()

        return [
            {
                'date': row['run_date'],
                'mentions': row['mention_count'],
                'engagement': row['engagement_score'],
            }
            for row in rows
        ]

    def get_baseline_stats(self, days: int = 30) -> dict:
        """
        Calculate baseline statistics for comparison.

        Returns average engagement, typical trend count, etc.
        """
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_runs,
                    AVG(top_engagement) as avg_top_engagement,
                    AVG(tweet_count) as avg_tweets,
                    SUM(notable) as notable_days
                FROM digests
                WHERE run_date >= ?
                """,
                (cutoff.isoformat(),)
            ).fetchone()

        if not row or row[0] == 0:
            return {
                'total_runs': 0,
                'avg_top_engagement': 0,
                'avg_tweets': 0,
                'notable_days': 0,
                'notable_rate': 0,
            }

        return {
            'total_runs': row[0],
            'avg_top_engagement': row[1] or 0,
            'avg_tweets': row[2] or 0,
            'notable_days': row[3] or 0,
            'notable_rate': (row[3] or 0) / row[0] if row[0] > 0 else 0,
        }

    def format_context_for_llm(self, days: int = 7) -> str:
        """
        Format recent history as context for the LLM.

        This helps the LLM understand what's normal vs unusual.
        """
        recent = self.get_recent_digests(days)
        baseline = self.get_baseline_stats(30)

        if not recent:
            return """
## Historical Context
This is the first run - no historical data available yet.
After a few days of data, the system will be able to compare against baseline.
"""

        lines = [
            "## Historical Context (last 7 days)",
            f"Runs in database: {baseline['total_runs']}",
            f"Average top engagement: {baseline['avg_top_engagement']:.0f}",
            f"Days flagged as notable: {baseline['notable_days']} ({baseline['notable_rate']*100:.0f}%)",
            "",
            "### Recent Digests:",
        ]

        for digest in recent[:5]:  # Last 5 digests
            date_str = digest.run_date.strftime("%Y-%m-%d")
            notable_flag = " [NOTABLE]" if digest.notable else ""
            lines.append(
                f"- {date_str}: {', '.join(digest.trends[:3])} "
                f"(signal: {digest.signal_strength}, engagement: {digest.top_engagement:.0f}){notable_flag}"
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
