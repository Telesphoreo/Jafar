"""
Admin Diagnostics Module.

Tracks run statistics, detects errors, and generates diagnostic reports
for system monitoring and alerting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("jafar.diagnostics")


@dataclass
class RunDiagnostics:
    """Complete diagnostic information for a pipeline run."""

    # Run metadata
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Scraping statistics
    broad_topics_attempted: int = 0
    broad_topics_completed: int = 0
    broad_tweets_scraped: int = 0
    trends_discovered: int = 0
    trends_filtered_by_llm: int = 0
    deep_dive_trends_attempted: int = 0
    deep_dive_trends_completed: int = 0
    deep_dive_tweets_scraped: int = 0

    # Twitter account health
    twitter_accounts_total: int = 0
    twitter_accounts_active: int = 0
    twitter_accounts_rate_limited: int = 0

    # Analysis statistics
    llm_calls_made: int = 0
    llm_tokens_used: int = 0
    fact_checks_performed: int = 0
    temporal_patterns_detected: int = 0
    vector_memories_searched: int = 0
    vector_memories_stored: int = 0

    # Performance metrics (seconds)
    time_step1_scraping: float = 0.0
    time_step2_analysis: float = 0.0
    time_step3_deep_dive: float = 0.0
    time_step4_llm: float = 0.0
    time_step5_email: float = 0.0
    time_step6_storage: float = 0.0

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Final results
    signal_strength: str = "unknown"
    notable: bool = False
    email_sent: bool = False
    admin_email_sent: bool = False

    @property
    def duration_seconds(self) -> float:
        """Total run duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def duration_formatted(self) -> str:
        """Human-readable duration."""
        seconds = self.duration_seconds
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @property
    def total_tweets(self) -> int:
        """Total tweets scraped (broad + deep dive)."""
        return self.broad_tweets_scraped + self.deep_dive_tweets_scraped

    @property
    def has_critical_errors(self) -> bool:
        """Check if run encountered critical errors."""
        critical_conditions = [
            self.total_tweets == 0,  # No tweets scraped at all
            self.twitter_accounts_active == 0 and self.twitter_accounts_total > 0,  # All accounts dead
            len(self.errors) > 0 and not self.email_sent,  # Errors but no email sent
        ]
        return any(critical_conditions)

    @property
    def has_warnings(self) -> bool:
        """Check if run has warning conditions."""
        warning_conditions = [
            self.total_tweets < 100,  # Very low tweet count
            self.trends_discovered == 0,  # No trends found
            self.twitter_accounts_active < self.twitter_accounts_total * 0.5,  # >50% accounts unavailable
            len(self.warnings) > 0,
        ]
        return any(warning_conditions)

    def add_error(self, error: str) -> None:
        """Record an error during the run."""
        self.errors.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error}")
        logger.error(f"Diagnostic error recorded: {error}")

    def add_warning(self, warning: str) -> None:
        """Record a warning during the run."""
        self.warnings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {warning}")
        logger.warning(f"Diagnostic warning recorded: {warning}")

    def format_summary(self) -> str:
        """Generate a text summary for logging."""
        lines = [
            "=" * 60,
            "RUN DIAGNOSTICS SUMMARY",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Duration: {self.duration_formatted}",
            f"Status: {'✓ SUCCESS' if not self.has_critical_errors else '✗ FAILED'}",
            "",
            "SCRAPING STATS:",
            f"  Broad topics: {self.broad_topics_completed}/{self.broad_topics_attempted}",
            f"  Broad tweets: {self.broad_tweets_scraped}",
            f"  Trends discovered: {self.trends_discovered}",
            f"  Trends after LLM filter: {self.trends_filtered_by_llm}",
            f"  Deep dive trends: {self.deep_dive_trends_completed}/{self.deep_dive_trends_attempted}",
            f"  Deep dive tweets: {self.deep_dive_tweets_scraped}",
            f"  Total tweets: {self.total_tweets}",
            "",
            "TWITTER ACCOUNTS:",
            f"  Active: {self.twitter_accounts_active}/{self.twitter_accounts_total}",
            f"  Rate limited: {self.twitter_accounts_rate_limited}",
            "",
            "ANALYSIS:",
            f"  Signal strength: {self.signal_strength.upper()}",
            f"  Notable: {self.notable}",
            f"  LLM calls: {self.llm_calls_made} ({self.llm_tokens_used} tokens)",
            f"  Fact checks: {self.fact_checks_performed}",
            f"  Temporal patterns: {self.temporal_patterns_detected}",
            "",
            "PERFORMANCE:",
            f"  Step 1 (broad scraping): {self.time_step1_scraping:.1f}s",
            f"  Step 2 (trend analysis): {self.time_step2_analysis:.1f}s",
            f"  Step 3 (deep dive): {self.time_step3_deep_dive:.1f}s",
            f"  Step 4 (LLM analysis): {self.time_step4_llm:.1f}s",
            f"  Step 5 (email): {self.time_step5_email:.1f}s",
            f"  Step 6 (storage): {self.time_step6_storage:.1f}s",
        ]

        if self.errors:
            lines.extend([
                "",
                "ERRORS:",
            ])
            lines.extend(f"  • {e}" for e in self.errors)

        if self.warnings:
            lines.extend([
                "",
                "WARNINGS:",
            ])
            lines.extend(f"  • {w}" for w in self.warnings)

        lines.append("=" * 60)
        return "\n".join(lines)


class DiagnosticsCollector:
    """
    Collects diagnostic information throughout the pipeline run.

    This is used by main.py to track statistics and generate admin reports.
    """

    def __init__(self, run_id: str):
        """Initialize diagnostics collector for a new run."""
        self.diagnostics = RunDiagnostics(
            run_id=run_id,
            start_time=datetime.now(),
        )
        logger.info(f"Diagnostics collector initialized for run {run_id}")

    def finalize(self) -> RunDiagnostics:
        """Mark run as complete and return final diagnostics."""
        self.diagnostics.end_time = datetime.now()
        logger.info(f"Run completed in {self.diagnostics.duration_formatted}")

        # Log summary
        summary = self.diagnostics.format_summary()
        for line in summary.split('\n'):
            logger.info(line)

        return self.diagnostics


def rotate_logs(log_file: str = "pipeline.log", keep_count: int = 10) -> None:
    """
    Rotate log files by renaming current log with timestamp.

    Args:
        log_file: Path to the current log file
        keep_count: Number of old logs to keep (default: 10)
    """
    log_path = Path(log_file)

    # Only rotate if log file exists and has content
    if not log_path.exists() or log_path.stat().st_size == 0:
        logger.debug(f"No log file to rotate: {log_file}")
        return

    # Create archive filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{log_path.stem}_{timestamp}.log"
    archive_path = log_path.parent / archive_name

    try:
        # Rename current log
        log_path.rename(archive_path)
        logger.info(f"Rotated log: {log_file} -> {archive_name}")

        # Clean up old logs (keep only N most recent)
        log_dir = log_path.parent
        pattern = f"{log_path.stem}_*.log"
        old_logs = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if len(old_logs) > keep_count:
            for old_log in old_logs[keep_count:]:
                old_log.unlink()
                logger.info(f"Cleaned up old log: {old_log.name}")

    except Exception as e:
        logger.warning(f"Failed to rotate log {log_file}: {e}")


def should_send_admin_alert(diagnostics: RunDiagnostics) -> tuple[bool, str]:
    """
    Determine if admin should be alerted based on diagnostics.

    Returns:
        Tuple of (should_alert, reason)
    """
    # Critical errors always trigger alert
    if diagnostics.has_critical_errors:
        if diagnostics.total_tweets == 0:
            return True, "CRITICAL: Zero tweets scraped"
        elif diagnostics.twitter_accounts_active == 0:
            return True, "CRITICAL: All Twitter accounts unavailable"
        elif len(diagnostics.errors) > 0:
            return True, f"CRITICAL: {len(diagnostics.errors)} error(s) occurred"

    # Check for memory storage failures (important infrastructure issue)
    memory_failures = [w for w in diagnostics.warnings if "Failed to store memory" in w]
    if memory_failures:
        return True, f"WARNING: Memory storage failed - {len(memory_failures)} failure(s)"

    # Warning conditions trigger alert if severe enough
    if diagnostics.total_tweets < 50:
        return True, f"WARNING: Very low tweet count ({diagnostics.total_tweets})"

    if diagnostics.trends_discovered == 0:
        return True, "WARNING: No trends discovered"

    if diagnostics.twitter_accounts_active < diagnostics.twitter_accounts_total * 0.3:
        return True, f"WARNING: Only {diagnostics.twitter_accounts_active}/{diagnostics.twitter_accounts_total} accounts active"

    # No alert needed - everything looks good
    return False, "All systems operational"
