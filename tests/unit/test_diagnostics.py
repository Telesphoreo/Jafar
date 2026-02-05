"""
Unit tests for src/diagnostics.py

Tests diagnostics collection, log rotation, and admin alert determination.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.diagnostics import (
    DiagnosticsCollector,
    RunDiagnostics,
    rotate_logs,
    should_send_admin_alert,
)


class TestRunDiagnostics:
    """Tests for RunDiagnostics dataclass."""

    def test_default_values(self):
        """Test RunDiagnostics default values."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
        )

        assert diag.broad_topics_attempted == 0
        assert diag.broad_tweets_scraped == 0
        assert diag.trends_discovered == 0
        assert diag.llm_calls_made == 0
        assert diag.errors == []
        assert diag.warnings == []

    def test_duration_seconds(self):
        """Test duration calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=120)

        diag = RunDiagnostics(
            run_id="20240101",
            start_time=start,
            end_time=end,
        )

        assert diag.duration_seconds == 120.0

    def test_duration_seconds_ongoing(self):
        """Test duration calculation for ongoing run."""
        start = datetime.now() - timedelta(seconds=60)

        diag = RunDiagnostics(
            run_id="20240101",
            start_time=start,
            end_time=None,
        )

        # Should be approximately 60 seconds
        assert 59 <= diag.duration_seconds <= 61

    def test_duration_formatted_seconds(self):
        """Test formatted duration for short runs."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=45),
        )

        assert "s" in diag.duration_formatted
        assert "45" in diag.duration_formatted

    def test_duration_formatted_minutes(self):
        """Test formatted duration for minute-long runs."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=5, seconds=30),
        )

        assert "m" in diag.duration_formatted

    def test_duration_formatted_hours(self):
        """Test formatted duration for hour-long runs."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2),
        )

        assert "h" in diag.duration_formatted

    def test_total_tweets(self):
        """Test total tweet count calculation."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=1500,
            deep_dive_tweets_scraped=500,
        )

        assert diag.total_tweets == 2000

    def test_has_critical_errors_zero_tweets(self):
        """Test critical error detection for zero tweets."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=0,
            deep_dive_tweets_scraped=0,
        )

        assert diag.has_critical_errors is True

    def test_has_critical_errors_all_accounts_dead(self):
        """Test critical error detection when all accounts unavailable."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            twitter_accounts_total=5,
            twitter_accounts_active=0,
            broad_tweets_scraped=100,  # Some tweets before accounts died
        )

        assert diag.has_critical_errors is True

    def test_has_critical_errors_false_for_normal_run(self):
        """Test no critical errors for normal run."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            twitter_accounts_total=5,
            twitter_accounts_active=4,
            broad_tweets_scraped=1000,
            deep_dive_tweets_scraped=500,
            email_sent=True,
        )

        assert diag.has_critical_errors is False

    def test_has_warnings_low_tweets(self):
        """Test warning detection for low tweet count."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=50,
            deep_dive_tweets_scraped=30,
        )

        assert diag.has_warnings is True

    def test_has_warnings_no_trends(self):
        """Test warning detection for no trends."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            trends_discovered=0,
            broad_tweets_scraped=1000,
        )

        assert diag.has_warnings is True

    def test_has_warnings_many_inactive_accounts(self):
        """Test warning when many accounts inactive."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            twitter_accounts_total=10,
            twitter_accounts_active=3,  # Only 30% active
            broad_tweets_scraped=1000,
            trends_discovered=5,
        )

        assert diag.has_warnings is True

    def test_add_error(self):
        """Test adding error to diagnostics."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
        )

        diag.add_error("Test error message")

        assert len(diag.errors) == 1
        assert "Test error message" in diag.errors[0]

    def test_add_warning(self):
        """Test adding warning to diagnostics."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
        )

        diag.add_warning("Test warning message")

        assert len(diag.warnings) == 1
        assert "Test warning message" in diag.warnings[0]

    def test_format_summary(self, sample_diagnostics):
        """Test summary formatting."""
        summary = sample_diagnostics.format_summary()

        assert "RUN DIAGNOSTICS SUMMARY" in summary
        assert sample_diagnostics.run_id in summary
        assert "SCRAPING STATS" in summary
        assert "TWITTER ACCOUNTS" in summary
        assert "ANALYSIS" in summary
        assert "PERFORMANCE" in summary


class TestDiagnosticsCollector:
    """Tests for DiagnosticsCollector class."""

    def test_init(self):
        """Test collector initialization."""
        collector = DiagnosticsCollector(run_id="20240101")

        assert collector.diagnostics.run_id == "20240101"
        assert collector.diagnostics.start_time is not None

    def test_finalize(self):
        """Test finalizing diagnostics."""
        collector = DiagnosticsCollector(run_id="20240101")
        collector.diagnostics.broad_tweets_scraped = 1000

        result = collector.finalize()

        assert result.end_time is not None
        assert result.broad_tweets_scraped == 1000


class TestRotateLogs:
    """Tests for rotate_logs function."""

    def test_rotate_creates_archive(self, tmp_path):
        """Test that rotation creates archive file."""
        log_file = tmp_path / "pipeline.log"
        log_file.write_text("Test log content\nLine 2\n")

        rotate_logs(str(log_file), keep_count=5)

        # Original should be gone
        assert not log_file.exists()

        # Archive should exist
        archives = list(tmp_path.glob("pipeline_*.log"))
        assert len(archives) == 1

    def test_rotate_skips_empty_file(self, tmp_path):
        """Test that empty files are not rotated."""
        log_file = tmp_path / "pipeline.log"
        log_file.write_text("")

        rotate_logs(str(log_file), keep_count=5)

        # No archive should be created
        archives = list(tmp_path.glob("pipeline_*.log"))
        assert len(archives) == 0

    def test_rotate_skips_missing_file(self, tmp_path):
        """Test that missing files are handled gracefully."""
        log_file = tmp_path / "nonexistent.log"

        # Should not raise
        rotate_logs(str(log_file), keep_count=5)

    def test_rotate_cleans_old_logs(self, tmp_path):
        """Test that old logs are cleaned up."""
        log_file = tmp_path / "pipeline.log"

        # Create multiple archive files
        for i in range(5):
            archive = tmp_path / f"pipeline_2024010{i}_120000.log"
            archive.write_text(f"Old log {i}")

        # Create current log
        log_file.write_text("Current log content")

        # Rotate with keep_count=2
        rotate_logs(str(log_file), keep_count=2)

        # Should keep only 2 + 1 new = 3 archives
        archives = list(tmp_path.glob("pipeline_*.log"))
        assert len(archives) <= 3


class TestShouldSendAdminAlert:
    """Tests for should_send_admin_alert function."""

    def test_alert_on_zero_tweets(self):
        """Test alert triggered for zero tweets."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=0,
            deep_dive_tweets_scraped=0,
        )

        should_alert, reason = should_send_admin_alert(diag)

        assert should_alert is True
        assert "CRITICAL" in reason
        assert "Zero tweets" in reason

    def test_alert_on_dead_accounts(self):
        """Test alert triggered when all accounts unavailable."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            twitter_accounts_total=5,
            twitter_accounts_active=0,
            broad_tweets_scraped=100,  # Some tweets scraped before failure
        )

        should_alert, reason = should_send_admin_alert(diag)

        assert should_alert is True
        assert "CRITICAL" in reason
        assert "accounts" in reason.lower()

    def test_alert_on_very_low_tweets(self):
        """Test alert triggered for very low tweet count."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=30,
            deep_dive_tweets_scraped=10,
            twitter_accounts_total=5,
            twitter_accounts_active=5,
        )

        should_alert, reason = should_send_admin_alert(diag)

        assert should_alert is True
        assert "WARNING" in reason
        assert "low" in reason.lower()

    def test_alert_on_no_trends(self):
        """Test alert triggered when no trends discovered."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=1000,
            trends_discovered=0,
            twitter_accounts_total=5,
            twitter_accounts_active=5,
        )

        should_alert, reason = should_send_admin_alert(diag)

        assert should_alert is True
        assert "No trends" in reason

    def test_no_alert_for_normal_run(self):
        """Test no alert for normal successful run."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=1500,
            deep_dive_tweets_scraped=500,
            trends_discovered=10,
            twitter_accounts_total=5,
            twitter_accounts_active=4,
            email_sent=True,
        )

        should_alert, reason = should_send_admin_alert(diag)

        assert should_alert is False
        assert "operational" in reason.lower()

    def test_alert_on_memory_storage_failure(self):
        """Test alert triggered when memory storage fails."""
        diag = RunDiagnostics(
            run_id="20240101",
            start_time=datetime.now(),
            broad_tweets_scraped=1500,
            deep_dive_tweets_scraped=500,
            trends_discovered=10,
            twitter_accounts_total=5,
            twitter_accounts_active=4,
            email_sent=True,
        )
        # Add a memory storage failure warning
        diag.add_warning("Failed to store memory: expected 1536 dimensions, not 3072")

        should_alert, reason = should_send_admin_alert(diag)

        assert should_alert is True
        assert "WARNING" in reason
        assert "Memory storage failed" in reason
