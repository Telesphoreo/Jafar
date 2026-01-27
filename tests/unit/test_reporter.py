"""
Unit tests for src/reporter.py

Tests email generation and HTML formatting.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.reporter import EmailConfig, EmailReporter, create_reporter_from_config
from src.temporal_analyzer import TrendTimeline


class TestEmailConfig:
    """Tests for EmailConfig dataclass."""

    def test_email_config_creation(self):
        """Test creating EmailConfig."""
        config = EmailConfig(
            host="smtp.example.com",
            port=587,
            username="user@example.com",
            password="password123",
            use_tls=True,
            email_from="sender@example.com",
            email_from_name="Test Sender",
            email_to=["recipient@example.com"],
        )

        assert config.host == "smtp.example.com"
        assert config.port == 587
        assert config.use_tls is True
        assert len(config.email_to) == 1


class TestEmailReporter:
    """Tests for EmailReporter class."""

    @pytest.fixture
    def reporter(self):
        """Create a test reporter."""
        config = EmailConfig(
            host="smtp.example.com",
            port=587,
            username="user@example.com",
            password="password123",
            use_tls=True,
            email_from="sender@example.com",
            email_from_name="Jafar Test",
            email_to=["recipient@example.com"],
        )
        return EmailReporter(config)

    def test_init(self, reporter):
        """Test reporter initialization."""
        assert reporter.config.host == "smtp.example.com"
        assert reporter.config.port == 587

    def test_generate_html_report(self, reporter):
        """Test HTML report generation."""
        html = reporter._generate_html_report(
            report_content="This is the **test** analysis content.",
            trends=["$NVDA", "Silver", "#inflation"],
            tweet_count=1500,
            provider_info="GPT-4o",
            signal_strength="high",
        )

        # Check structure
        assert "<!DOCTYPE html>" in html
        assert "Market Digest" in html
        assert "Jafar Intelligence System" in html

        # Check content
        assert "test" in html  # From report content
        assert "$NVDA" in html
        assert "Silver" in html
        assert "#inflation" in html
        assert "1500" in html or "1,500" in html

        # Check signal strength indicator
        assert "High" in html or "high" in html.lower()

    def test_generate_html_report_with_timelines(self, reporter):
        """Test HTML report with temporal badges."""
        mock_timeline = TrendTimeline(
            term="$NVDA",
            term_normalized="nvda",
            first_seen_today=datetime.now(),
            last_seen_today=datetime.now(),
            mentions_today=50,
            engagement_today=10000.0,
            consecutive_days=5,  # >= 3 to trigger "Day N" badge
            total_appearances=5,  # Not new
        )

        html = reporter._generate_html_report(
            report_content="Test content.",
            trends=["$NVDA"],
            tweet_count=500,
            provider_info="Gemini",
            signal_strength="medium",
            timelines={"$NVDA": mock_timeline},
        )

        # Should include temporal badge - badge appears as "Day N" for consecutive_days >= 3
        assert "Day 5" in html or mock_timeline.temporal_badge in html

    def test_generate_html_report_low_signal(self, reporter):
        """Test HTML report with low signal."""
        html = reporter._generate_html_report(
            report_content="Quiet day in the markets.",
            trends=["$SPY"],
            tweet_count=200,
            provider_info="Test",
            signal_strength="low",
        )

        # Should have low signal styling
        assert "Low" in html or "low" in html.lower()

    def test_generate_html_report_no_signal(self, reporter):
        """Test HTML report with no signal."""
        html = reporter._generate_html_report(
            report_content="Nothing to report.",
            trends=[],
            tweet_count=50,
            provider_info="Test",
            signal_strength="none",
        )

        assert html is not None

    def test_generate_plain_text(self, reporter):
        """Test plain text report generation."""
        text = reporter._generate_plain_text(
            report_content="This is the analysis content.",
            trends=["$NVDA", "Silver"],
            tweet_count=1000,
        )

        assert "Jafar Intelligence System" in text
        assert "Market Digest" in text
        assert "This is the analysis content." in text
        assert "$NVDA" in text
        assert "Silver" in text
        assert "1000" in text

    def test_generate_plain_text_includes_disclaimer(self, reporter):
        """Test that plain text includes disclaimer."""
        text = reporter._generate_plain_text(
            report_content="Analysis.",
            trends=["Test"],
            tweet_count=100,
        )

        assert "Not financial advice" in text

    @patch("smtplib.SMTP")
    def test_send_email_success(self, mock_smtp_class, reporter):
        """Test successful email sending."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value = mock_smtp

        result = reporter.send_email(
            report_content="Test analysis.",
            trends=["$NVDA"],
            tweet_count=500,
            provider_info="Test",
            signal_strength="medium",
        )

        assert result is True
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once()
        mock_smtp.sendmail.assert_called_once()
        mock_smtp.quit.assert_called_once()

    @patch("smtplib.SMTP")
    def test_send_email_auth_failure(self, mock_smtp_class, reporter):
        """Test email sending with auth failure."""
        import smtplib

        mock_smtp = MagicMock()
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")
        mock_smtp_class.return_value = mock_smtp

        result = reporter.send_email(
            report_content="Test.",
            trends=[],
            tweet_count=100,
            provider_info="Test",
        )

        assert result is False

    @patch("smtplib.SMTP")
    def test_send_email_timeout(self, mock_smtp_class, reporter):
        """Test email sending with timeout."""
        mock_smtp_class.side_effect = TimeoutError("Connection timeout")

        result = reporter.send_email(
            report_content="Test.",
            trends=[],
            tweet_count=100,
            provider_info="Test",
        )

        assert result is False

    def test_send_email_subject_includes_signal(self, reporter):
        """Test that email subject reflects signal strength."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp

            reporter.send_email(
                report_content="Test.",
                trends=["$NVDA"],
                tweet_count=500,
                provider_info="Test",
                signal_strength="high",
            )

            # Check sendmail was called with subject containing signal
            call_args = mock_smtp.sendmail.call_args
            message = call_args[0][2]  # Third arg is the message
            assert "[High signal]" in message

    def test_generate_admin_html_report(self, reporter, sample_diagnostics):
        """Test admin diagnostics HTML generation."""
        html = reporter._generate_admin_html_report(
            diagnostics=sample_diagnostics,
            alert_reason="Test alert reason",
        )

        assert "<!DOCTYPE html>" in html
        assert "Admin Diagnostics" in html
        assert "System Report" in html
        assert sample_diagnostics.run_id in html
        assert "Test alert reason" in html

    def test_generate_admin_html_report_critical(self, reporter, sample_diagnostics):
        """Test admin report for critical errors."""
        sample_diagnostics.errors = ["Critical error 1", "Critical error 2"]

        html = reporter._generate_admin_html_report(
            diagnostics=sample_diagnostics,
            alert_reason="CRITICAL: Test failure",
        )

        assert "Critical error 1" in html
        assert "Critical error 2" in html

    def test_generate_admin_html_report_warnings(self, reporter, sample_diagnostics):
        """Test admin report with warnings."""
        sample_diagnostics.warnings = ["Warning 1"]

        html = reporter._generate_admin_html_report(diagnostics=sample_diagnostics)

        assert "Warning 1" in html

    @patch("smtplib.SMTP")
    def test_send_admin_email_success(self, mock_smtp_class, reporter, sample_diagnostics):
        """Test successful admin email sending."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value = mock_smtp

        result = reporter.send_admin_email(
            diagnostics=sample_diagnostics,
            alert_reason="Test alert",
        )

        assert result is True
        mock_smtp.sendmail.assert_called_once()

    @patch("smtplib.SMTP")
    def test_send_admin_email_failure(self, mock_smtp_class, reporter, sample_diagnostics):
        """Test admin email sending failure."""
        mock_smtp = MagicMock()
        mock_smtp.login.side_effect = Exception("Login failed")
        mock_smtp_class.return_value = mock_smtp

        result = reporter.send_admin_email(diagnostics=sample_diagnostics)

        assert result is False


class TestCreateReporterFromConfig:
    """Tests for create_reporter_from_config factory function."""

    def test_creates_reporter(self):
        """Test factory function creates reporter correctly."""
        reporter = create_reporter_from_config(
            host="smtp.test.com",
            port=465,
            username="user@test.com",
            password="pass123",
            use_tls=False,
            email_from="from@test.com",
            email_from_name="Test System",
            email_to=["to@test.com", "to2@test.com"],
        )

        assert isinstance(reporter, EmailReporter)
        assert reporter.config.host == "smtp.test.com"
        assert reporter.config.port == 465
        assert reporter.config.use_tls is False
        assert len(reporter.config.email_to) == 2


class TestMarkdownFormatting:
    """Tests for markdown to HTML conversion in reports."""

    @pytest.fixture
    def reporter(self):
        """Create test reporter."""
        config = EmailConfig(
            host="smtp.example.com",
            port=587,
            username="user@example.com",
            password="password123",
            use_tls=True,
            email_from="sender@example.com",
            email_from_name="Test",
            email_to=["recipient@example.com"],
        )
        return EmailReporter(config)

    def test_bold_conversion(self, reporter):
        """Test that **bold** is converted to <strong>."""
        html = reporter._generate_html_report(
            report_content="This is **bold text** here.",
            trends=[],
            tweet_count=100,
            provider_info="Test",
            signal_strength="low",
        )

        assert "<strong>bold text</strong>" in html

    def test_italic_conversion(self, reporter):
        """Test that *italic* is converted to <em>."""
        html = reporter._generate_html_report(
            report_content="This is *italic text* here.",
            trends=[],
            tweet_count=100,
            provider_info="Test",
            signal_strength="low",
        )

        assert "<em>italic text</em>" in html

    def test_paragraph_breaks(self, reporter):
        """Test that double newlines create paragraphs."""
        html = reporter._generate_html_report(
            report_content="First paragraph.\n\nSecond paragraph.",
            trends=[],
            tweet_count=100,
            provider_info="Test",
            signal_strength="low",
        )

        assert "</p><p" in html

    def test_line_breaks(self, reporter):
        """Test that single newlines create <br> tags."""
        html = reporter._generate_html_report(
            report_content="Line one.\nLine two.",
            trends=[],
            tweet_count=100,
            provider_info="Test",
            signal_strength="low",
        )

        assert "<br>" in html
