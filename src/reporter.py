"""
Email Reporter Module.

Sends the economic digest via SMTP with nicely formatted HTML.
"""

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .diagnostics import RunDiagnostics

logger = logging.getLogger("jafar.reporter")


@dataclass
class EmailConfig:
    """Email configuration for the reporter."""
    host: str
    port: int
    username: str
    password: str
    use_tls: bool
    email_from: str
    email_from_name: str
    email_to: list[str]


class EmailReporter:
    """
    Sends economic digest reports via email.

    Formats the LLM-generated analysis into a clean HTML email.
    """

    def __init__(self, config: EmailConfig):
        """
        Initialize the email reporter.

        Args:
            config: Email configuration with SMTP credentials.
        """
        self.config = config
        logger.info(f"EmailReporter initialized for {config.host}:{config.port}")

    def _generate_html_report(
        self,
        report_content: str,
        trends: list[str],
        tweet_count: int,
        provider_info: str,
        signal_strength: str = "low",
        timelines: dict = None,
    ) -> str:
        """
        Generate a clean, minimalist HTML email.

        Args:
            report_content: The LLM-generated analysis.
            trends: List of trending topics analyzed.
            tweet_count: Total number of tweets analyzed.
            provider_info: LLM provider used for analysis.
            signal_strength: Signal strength rating (high/medium/low/none).
            timelines: Dict of {trend: TrendTimeline} for temporal badges.

        Returns:
            HTML formatted email body.
        """
        today = datetime.now().strftime("%B %d, %Y")

        # Minimalist signal indicators (Monochrome/High Contrast)
        signal_styles = {
            "high": "border-left: 4px solid #000000; background-color: #fafafa;",
            "medium": "border-left: 4px solid #666666; background-color: #fafafa;",
            "low": "border-left: 4px solid #bbbbbb; background-color: #ffffff;",
            "none": "border-left: 4px solid #eeeeee; background-color: #ffffff; color: #888888;",
        }
        signal_style = signal_styles.get(signal_strength, signal_styles["low"])
        signal_text = signal_strength.capitalize()

        # Format trends as simple tags with temporal badges
        trend_tag_list = []
        for trend in trends:
            # Get temporal badge if available
            badge = ""
            if timelines and trend in timelines:
                timeline = timelines[trend]
                badge = timeline.temporal_badge
                if badge:
                    badge = f' <span style="font-size: 11px; opacity: 0.7;">{badge}</span>'

            trend_tag_list.append(
                f'<span style="border: 1px solid #ddd; padding: 4px 10px; font-size: 12px; margin-right: 8px; display: inline-block; margin-bottom: 8px; font-family: monospace; background-color: #fff;">{trend}{badge}</span>'
            )

        trend_tags = " ".join(trend_tag_list)

        # HTML Formatting
        import re
        formatted_content = report_content
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
        formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
        formatted_content = formatted_content.replace('\n\n', '</p><p style="margin: 20px 0;">')
        formatted_content = formatted_content.replace('\n', '<br>')

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jafar Digest - {today}</title>
</head>
<body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #111; line-height: 1.6; margin: 0; padding: 40px 20px; background-color: #f6f6f6;">
    <div style="max-width: 850px; margin: 0 auto; border: 1px solid #000; background-color: #fff; box-shadow: 10px 10px 0px #000;">
        
        <!-- Header -->
        <div style="border-bottom: 1px solid #000; padding: 30px 40px; background-color: #000; color: #fff;">
            <div style="font-family: monospace; font-size: 13px; letter-spacing: 2px; margin-bottom: 10px; opacity: 0.8;">
                Jafar Intelligence System
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td align="left">
                        <h1 style="margin: 0; font-size: 32px; font-weight: 700; letter-spacing: -1px;">
                            Market Digest
                        </h1>
                    </td>
                    <td align="right" style="font-family: monospace; font-size: 16px; opacity: 0.9;">
                        {today}
                    </td>
                </tr>
            </table>
        </div>

        <!-- Signal Banner -->
        <div style="padding: 20px 40px; {signal_style} border-bottom: 1px solid #000;">
            <span style="font-family: monospace; font-weight: bold; font-size: 14px; letter-spacing: 2px;">
                Signal strength: {signal_text}
            </span>
        </div>

        <!-- Metadata Row -->
        <div style="display: flex; border-bottom: 1px solid #000; font-family: monospace; font-size: 13px; background-color: #fcfcfc;">
            <div style="padding: 15px 40px; border-right: 1px solid #000; flex: 1;">
                Analyzed: <strong>{tweet_count}</strong> tweets
            </div>
            <div style="padding: 15px 40px; flex: 1;">
                Detected: <strong>{len(trends)}</strong> trends
            </div>
        </div>

        <!-- Trends Section -->
        <div style="padding: 25px 40px; border-bottom: 1px solid #eee; background-color: #fff;">
            <div style="font-family: monospace; font-size: 11px; color: #888; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;">
                Current Market Topics
            </div>
            <div style="line-height: 1.8;">
                {trend_tags}
            </div>
        </div>

        <!-- Main Content -->
        <div style="padding: 50px 40px;">
            <div style="font-size: 18px; color: #111; line-height: 1.7; max-width: 750px;">
                <p style="margin-top: 0;">
                    {formatted_content}
                </p>
            </div>
        </div>

        <!-- Footer -->
        <div style="border-top: 1px solid #000; padding: 30px 40px; font-size: 13px; color: #444; background-color: #fafafa;">
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td style="padding-bottom: 20px;">
                        <div style="font-weight: bold; font-size: 15px; margin-bottom: 5px; color: #000;">Jafar</div>
                        <div style="opacity: 0.8;">The villain to BlackRock's Aladdin.</div>
                    </td>
                </tr>
                <tr>
                    <td style="border-top: 1px solid #ddd; padding-top: 20px;">
                        <p style="margin: 0; font-family: monospace; font-size: 11px; color: #999; letter-spacing: 1px;">
                            Not financial advice | {today}
                        </p>
                    </td>
                </tr>
            </table>
        </div>
    </div>
    <div style="max-width: 850px; margin: 20px auto 0; text-align: center; font-family: monospace; font-size: 10px; color: #bbb; letter-spacing: 2px;">
        End of transmission
    </div>
</body>
</html>
"""
        return html

    def _generate_plain_text(
        self,
        report_content: str,
        trends: list[str],
        tweet_count: int,
    ) -> str:
        """
        Generate a plain text version of the report.

        Args:
            report_content: The LLM-generated analysis.
            trends: List of trending topics analyzed.
            tweet_count: Total number of tweets analyzed.

        Returns:
            Plain text formatted email body.
        """
        today = datetime.now().strftime("%B %d, %Y")
        trends_str = ", ".join(trends)

        return f"""
Jafar Intelligence System
Market Digest - {today}
--------------------------------------------------------------------------------

Signal: Detected
Inputs: {tweet_count} tweets
Topics: {trends_str}

--------------------------------------------------------------------------------

{report_content}

--------------------------------------------------------------------------------

Disclaimer: Not financial advice.
"""

    def send_email(
        self,
        report_content: str,
        trends: list[str],
        tweet_count: int,
        provider_info: str = "AI",
        signal_strength: str = "low",
        timelines: dict = None,
    ) -> bool:
        """
        Send the economic digest via email.

        Args:
            report_content: The LLM-generated analysis.
            trends: List of trending topics analyzed.
            tweet_count: Total number of tweets analyzed.
            provider_info: Description of LLM used.
            signal_strength: Signal strength rating (high/medium/low/none).
            timelines: Dict of {trend: TrendTimeline} for temporal badges.

        Returns:
            True if email was sent successfully.
        """
        today = datetime.now().strftime("%B %d, %Y")

        # Subject line reflects signal strength
        signal_prefix = {
            "high": "[High signal]",
            "medium": "[Medium]",
            "low": "",
            "none": "[Quiet day]",
        }.get(signal_strength, "")

        subject = f"{signal_prefix} Jafar Market Digest - {today}".strip()

        # Create message container
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.email_from_name} <{self.config.email_from}>"
        # Using the sender's email in 'To' and putting everyone else in BCC
        msg["To"] = self.config.email_from

        # Generate both plain text and HTML versions
        text_content = self._generate_plain_text(report_content, trends, tweet_count)
        html_content = self._generate_html_report(
            report_content, trends, tweet_count, provider_info, signal_strength, timelines
        )

        # Attach both versions (email clients will choose the best one)
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)

        logger.info(f"Sending email to {len(self.config.email_to)} recipient(s) via BCC")
        logger.info(f"SMTP Server: {self.config.host}:{self.config.port} (TLS: {self.config.use_tls})")
        logger.info(f"From: {msg['From']}")
        logger.info(f"To (Visible): {msg['To']}")
        logger.info(f"Subject: {subject}")

        try:
            # Connect to SMTP server with timeout protection
            timeout = 30  # 30 seconds for all SMTP operations

            logger.info("Step 1: Connecting to SMTP server...")
            if self.config.use_tls:
                server = smtplib.SMTP(self.config.host, self.config.port, timeout=timeout)
                server.set_debuglevel(1)  # Enable SMTP protocol debugging
                logger.info("Step 2: Starting TLS...")
                server.starttls()
                logger.info("TLS started successfully")
            else:
                server = smtplib.SMTP_SSL(self.config.host, self.config.port, timeout=timeout)
                server.set_debuglevel(1)  # Enable SMTP protocol debugging
                logger.info("SSL connection established")

            logger.info(f"Step 3: Logging in as {self.config.username}...")
            server.login(self.config.username, self.config.password)
            logger.info("Login successful")

            logger.info("Step 4: Sending email...")
            # to_addrs should include all recipients (To + BCC)
            all_recipients = self.config.email_to
            server.sendmail(
                self.config.email_from,
                all_recipients,
                msg.as_string(),
            )
            logger.info("Email sent, closing connection...")

            server.quit()
            logger.info("✓ Email sent successfully")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"✗ SMTP authentication failed: {e}")
            logger.error(f"  Username: {self.config.username}")
            logger.error("  Check SMTP_USERNAME and SMTP_PASSWORD in .env")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"✗ SMTP error: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            return False
        except TimeoutError as e:
            logger.error(f"✗ SMTP connection timeout after 30s: {e}")
            logger.error(f"  Check if {self.config.host}:{self.config.port} is reachable")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to send email: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return False

    def _generate_admin_html_report(
        self,
        diagnostics: "RunDiagnostics",
        alert_reason: str = "",
    ) -> str:
        """
        Generate HTML admin diagnostics email.

        Args:
            diagnostics: RunDiagnostics object with run statistics
            alert_reason: Reason for alert (if any)

        Returns:
            HTML formatted admin email
        """
        today = datetime.now().strftime("%B %d, %Y %H:%M:%S")

        # Determine status styling
        if diagnostics.has_critical_errors:
            status_text = "Critical error"
            status_color = "#ff0000"
            status_bg = "#fff0f0"
        elif diagnostics.has_warnings:
            status_text = "Warning"
            status_color = "#ff8800"
            status_bg = "#fff8f0"
        else:
            status_text = "Operational"
            status_color = "#00aa00"
            status_bg = "#f0fff0"

        # Format errors and warnings
        errors_html = ""
        if diagnostics.errors:
            errors_list = "".join(f"<li style='margin: 5px 0; font-family: monospace; font-size: 13px; color: #c00;'>{e}</li>" for e in diagnostics.errors)
            errors_html = f"""
            <div style="margin: 20px 0; padding: 20px; background-color: #fff5f5; border-left: 4px solid #c00;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #c00;">Errors detected:</div>
                <ul style="margin: 0; padding-left: 20px;">
                    {errors_list}
                </ul>
            </div>
            """

        warnings_html = ""
        if diagnostics.warnings:
            warnings_list = "".join(f"<li style='margin: 5px 0; font-family: monospace; font-size: 13px; color: #c80;'>{w}</li>" for w in diagnostics.warnings)
            warnings_html = f"""
            <div style="margin: 20px 0; padding: 20px; background-color: #fffef5; border-left: 4px solid #c80;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #c80;">Warnings:</div>
                <ul style="margin: 0; padding-left: 20px;">
                    {warnings_list}
                </ul>
            </div>
            """

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jafar Admin Diagnostics - {today}</title>
</head>
<body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #111; line-height: 1.6; margin: 0; padding: 40px 20px; background-color: #f6f6f6;">
    <div style="max-width: 850px; margin: 0 auto; border: 1px solid #000; background-color: #fff; box-shadow: 10px 10px 0px #000;">

        <!-- Header -->
        <div style="border-bottom: 1px solid #000; padding: 30px 40px; background-color: #000; color: #fff;">
            <div style="font-family: monospace; font-size: 13px; letter-spacing: 2px; margin-bottom: 10px; opacity: 0.8;">
                Jafar Admin Diagnostics
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td align="left">
                        <h1 style="margin: 0; font-size: 32px; font-weight: 700; letter-spacing: -1px;">
                            System Report
                        </h1>
                    </td>
                    <td align="right" style="font-family: monospace; font-size: 16px; opacity: 0.9;">
                        {today}
                    </td>
                </tr>
            </table>
        </div>

        <!-- Status Banner -->
        <div style="padding: 20px 40px; background-color: {status_bg}; border-bottom: 1px solid #000; border-left: 4px solid {status_color};">
            <span style="font-family: monospace; font-weight: bold; font-size: 14px; letter-spacing: 2px; color: {status_color};">
                Status: {status_text}
            </span>
            {f'<div style="margin-top: 10px; font-size: 13px; color: #333;">{alert_reason}</div>' if alert_reason else ''}
        </div>

        <!-- Run Stats -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="font-family: monospace; font-size: 11px; color: #888; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">
                Run Statistics
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace;">
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Run ID</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.run_id}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Duration</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.duration_formatted}</td>
                </tr>
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Signal Strength</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.signal_strength.upper()}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Email Sent</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{'✓ Yes' if diagnostics.email_sent else '✗ No'}</td>
                </tr>
            </table>
        </div>

        <!-- Scraping Stats -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="font-family: monospace; font-size: 11px; color: #888; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">
                Scraping Statistics
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace;">
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Broad Topics</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.broad_topics_completed}/{diagnostics.broad_topics_attempted}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Broad Tweets</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.broad_tweets_scraped}</td>
                </tr>
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Trends Discovered</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.trends_discovered}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>After LLM Filter</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.trends_filtered_by_llm}</td>
                </tr>
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Deep Dive Trends</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.deep_dive_trends_completed}/{diagnostics.deep_dive_trends_attempted}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Deep Dive Tweets</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.deep_dive_tweets_scraped}</td>
                </tr>
                <tr style="background-color: #fafafa; font-weight: bold;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Total Tweets</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.total_tweets}</td>
                </tr>
            </table>
        </div>

        <!-- Twitter Accounts -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="font-family: monospace; font-size: 11px; color: #888; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">
                Twitter Account Health
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace;">
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Active Accounts</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.twitter_accounts_active}/{diagnostics.twitter_accounts_total}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Rate Limited</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.twitter_accounts_rate_limited}</td>
                </tr>
            </table>
        </div>

        <!-- Analysis Stats -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="font-family: monospace; font-size: 11px; color: #888; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">
                Analysis & Processing
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace;">
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>LLM Calls</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.llm_calls_made}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>LLM Tokens Used</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.llm_tokens_used:,}</td>
                </tr>
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Fact Checks</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.fact_checks_performed}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Temporal Patterns</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.temporal_patterns_detected}</td>
                </tr>
            </table>
        </div>

        <!-- Performance -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="font-family: monospace; font-size: 11px; color: #888; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">
                Performance Breakdown
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace;">
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Step 1: Broad Scraping</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.time_step1_scraping:.1f}s</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Step 2: Trend Analysis</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.time_step2_analysis:.1f}s</td>
                </tr>
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Step 3: Deep Dive</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.time_step3_deep_dive:.1f}s</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Step 4: LLM Analysis</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.time_step4_llm:.1f}s</td>
                </tr>
                <tr style="background-color: #fafafa;">
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Step 5: Email</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.time_step5_email:.1f}s</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Step 6: Storage</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{diagnostics.time_step6_storage:.1f}s</td>
                </tr>
            </table>
        </div>

        {errors_html}
        {warnings_html}

        <!-- Footer -->
        <div style="border-top: 1px solid #000; padding: 30px 40px; font-size: 13px; color: #444; background-color: #fafafa;">
            <p style="margin: 0; font-family: monospace; font-size: 11px; color: #999; letter-spacing: 1px;">
                Jafar Admin Diagnostics | {today}
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def send_admin_email(
        self,
        diagnostics: "RunDiagnostics",
        alert_reason: str = "",
        admin_recipients: list[str] = None,
    ) -> bool:
        """
        Send admin diagnostics email.

        Args:
            diagnostics: RunDiagnostics object with run statistics
            alert_reason: Reason for alert (if any)
            admin_recipients: List of admin email addresses (defaults to main recipients)

        Returns:
            True if email sent successfully
        """
        recipients = admin_recipients or self.config.email_to
        today = datetime.now().strftime("%B %d, %Y %H:%M")

        # Subject line reflects status
        if diagnostics.has_critical_errors:
            prefix = "[Critical]"
        elif diagnostics.has_warnings:
            prefix = "[Warning]"
        else:
            prefix = "[Info]"

        subject = f"{prefix} Jafar Admin Diagnostics - {today}"

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.email_from_name} <{self.config.email_from}>"
        # Using the sender's email in 'To' and putting everyone else in BCC
        msg["To"] = self.config.email_from

        # Generate HTML
        html_content = self._generate_admin_html_report(diagnostics, alert_reason)

        # Simple plain text version
        text_content = f"""
Jafar Admin Diagnostics
{today}

Status: {diagnostics.has_critical_errors and 'Critical error' or diagnostics.has_warnings and 'Warning' or 'Operational'}
{alert_reason and f'Alert reason: {alert_reason}' or ''}

Run ID: {diagnostics.run_id}
Duration: {diagnostics.duration_formatted}
Total tweets: {diagnostics.total_tweets}
Signal strength: {diagnostics.signal_strength.capitalize()}

Twitter accounts: {diagnostics.twitter_accounts_active}/{diagnostics.twitter_accounts_total} active

See HTML version for full details.
"""

        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)

        logger.info(f"Sending admin diagnostics email to {len(recipients)} recipient(s) via BCC")

        try:
            timeout = 30

            if self.config.use_tls:
                server = smtplib.SMTP(self.config.host, self.config.port, timeout=timeout)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.config.host, self.config.port, timeout=timeout)

            server.login(self.config.username, self.config.password)
            server.sendmail(self.config.email_from, recipients, msg.as_string())
            server.quit()

            logger.info("✓ Admin diagnostics email sent successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to send admin diagnostics email: {e}")
            return False


def create_reporter_from_config(
    host: str,
    port: int,
    username: str,
    password: str,
    use_tls: bool,
    email_from: str,
    email_from_name: str,
    email_to: list[str],
) -> EmailReporter:
    """
    Factory function to create an EmailReporter from configuration values.

    Args:
        host: SMTP server host.
        port: SMTP server port.
        username: SMTP username.
        password: SMTP password.
        use_tls: Whether to use TLS.
        email_from: Sender email address.
        email_from_name: Sender display name.
        email_to: List of recipient email addresses.

    Returns:
        Configured EmailReporter instance.
    """
    config = EmailConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        use_tls=use_tls,
        email_from=email_from,
        email_from_name=email_from_name,
        email_to=email_to,
    )
    return EmailReporter(config)
