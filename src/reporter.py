# pylint: disable=too-many-lines
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
        Generate a clean, minimalist HTML email for the news digest.

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

        signal_configs = {
            "high": {"border_color": "#000000"},
            "medium": {"border_color": "#666666"},
            "low": {"border_color": "#bbbbbb"},
            "none": {"border_color": "#eeeeee"},
        }
        signal_config = signal_configs.get(signal_strength, signal_configs["low"])
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
                f'<span style="border: 1px solid #ddd; padding: 4px 10px;'
                f' font-size: 12px; margin-right: 8px; display: inline-block;'
                f' margin-bottom: 8px; font-family: monospace;'
                f' background-color: #fff;">{trend}{badge}</span>'
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

        <!-- Header with perspective grid SVG -->
        <div style="border-bottom: 1px solid #333; padding: 30px 40px; background-color: #0a0a0a; color: #ffffff; position: relative;">
            <svg width="100%" height="100%" style="position: absolute; top: 0; left: 0; pointer-events: none;">
                <line x1="50%" y1="0" x2="0%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="10%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="20%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="30%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="40%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="60%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="70%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="80%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="90%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="100%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="0" y1="40%" x2="100%" y2="40%" stroke="#1a1a1a" stroke-width="0.5"/>
                <line x1="0" y1="60%" x2="100%" y2="60%" stroke="#1a1a1a" stroke-width="0.5"/>
                <line x1="0" y1="80%" x2="100%" y2="80%" stroke="#1a1a1a" stroke-width="0.5"/>
            </svg>
            <div style="font-family: monospace; font-size: 13px; letter-spacing: 2px; margin-bottom: 10px; color: #888888; position: relative;">
                Jafar Intelligence System
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" border="0" style="position: relative;">
                <tr>
                    <td align="left">
                        <h1 style="margin: 0; font-size: 32px; font-weight: 700; letter-spacing: -1px; color: #ffffff;">
                            Market Digest
                        </h1>
                    </td>
                    <td align="right" style="font-family: monospace; font-size: 16px; color: #888888;">
                        {today}
                    </td>
                </tr>
            </table>
        </div>

        <!-- Signal Banner -->
        <div style="padding: 20px 40px; border-left: 4px solid {signal_config['border_color']}; border-bottom: 1px solid #000;">
            <span style="font-family: monospace; font-weight: bold; font-size: 14px; letter-spacing: 2px; color: #111;">
                Signal Strength: {signal_text}
            </span>
        </div>

        <!-- Metadata Row (table layout for email compatibility) -->
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="border-bottom: 1px solid #000; font-family: monospace; font-size: 13px; background-color: #fcfcfc;">
            <tr>
                <td style="padding: 15px 40px; border-right: 1px solid #000; width: 50%;">
                    Analyzed: <strong>{tweet_count}</strong> tweets
                </td>
                <td style="padding: 15px 40px; width: 50%;">
                    Detected: <strong>{len(trends)}</strong> trends
                </td>
            </tr>
        </table>

        <!-- Trends Section -->
        <div style="padding: 25px 40px; border-bottom: 1px solid #000; background-color: #fff;">
            <div style="font-family: monospace; font-size: 11px; color: #888; margin-bottom: 12px; letter-spacing: 1px;">
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
        <div style="border-top: 1px solid #ddd; padding: 30px 40px; font-size: 13px; color: #444; background-color: #fafafa;">
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td style="padding-bottom: 20px;">
                        <div style="font-weight: bold; font-size: 15px; margin-bottom: 5px; color: #000;">Jafar</div>
                        <div style="color: #999;">The villain to BlackRock's Aladdin.</div>
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
    <!-- End of transmission -->
    <div style="max-width: 850px; margin: 20px auto 0; text-align: center; padding: 15px 0;">
        <span style="font-family: monospace; font-size: 10px; color: #bbb; letter-spacing: 2px;">
            End of transmission
        </span>
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
        subject_line: str = None,
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
            subject_line: Optional custom subject line from LLM.

        Returns:
            True if email was sent successfully.
        """
        today = datetime.now().strftime("%B %d, %Y")

        # Use LLM-generated subject if provided, otherwise fallback
        if subject_line:
            subject = f"{subject_line} - {today}"
        else:
            subject = f"Jafar Market Digest - {today}"

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

        # Attach both for multipart/alternative.
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)

        logger.info(f"Sending email: {subject}")
        logger.debug(f"Recipients: {len(self.config.email_to)} via BCC")
        logger.debug(f"SMTP: {self.config.host}:{self.config.port} (TLS: {self.config.use_tls})")

        try:
            timeout = 30

            logger.debug("Connecting to SMTP server...")
            if self.config.use_tls:
                server = smtplib.SMTP(self.config.host, self.config.port, timeout=timeout)
                logger.debug("Starting TLS...")
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.config.host, self.config.port, timeout=timeout)

            logger.debug(f"Logging in as {self.config.username}...")
            server.login(self.config.username, self.config.password)

            logger.debug("Sending email...")
            all_recipients = self.config.email_to
            server.sendmail(
                self.config.email_from,
                all_recipients,
                msg.as_string(),
            )

            server.quit()
            logger.info("Email sent successfully")
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
            status_text = "Critical Error"
            status_color = "#ff0000"
            status_bg = "#fff0f0"
        elif diagnostics.has_warnings:
            status_text = "Warning"
            status_color = "#ffaa00"
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
            <div style="margin: 20px 40px; padding: 20px; background-color: #fff5f5; border-left: 4px solid #c00;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #c00;">Errors Detected:</div>
                <ul style="margin: 0; padding-left: 20px;">
                    {errors_list}
                </ul>
            </div>
            """

        warnings_html = ""
        if diagnostics.warnings:
            warnings_list = "".join(f"<li style='margin: 5px 0; font-family: monospace; font-size: 13px; color: #c80;'>{w}</li>" for w in diagnostics.warnings)
            warnings_html = f"""
            <div style="margin: 20px 40px; padding: 20px; background-color: #fffef5; border-left: 4px solid #c80;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #c80;">Warnings:</div>
                <ul style="margin: 0; padding-left: 20px;">
                    {warnings_list}
                </ul>
            </div>
            """

        # Admin table styles
        admin_td = "padding: 10px; border-bottom: 1px solid #eee; font-family: monospace; font-size: 14px; background-color: #fff;"
        admin_td_alt = "padding: 10px; border-bottom: 1px solid #eee; font-family: monospace; font-size: 14px; background-color: #fafafa;"
        admin_section_header = "font-family: monospace; font-size: 11px; color: #888; margin-bottom: 15px; letter-spacing: 1px;"

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

        <!-- Header with perspective grid SVG -->
        <div style="border-bottom: 1px solid #333; padding: 30px 40px; background-color: #0a0a0a; color: #ffffff; position: relative;">
            <svg width="100%" height="100%" style="position: absolute; top: 0; left: 0; pointer-events: none;">
                <line x1="50%" y1="0" x2="0%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="10%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="20%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="30%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="40%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="60%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="70%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="80%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="90%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="50%" y1="0" x2="100%" y2="100%" stroke="#1a1a1a" stroke-width="1"/>
                <line x1="0" y1="40%" x2="100%" y2="40%" stroke="#1a1a1a" stroke-width="0.5"/>
                <line x1="0" y1="60%" x2="100%" y2="60%" stroke="#1a1a1a" stroke-width="0.5"/>
                <line x1="0" y1="80%" x2="100%" y2="80%" stroke="#1a1a1a" stroke-width="0.5"/>
            </svg>
            <div style="font-family: monospace; font-size: 13px; letter-spacing: 2px; margin-bottom: 10px; color: #888888; position: relative;">
                Jafar Admin Diagnostics
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" border="0" style="position: relative;">
                <tr>
                    <td align="left">
                        <h1 style="margin: 0; font-size: 32px; font-weight: 700; letter-spacing: -1px; color: #ffffff;">
                            System Report
                        </h1>
                    </td>
                    <td align="right" style="font-family: monospace; font-size: 16px; color: #888888;">
                        {today}
                    </td>
                </tr>
            </table>
        </div>

        <!-- Status Banner -->
        <div style="padding: 20px 40px; background-color: {status_bg}; border-bottom: 1px solid #eee; border-left: 4px solid {status_color};">
            <span style="font-family: monospace; font-weight: bold; font-size: 14px; letter-spacing: 2px; color: {status_color};">
                Status: {status_text}
            </span>
            {f'<div style="margin-top: 10px; font-size: 13px; color: #333;">{alert_reason}</div>' if alert_reason else ''}
        </div>

        <!-- Run Stats -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="{admin_section_header}">
                Run Statistics
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace; border-collapse: collapse;">
                <tr>
                    <td style="{admin_td_alt}"><strong>Run ID</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.run_id}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Duration</strong></td>
                    <td style="{admin_td}">{diagnostics.duration_formatted}</td>
                </tr>
                <tr>
                    <td style="{admin_td_alt}"><strong>Signal Strength</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.signal_strength.capitalize()}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Email Sent</strong></td>
                    <td style="{admin_td}">{'Yes' if diagnostics.email_sent else 'No'}</td>
                </tr>
            </table>
        </div>

        <!-- Scraping Stats -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="{admin_section_header}">
                Scraping Statistics
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace; border-collapse: collapse;">
                <tr>
                    <td style="{admin_td_alt}"><strong>Broad Topics</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.broad_topics_completed}/{diagnostics.broad_topics_attempted}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Broad Tweets</strong></td>
                    <td style="{admin_td}">{diagnostics.broad_tweets_scraped}</td>
                </tr>
                <tr>
                    <td style="{admin_td_alt}"><strong>Trends Discovered</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.trends_discovered}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>After LLM Filter</strong></td>
                    <td style="{admin_td}">{diagnostics.trends_filtered_by_llm}</td>
                </tr>
                <tr>
                    <td style="{admin_td_alt}"><strong>Deep Dive Trends</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.deep_dive_trends_completed}/{diagnostics.deep_dive_trends_attempted}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Deep Dive Tweets</strong></td>
                    <td style="{admin_td}">{diagnostics.deep_dive_tweets_scraped}</td>
                </tr>
                <tr>
                    <td style="{admin_td_alt}"><strong>Total Tweets</strong></td>
                    <td style="{admin_td_alt} font-weight: bold;">{diagnostics.total_tweets}</td>
                </tr>
            </table>
        </div>

        <!-- Twitter Accounts -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="{admin_section_header}">
                Twitter Account Health
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace; border-collapse: collapse;">
                <tr>
                    <td style="{admin_td_alt}"><strong>Active Accounts</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.twitter_accounts_active}/{diagnostics.twitter_accounts_total}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Rate Limited</strong></td>
                    <td style="{admin_td}">{diagnostics.twitter_accounts_rate_limited}</td>
                </tr>
            </table>
        </div>

        <!-- Analysis Stats -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="{admin_section_header}">
                Analysis & Processing
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace; border-collapse: collapse;">
                <tr>
                    <td style="{admin_td_alt}"><strong>LLM Calls</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.llm_calls_made}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>LLM Tokens Used</strong></td>
                    <td style="{admin_td}">{diagnostics.llm_tokens_used:,}</td>
                </tr>
                <tr>
                    <td style="{admin_td_alt}"><strong>Fact Checks</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.fact_checks_performed}</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Temporal Patterns</strong></td>
                    <td style="{admin_td}">{diagnostics.temporal_patterns_detected}</td>
                </tr>
            </table>
        </div>

        <!-- Performance -->
        <div style="padding: 30px 40px; border-bottom: 1px solid #eee;">
            <div style="{admin_section_header}">
                Performance Breakdown
            </div>

            <table width="100%" cellpadding="8" cellspacing="0" style="font-size: 14px; font-family: monospace; border-collapse: collapse;">
                <tr>
                    <td style="{admin_td_alt}"><strong>Step 1: Broad Scraping</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.time_step1_scraping:.1f}s</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Step 2: Trend Analysis</strong></td>
                    <td style="{admin_td}">{diagnostics.time_step2_analysis:.1f}s</td>
                </tr>
                <tr>
                    <td style="{admin_td_alt}"><strong>Step 3: Deep Dive</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.time_step3_deep_dive:.1f}s</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Step 4: LLM Analysis</strong></td>
                    <td style="{admin_td}">{diagnostics.time_step4_llm:.1f}s</td>
                </tr>
                <tr>
                    <td style="{admin_td_alt}"><strong>Step 5: Email</strong></td>
                    <td style="{admin_td_alt}">{diagnostics.time_step5_email:.1f}s</td>
                </tr>
                <tr>
                    <td style="{admin_td}"><strong>Step 6: Storage</strong></td>
                    <td style="{admin_td}">{diagnostics.time_step6_storage:.1f}s</td>
                </tr>
            </table>
        </div>

        {errors_html}
        {warnings_html}

        <!-- Footer -->
        <div style="border-top: 1px solid #eee; padding: 30px 40px; font-size: 13px; color: #999; background-color: #fafafa;">
            <p style="margin: 0; font-family: monospace; font-size: 11px; color: #999; letter-spacing: 1px;">
                Jafar Admin Diagnostics | {today}
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def send_ml_email(
        self,
        ml_results: dict,
        recipients: list[str],
    ) -> bool:
        """
        Send a dedicated ML analysis email to admin recipients.

        This is a separate email from the news digest, containing detailed
        ML insights: high-signal accounts with evidence tweets, bot suspects
        with suspicious examples, cluster analysis, and model health.

        Args:
            ml_results: Dict with bot_suspects, top_accounts, bot_judgments,
                ml_evaluation, cluster_distribution, total_accounts_analyzed,
                total_tweets_analyzed. Accounts should include sample_tweets.
            recipients: Admin email addresses.

        Returns:
            True if email sent successfully.
        """
        today = datetime.now().strftime("%B %d, %Y %H:%M")

        subject = f"Jafar ML Intelligence Report - {today}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.email_from_name} <{self.config.email_from}>"
        msg["To"] = self.config.email_from

        html_content = self._generate_ml_email_html(ml_results)
        text_content = self._generate_ml_email_plain_text(ml_results)

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        logger.info(f"Sending ML diagnostics email to {len(recipients)} recipient(s)")

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

            logger.info("ML diagnostics email sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send ML diagnostics email: {e}")
            return False

    def _generate_ml_email_html(self, ml_results: dict) -> str:
        """Generate comprehensive HTML for the dedicated ML analysis email."""
        today = datetime.now().strftime("%B %d, %Y")

        bot_suspects = ml_results.get("bot_suspects", [])
        top_accounts = ml_results.get("top_accounts", [])
        bot_judgments = ml_results.get("bot_judgments", [])
        ml_evaluation = ml_results.get("ml_evaluation")
        cluster_dist = ml_results.get("cluster_distribution", {})
        total_accounts = ml_results.get("total_accounts_analyzed", 0)
        total_tweets = ml_results.get("total_tweets_analyzed", 0)

        judgment_map = {j["username"]: j for j in bot_judgments}

        # Shared styles
        th = (
            "padding: 10px 12px; border-bottom: 2px solid #000;"
            " text-align: left; font-family: monospace; font-size: 12px;"
            " letter-spacing: 1px; background-color: #fafafa; color: #111;"
        )
        td = "padding: 10px 12px; border-bottom: 1px solid #eee; font-family: monospace; font-size: 13px;"
        td_alt = "padding: 10px 12px; border-bottom: 1px solid #eee; font-family: monospace; font-size: 13px; background-color: #fafafa;"
        section_hdr = "font-family: monospace; font-size: 11px; color: #888; margin-bottom: 15px; letter-spacing: 1px;"
        tweet_style = (
            "padding: 10px 14px; margin: 6px 0; border-left: 3px solid #ddd;"
            " font-size: 13px; line-height: 1.5; color: #333;"
            " font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;"
            " background-color: #fafafa;"
        )
        engagement_style = "font-size: 11px; color: #999; margin-top: 4px; font-family: monospace;"

        sections = []

        # --- Overview Stats ---
        sections.append(f"""
        <div style="padding: 20px 40px; border-bottom: 1px solid #000; background-color: #fcfcfc;">
            <table width="100%" cellpadding="0" cellspacing="0" border="0" style="font-family: monospace; font-size: 13px;">
                <tr>
                    <td style="padding: 8px 0; border-right: 1px solid #ddd; width: 25%; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold;">{total_tweets:,}</div>
                        <div style="color: #888; font-size: 11px;">tweets in DB</div>
                    </td>
                    <td style="padding: 8px 0; border-right: 1px solid #ddd; width: 25%; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold;">{total_accounts}</div>
                        <div style="color: #888; font-size: 11px;">accounts analyzed</div>
                    </td>
                    <td style="padding: 8px 0; border-right: 1px solid #ddd; width: 25%; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold;">{len(bot_suspects)}</div>
                        <div style="color: #888; font-size: 11px;">bot suspects</div>
                    </td>
                    <td style="padding: 8px 0; width: 25%; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold;">{len(top_accounts)}</div>
                        <div style="color: #888; font-size: 11px;">high-signal accounts</div>
                    </td>
                </tr>
            </table>
        </div>
        """)

        # --- Cluster Distribution ---
        if cluster_dist:
            cluster_rows = []
            cluster_colors = {
                "high_signal": "#2a7a2a",
                "news_aggregator": "#666",
                "casual": "#888",
                "low_activity": "#aaa",
                "unclustered": "#ccc",
            }
            total_clustered = sum(cluster_dist.values()) or 1
            for label in ["high_signal", "news_aggregator", "casual", "low_activity", "unclustered"]:
                count = cluster_dist.get(label, 0)
                if count == 0:
                    continue
                pct = count / total_clustered * 100
                color = cluster_colors.get(label, "#888")
                bar_width = max(int(pct * 3), 1)
                cluster_rows.append(f"""
                <tr>
                    <td style="{td} width: 140px;">{label}</td>
                    <td style="{td} width: 60px; text-align: right; font-weight: bold;">{count}</td>
                    <td style="{td}">
                        <div style="background-color: {color}; height: 14px; width: {bar_width}px; display: inline-block;"></div>
                        <span style="font-size: 11px; color: #999; margin-left: 6px;">{pct:.0f}%</span>
                    </td>
                </tr>
                """)

            sections.append(f"""
            <div style="padding: 20px 40px; border-bottom: 1px solid #eee;">
                <div style="{section_hdr}">Account Cluster Distribution</div>
                <p style="font-size: 13px; color: #666; margin: 0 0 12px 0;">
                    KMeans clustering of {total_accounts} accounts by behavioral features.
                </p>
                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                    {"".join(cluster_rows)}
                </table>
            </div>
            """)

        # --- High-Signal Accounts with Evidence ---
        if top_accounts:
            account_blocks = []
            for i, account in enumerate(top_accounts[:10]):
                username = account.get("username", "unknown")
                score = account.get("signal_score", 0.0)
                cluster = account.get("cluster_label", "unknown")
                tweet_count = account.get("tweet_count", 0)
                features = account.get("features", {})
                sample_tweets = account.get("sample_tweets", [])

                # Key feature highlights
                highlights = []
                epv = features.get("engagement_per_view", 0)
                if epv > 0:
                    highlights.append(f"engagement/view: {epv:.3f}")
                ucr = features.get("unique_content_ratio", 0)
                if ucr > 0:
                    highlights.append(f"unique content: {ucr:.0%}")
                url_r = features.get("url_ratio", 0)
                if url_r > 0.5:
                    highlights.append(f"URL ratio: {url_r:.0%} (aggregator)")
                elif url_r < 0.1:
                    highlights.append("original content (low URL ratio)")
                tr = features.get("trend_coverage_ratio", 0)
                if tr > 0:
                    highlights.append(f"trend coverage: {tr:.0%}")

                feature_text = " | ".join(highlights[:4]) if highlights else "—"

                # Build tweet samples
                tweet_html = ""
                for tw in sample_tweets[:3]:
                    content = tw.get("content", "")[:280]
                    likes = tw.get("likes", 0)
                    rts = tw.get("retweets", 0)
                    replies = tw.get("replies", 0)
                    tweet_html += f"""
                    <div style="{tweet_style}">
                        {content}
                        <div style="{engagement_style}">{likes:,} likes | {rts:,} RTs | {replies:,} replies</div>
                    </div>
                    """

                bg = "#fff" if i % 2 == 0 else "#fcfcfc"
                account_blocks.append(f"""
                <div style="padding: 16px 40px; background-color: {bg}; border-bottom: 1px solid #eee;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0">
                        <tr>
                            <td>
                                <strong style="font-size: 15px;">@{username}</strong>
                                <span style="font-family: monospace; font-size: 12px; color: #888; margin-left: 10px;">
                                    signal: <strong style="color: #2a7a2a;">{score:.2f}</strong> | cluster: {cluster} | {tweet_count} tweets
                                </span>
                            </td>
                        </tr>
                    </table>
                    <div style="font-size: 12px; color: #666; margin: 4px 0 8px 0; font-family: monospace;">{feature_text}</div>
                    {tweet_html if tweet_html else '<div style="font-size: 12px; color: #aaa; font-style: italic;">No sample tweets available</div>'}
                </div>
                """)

            sections.append(f"""
            <div style="padding: 25px 40px 10px 40px; border-top: 1px solid #000;">
                <h2 style="margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">High-Signal Accounts</h2>
                <p style="font-size: 13px; color: #666; margin: 6px 0 0 0;">
                    Accounts ranked by weighted signal score: engagement quality, content originality, trend coverage, and recurrence.
                    These accounts provide the most valuable signal for economic trend detection.
                </p>
            </div>
            {"".join(account_blocks)}
            """)

        # --- Bot Suspects with Evidence ---
        if bot_suspects:
            suspect_blocks = []
            for i, suspect in enumerate(bot_suspects[:10]):
                username = suspect.get("username", "unknown")
                bot_score = suspect.get("bot_score", 0.0)
                features = suspect.get("features", {})
                sample_tweets = suspect.get("sample_tweets", [])
                judgment = judgment_map.get(username, {})
                llm_verdict = judgment.get("classification", "not judged")
                confidence = judgment.get("confidence", 0.0)
                reasoning = judgment.get("reasoning", "—")

                # Key bot signals
                signals = []
                dup = features.get("duplicate_content_ratio", 0)
                if dup > 0.3:
                    signals.append(f"duplicate content: {dup:.0%}")
                night = features.get("night_ratio", 0)
                if night > 0.5:
                    signals.append(f"night posting: {night:.0%}")
                zero_eng = features.get("zero_engagement_ratio", 0)
                if zero_eng > 0.5:
                    signals.append(f"zero engagement: {zero_eng:.0%}")
                cv = features.get("coefficient_of_variation", 0)
                if 0 < cv < 0.3:
                    signals.append(f"suspiciously regular timing (CV: {cv:.2f})")
                ht = features.get("avg_hashtags_per_tweet", 0)
                if ht > 3:
                    signals.append(f"hashtag spam: {ht:.1f}/tweet")

                signals_text = " | ".join(signals[:4]) if signals else "anomalous behavioral pattern"

                score_color = "#c00" if bot_score >= 0.8 else "#c80" if bot_score >= 0.7 else "#333"

                # Build tweet samples
                tweet_html = ""
                for tw in sample_tweets[:3]:
                    content = tw.get("content", "")[:280]
                    likes = tw.get("likes", 0)
                    rts = tw.get("retweets", 0)
                    tweet_html += f"""
                    <div style="{tweet_style} border-left-color: #c00;">
                        {content}
                        <div style="{engagement_style}">{likes:,} likes | {rts:,} RTs</div>
                    </div>
                    """

                bg = "#fff" if i % 2 == 0 else "#fcfcfc"
                suspect_blocks.append(f"""
                <div style="padding: 16px 40px; background-color: {bg}; border-bottom: 1px solid #eee;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0">
                        <tr>
                            <td>
                                <strong style="font-size: 15px;">@{username}</strong>
                                <span style="font-family: monospace; font-size: 12px; color: #888; margin-left: 10px;">
                                    ML: <strong style="color: {score_color};">{bot_score:.2f}</strong> |
                                    LLM: <strong>{llm_verdict}</strong> (conf: {confidence:.2f})
                                </span>
                            </td>
                        </tr>
                    </table>
                    <div style="font-size: 12px; color: #666; margin: 4px 0 4px 0; font-family: monospace;">{signals_text}</div>
                    <div style="font-size: 12px; color: #555; margin: 0 0 8px 0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
                        <em>LLM reasoning:</em> {reasoning}
                    </div>
                    {tweet_html if tweet_html else '<div style="font-size: 12px; color: #aaa; font-style: italic;">No sample tweets available</div>'}
                </div>
                """)

            sections.append(f"""
            <div style="padding: 25px 40px 10px 40px; border-top: 1px solid #000;">
                <h2 style="margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.5px; color: #900;">Bot Suspects</h2>
                <p style="font-size: 13px; color: #666; margin: 6px 0 0 0;">
                    Accounts flagged by IsolationForest anomaly detection (score &gt; 0.7), then validated by Gemini LLM review.
                    Higher ML score = more anomalous behavioral pattern.
                </p>
            </div>
            {"".join(suspect_blocks)}
            """)

        # --- Model Health ---
        if ml_evaluation:
            agreement = ml_evaluation.get("agreement_rate", 0.0)
            precision = ml_evaluation.get("precision", 0.0)
            recall = ml_evaluation.get("recall", 0.0)
            f1 = ml_evaluation.get("f1_score", 0.0)
            total_eval = ml_evaluation.get("total_evaluated", 0)
            cm = ml_evaluation.get("confusion_matrix", {})

            agreement_color = "#2a7a2a" if agreement >= 0.8 else "#c80" if agreement >= 0.6 else "#c00"

            sections.append(f"""
            <div style="padding: 25px 40px 10px 40px; border-top: 1px solid #000;">
                <h2 style="margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">Model Health</h2>
                <p style="font-size: 13px; color: #666; margin: 6px 0 15px 0;">
                    ML (IsolationForest) vs LLM (Gemini) agreement on {total_eval} sampled accounts.
                </p>
            </div>
            <div style="padding: 10px 40px 20px 40px; border-bottom: 1px solid #eee;">
                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; max-width: 500px;">
                    <tr>
                        <td style="{td_alt}"><strong>Agreement Rate</strong></td>
                        <td style="{td_alt} color: {agreement_color}; font-weight: bold; font-size: 16px;">{agreement:.0%}</td>
                    </tr>
                    <tr>
                        <td style="{td}"><strong>Precision</strong> <span style="color: #999; font-size: 11px;">(% of ML bot flags confirmed by LLM)</span></td>
                        <td style="{td} font-weight: bold;">{precision:.2f}</td>
                    </tr>
                    <tr>
                        <td style="{td_alt}"><strong>Recall</strong> <span style="color: #999; font-size: 11px;">(% of LLM bots caught by ML)</span></td>
                        <td style="{td_alt} font-weight: bold;">{recall:.2f}</td>
                    </tr>
                    <tr>
                        <td style="{td}"><strong>F1 Score</strong></td>
                        <td style="{td} font-weight: bold;">{f1:.2f}</td>
                    </tr>
                </table>
            </div>
            """)

            # Confusion matrix
            if cm:
                sections.append(f"""
                <div style="padding: 15px 40px 20px 40px; border-bottom: 1px solid #eee;">
                    <div style="{section_hdr}">Confusion Matrix</div>
                    <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; font-family: monospace; font-size: 13px;">
                        <tr>
                            <td style="padding: 8px 16px; border: 1px solid #ddd; background: #fafafa;"></td>
                            <td style="padding: 8px 16px; border: 1px solid #ddd; background: #fafafa; text-align: center;"><strong>LLM: Bot</strong></td>
                            <td style="padding: 8px 16px; border: 1px solid #ddd; background: #fafafa; text-align: center;"><strong>LLM: Human</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 16px; border: 1px solid #ddd; background: #fafafa;"><strong>ML: Bot</strong></td>
                            <td style="padding:8px 16px; border:1px solid #ddd; text-align:center; background:#e8f5e9; font-weight:bold;">{cm.get("tp", 0)}</td>
                            <td style="padding:8px 16px; border:1px solid #ddd; text-align:center; background:#fce4ec;">{cm.get("fp", 0)}</td>
                        </tr>
                        <tr>
                            <td style="padding:8px 16px; border:1px solid #ddd; background:#fafafa;"><strong>ML: Human</strong></td>
                            <td style="padding:8px 16px; border:1px solid #ddd; text-align:center; background:#fce4ec;">{cm.get("fn", 0)}</td>
                            <td style="padding:8px 16px; border:1px solid #ddd; text-align:center; background:#e8f5e9; font-weight:bold;">{cm.get("tn", 0)}</td>
                        </tr>
                    </table>
                </div>
                """)

            # Disagreements
            disagreements = ml_evaluation.get("disagreements", [])
            if disagreements:
                dis_rows = []
                for i, d in enumerate(disagreements[:10]):
                    username = d.get("username", "unknown")
                    ml_score = d.get("ml_bot_score", d.get("ml_score", d.get("bot_score", 0.0)))
                    llm_class = d.get("llm_classification", d.get("classification", "—"))
                    reason = d.get("llm_reasoning", d.get("reason", d.get("reasoning", "—")))
                    dtype = d.get("type", "")

                    row_bg = td_alt if i % 2 == 0 else td
                    type_badge = f'<span style="font-size: 10px; color: #c80;">({dtype.replace("_", " ")})</span>' if dtype else ""

                    dis_rows.append(f"""
                    <tr>
                        <td style="{row_bg}">@{username} {type_badge}</td>
                        <td style="{row_bg} font-weight: bold;">{ml_score:.2f}</td>
                        <td style="{row_bg}">{llm_class}</td>
                        <td style="{row_bg} font-size: 11px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #666;">{reason}</td>
                    </tr>
                    """)

                sections.append(f"""
                <div style="padding: 20px 40px; border-bottom: 1px solid #eee;">
                    <div style="{section_hdr}; color: #c80;">ML/LLM Disagreements — Review Recommended</div>
                    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                        <tr>
                            <th style="{th}">Account</th>
                            <th style="{th}">ML Score</th>
                            <th style="{th}">LLM Verdict</th>
                            <th style="{th}">Reasoning</th>
                        </tr>
                        {"".join(dis_rows)}
                    </table>
                </div>
                """)

        # --- Assemble full email ---
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jafar ML Report - {today}</title>
</head>
<body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #111; line-height: 1.6; margin: 0; padding: 40px 20px; background-color: #f6f6f6;">
    <div style="max-width: 850px; margin: 0 auto; border: 1px solid #000; background-color: #fff; box-shadow: 10px 10px 0px #000;">

        <!-- Header -->
        <div style="border-bottom: 1px solid #333; padding: 30px 40px; background-color: #0a0a0a; color: #ffffff;">
            <div style="font-family: monospace; font-size: 13px; letter-spacing: 2px; margin-bottom: 10px; color: #888888;">
                Jafar Intelligence System
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td align="left">
                        <h1 style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -1px; color: #ffffff;">
                            ML Intelligence Report
                        </h1>
                    </td>
                    <td align="right" style="font-family: monospace; font-size: 16px; color: #888888;">
                        {today}
                    </td>
                </tr>
            </table>
        </div>

        {"".join(sections)}

        <!-- Footer -->
        <div style="border-top: 1px solid #ddd; padding: 20px 40px; font-size: 12px; color: #999; background-color: #fafafa;">
            <p style="margin: 0; font-family: monospace; font-size: 11px; letter-spacing: 1px;">
                Jafar ML Intelligence | Internal use only | {today}
            </p>
        </div>
    </div>
</body>
</html>
"""

    def _generate_ml_email_plain_text(self, ml_results: dict) -> str:
        """Generate plain text for the ML analysis email."""
        top_accounts = ml_results.get("top_accounts", [])
        bot_suspects = ml_results.get("bot_suspects", [])
        bot_judgments = ml_results.get("bot_judgments", [])
        ml_evaluation = ml_results.get("ml_evaluation")
        total_accounts = ml_results.get("total_accounts_analyzed", 0)
        total_tweets = ml_results.get("total_tweets_analyzed", 0)
        cluster_dist = ml_results.get("cluster_distribution", {})

        judgment_map = {j["username"]: j for j in bot_judgments}

        lines = [
            "Jafar ML Intelligence Report",
            "=" * 60,
            "",
            f"Tweets in DB: {total_tweets:,}",
            f"Accounts analyzed: {total_accounts}",
            f"Bot suspects: {len(bot_suspects)}",
            f"High-signal accounts: {len(top_accounts)}",
        ]

        if cluster_dist:
            lines.append("")
            lines.append("Cluster Distribution:")
            for label, count in sorted(cluster_dist.items(), key=lambda x: -x[1]):
                lines.append(f"  {label}: {count}")

        if top_accounts:
            lines.append("")
            lines.append("-" * 60)
            lines.append("HIGH-SIGNAL ACCOUNTS")
            lines.append("-" * 60)
            for account in top_accounts[:10]:
                username = account.get("username", "unknown")
                score = account.get("signal_score", 0.0)
                cluster = account.get("cluster_label", "unknown")
                tweet_count = account.get("tweet_count", 0)
                lines.append("")
                lines.append(f"@{username} (signal: {score:.2f}, cluster: {cluster}, tweets: {tweet_count})")
                for tw in account.get("sample_tweets", [])[:3]:
                    content = tw.get("content", "")[:200]
                    likes = tw.get("likes", 0)
                    lines.append(f"  > {content}")
                    lines.append(f"    [{likes} likes]")

        if bot_suspects:
            lines.append("")
            lines.append("-" * 60)
            lines.append("BOT SUSPECTS")
            lines.append("-" * 60)
            for suspect in bot_suspects[:10]:
                username = suspect.get("username", "unknown")
                bot_score = suspect.get("bot_score", 0.0)
                judgment = judgment_map.get(username, {})
                llm_verdict = judgment.get("classification", "not judged")
                reasoning = judgment.get("reasoning", "—")
                lines.append("")
                lines.append(f"@{username} (ML: {bot_score:.2f}, LLM: {llm_verdict})")
                lines.append(f"  Reasoning: {reasoning}")
                for tw in suspect.get("sample_tweets", [])[:2]:
                    content = tw.get("content", "")[:200]
                    lines.append(f"  > {content}")

        if ml_evaluation:
            lines.append("")
            lines.append("-" * 60)
            lines.append("MODEL HEALTH")
            lines.append("-" * 60)
            lines.append(f"Agreement: {ml_evaluation.get('agreement_rate', 0):.0%}")
            lines.append(f"Precision: {ml_evaluation.get('precision', 0):.2f}")
            lines.append(f"Recall: {ml_evaluation.get('recall', 0):.2f}")
            lines.append(f"F1: {ml_evaluation.get('f1_score', 0):.2f}")

        return "\n".join(lines)

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

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.email_from_name} <{self.config.email_from}>"
        # Using the sender's email in 'To' and putting everyone else in BCC
        msg["To"] = self.config.email_from

        html_content = self._generate_admin_html_report(diagnostics, alert_reason)

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
