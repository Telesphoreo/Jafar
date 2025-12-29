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
        signal_text_upper = signal_strength.upper()

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
            <div style="font-family: monospace; font-size: 13px; letter-spacing: 2px; margin-bottom: 10px; opacity: 0.8; text-transform: uppercase;">
                JAFAR INTELLIGENCE SYSTEM
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td align="left">
                        <h1 style="margin: 0; font-size: 32px; font-weight: 700; letter-spacing: -1px; text-transform: uppercase;">
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
            <span style="font-family: monospace; font-weight: bold; font-size: 14px; text-transform: uppercase; letter-spacing: 2px;">
                SIGNAL STRENGTH: {signal_text_upper}
            </span>
        </div>

        <!-- Metadata Row -->
        <div style="display: flex; border-bottom: 1px solid #000; font-family: monospace; font-size: 13px; background-color: #fcfcfc;">
            <div style="padding: 15px 40px; border-right: 1px solid #000; flex: 1;">
                ANALYZED: <strong>{tweet_count}</strong> TWEETS
            </div>
            <div style="padding: 15px 40px; flex: 1;">
                DETECTED: <strong>{len(trends)}</strong> TRENDS
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
                        <p style="margin: 0 0 15px 0; font-style: italic; font-size: 14px; color: #000;">
                            "The only thing BlackRock manages better than assets is their conflict of interest."
                        </p>
                        <p style="margin: 0; font-family: monospace; font-size: 11px; color: #999; text-transform: uppercase; letter-spacing: 1px;">
                            ENGINE: {provider_info.upper()} | NOT FINANCIAL ADVICE | {today}
                        </p>
                    </td>
                </tr>
            </table>
        </div>
    </div>
    <div style="max-width: 850px; margin: 20px auto 0; text-align: center; font-family: monospace; font-size: 10px; color: #bbb; text-transform: uppercase; letter-spacing: 2px;">
        End of Transmission
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
JAFAR INTELLIGENCE SYSTEM
Market Digest - {today}
--------------------------------------------------------------------------------

SIGNAL: DETECTED
INPUTS: {tweet_count} tweets
TOPICS: {trends_str}

--------------------------------------------------------------------------------

{report_content}

--------------------------------------------------------------------------------
"The only thing BlackRock manages better than assets is their conflict of interest."

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
            "high": "[HIGH SIGNAL]",
            "medium": "[MEDIUM]",
            "low": "",
            "none": "[QUIET DAY]",
        }.get(signal_strength, "")

        subject = f"{signal_prefix} Jafar Market Digest - {today}".strip()

        # Create message container
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.config.email_from
        msg["To"] = ", ".join(self.config.email_to)

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

        logger.info(f"Sending email to {len(self.config.email_to)} recipient(s)")
        logger.info(f"SMTP Server: {self.config.host}:{self.config.port} (TLS: {self.config.use_tls})")
        logger.info(f"From: {self.config.email_from}")
        logger.info(f"To: {', '.join(self.config.email_to)}")
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
            server.sendmail(
                self.config.email_from,
                self.config.email_to,
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


def create_reporter_from_config(
    host: str,
    port: int,
    username: str,
    password: str,
    use_tls: bool,
    email_from: str,
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
        email_to=email_to,
    )
    return EmailReporter(config)
