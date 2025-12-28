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

logger = logging.getLogger("twitter_sentiment.reporter")


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
    ) -> str:
        """
        Generate a nicely formatted HTML email from the report content.

        Args:
            report_content: The LLM-generated analysis.
            trends: List of trending topics analyzed.
            tweet_count: Total number of tweets analyzed.
            provider_info: LLM provider used for analysis.

        Returns:
            HTML formatted email body.
        """
        today = datetime.now().strftime("%B %d, %Y")

        # Format trends as badges
        trend_badges = " ".join(
            f'<span style="background-color: #e3f2fd; color: #1976d2; '
            f'padding: 4px 12px; border-radius: 16px; margin-right: 8px; '
            f'font-size: 14px; display: inline-block; margin-bottom: 4px;">{trend}</span>'
            for trend in trends
        )

        # Convert markdown-style formatting to HTML
        formatted_content = report_content
        # Convert **bold** to <strong>
        import re
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
        # Convert *italic* to <em>
        formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
        # Convert newlines to <br> for proper HTML rendering
        formatted_content = formatted_content.replace('\n\n', '</p><p style="margin: 16px 0; line-height: 1.6;">')
        formatted_content = formatted_content.replace('\n', '<br>')

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Economic Sentiment Digest - {today}</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
             background-color: #f5f5f5; margin: 0; padding: 20px;">
    <div style="max-width: 700px; margin: 0 auto; background-color: #ffffff;
                border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

        <!-- Header -->
        <div style="background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
                    color: white; padding: 30px; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 28px; font-weight: 600;">
                Economic Sentiment Digest
            </h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;">
                {today}
            </p>
        </div>

        <!-- Stats Bar -->
        <div style="background-color: #f8f9fa; padding: 15px 30px;
                    border-bottom: 1px solid #e0e0e0; display: flex; gap: 30px;">
            <div>
                <span style="font-size: 24px; font-weight: 600; color: #1976d2;">{tweet_count}</span>
                <span style="color: #666; font-size: 14px; margin-left: 5px;">tweets analyzed</span>
            </div>
            <div>
                <span style="font-size: 24px; font-weight: 600; color: #1976d2;">{len(trends)}</span>
                <span style="color: #666; font-size: 14px; margin-left: 5px;">trends identified</span>
            </div>
        </div>

        <!-- Trending Topics -->
        <div style="padding: 20px 30px; border-bottom: 1px solid #e0e0e0;">
            <h2 style="margin: 0 0 12px 0; font-size: 14px; text-transform: uppercase;
                       color: #666; letter-spacing: 1px;">
                Today's Trending Topics
            </h2>
            <div style="line-height: 2.2;">
                {trend_badges}
            </div>
        </div>

        <!-- Main Content -->
        <div style="padding: 30px;">
            <h2 style="margin: 0 0 20px 0; font-size: 20px; color: #333;">
                Market Sentiment Analysis
            </h2>
            <div style="color: #444; font-size: 15px;">
                <p style="margin: 16px 0; line-height: 1.6;">
                    {formatted_content}
                </p>
            </div>
        </div>

        <!-- Footer -->
        <div style="background-color: #f8f9fa; padding: 20px 30px;
                    border-radius: 0 0 8px 8px; border-top: 1px solid #e0e0e0;">
            <p style="margin: 0; color: #888; font-size: 12px;">
                This digest was automatically generated using Twitter/X sentiment analysis
                and AI-powered summarization ({provider_info}).
            </p>
            <p style="margin: 10px 0 0 0; color: #888; font-size: 12px;">
                Disclaimer: This analysis is for informational purposes only and should not
                be considered financial advice.
            </p>
        </div>
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
ECONOMIC SENTIMENT DIGEST
{today}

================================================================================

STATS: {tweet_count} tweets analyzed | {len(trends)} trends identified

TRENDING TOPICS: {trends_str}

================================================================================

MARKET SENTIMENT ANALYSIS

{report_content}

================================================================================

This digest was automatically generated using Twitter/X sentiment analysis
and AI-powered summarization.

Disclaimer: This analysis is for informational purposes only and should not
be considered financial advice.
"""

    def send_email(
        self,
        report_content: str,
        trends: list[str],
        tweet_count: int,
        provider_info: str = "AI",
    ) -> bool:
        """
        Send the economic digest via email.

        Args:
            report_content: The LLM-generated analysis.
            trends: List of trending topics analyzed.
            tweet_count: Total number of tweets analyzed.
            provider_info: Description of LLM used.

        Returns:
            True if email was sent successfully.
        """
        today = datetime.now().strftime("%B %d, %Y")

        # Create message container
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Economic Sentiment Digest - {today}"
        msg["From"] = self.config.email_from
        msg["To"] = ", ".join(self.config.email_to)

        # Generate both plain text and HTML versions
        text_content = self._generate_plain_text(report_content, trends, tweet_count)
        html_content = self._generate_html_report(
            report_content, trends, tweet_count, provider_info
        )

        # Attach both versions (email clients will choose the best one)
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)

        logger.info(f"Sending email to {len(self.config.email_to)} recipient(s)")

        try:
            # Connect to SMTP server
            if self.config.use_tls:
                server = smtplib.SMTP(self.config.host, self.config.port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.config.host, self.config.port)

            # Login and send
            server.login(self.config.username, self.config.password)
            server.sendmail(
                self.config.email_from,
                self.config.email_to,
                msg.as_string(),
            )
            server.quit()

            logger.info("Email sent successfully")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
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
