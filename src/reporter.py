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
    ) -> str:
        """
        Generate a clean, minimalist HTML email.

        Args:
            report_content: The LLM-generated analysis.
            trends: List of trending topics analyzed.
            tweet_count: Total number of tweets analyzed.
            provider_info: LLM provider used for analysis.
            signal_strength: Signal strength rating (high/medium/low/none).

        Returns:
            HTML formatted email body.
        """
        today = datetime.now().strftime("%B %d, %Y")

        # Minimalist signal indicators (Monochrome/High Contrast)
        signal_styles = {
            "high": "border-left: 4px solid #000000; background-color: #f0f0f0;",
            "medium": "border-left: 4px solid #666666; background-color: #f8f8f8;",
            "low": "border-left: 4px solid #bbbbbb; background-color: #ffffff;",
            "none": "border-left: 4px solid #eeeeee; background-color: #ffffff; color: #888888;",
        }
        signal_style = signal_styles.get(signal_strength, signal_styles["low"])
        signal_text_upper = signal_strength.upper()

        # Format trends as simple tags
        trend_tags = " ".join(
            f'<span style="border: 1px solid #ddd; padding: 2px 8px; font-size: 12px; margin-right: 6px; display: inline-block; font-family: monospace;">{trend}</span>'
            for trend in trends
        )

        # HTML Formatting
        import re
        formatted_content = report_content
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
        formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
        formatted_content = formatted_content.replace('\n\n', '</p><p style="margin: 16px 0;">')
        formatted_content = formatted_content.replace('\n', '<br>')

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jafar Digest - {today}</title>
</head>
<body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #111; line-height: 1.5; margin: 0; padding: 20px; background-color: #ffffff;">
    <div style="max-width: 650px; margin: 0 auto; border: 1px solid #000;">
        
        <!-- Header -->
        <div style="border-bottom: 1px solid #000; padding: 20px; background-color: #000; color: #fff;">
            <div style="font-family: monospace; font-size: 12px; letter-spacing: 1px; margin-bottom: 8px; opacity: 0.8;">
                JAFAR INTELLIGENCE SYSTEM
            </div>
            <h1 style="margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                Market Digest
            </h1>
            <div style="margin-top: 5px; font-size: 14px; opacity: 0.9;">
                {today}
            </div>
        </div>

        <!-- Signal Banner -->
        <div style="padding: 15px 20px; {signal_style} border-bottom: 1px solid #eee;">
            <span style="font-family: monospace; font-weight: bold; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">
                SIGNAL STRENGTH: {signal_text_upper}
            </span>
        </div>

        <!-- Metadata Grid -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid #eee; font-family: monospace; font-size: 12px;">
            <div style="padding: 10px 20px; border-right: 1px solid #eee;">
                ANALYZED: {tweet_count} TWEETS
            </div>
            <div style="padding: 10px 20px;">
                DETECTED: {len(trends)} TRENDS
            </div>
        </div>

        <!-- Trends -->
        <div style="padding: 15px 20px; border-bottom: 1px solid #eee; background-color: #fcfcfc;">
            <div style="font-family: monospace; font-size: 10px; color: #666; margin-bottom: 8px; text-transform: uppercase;">
                Active Topics
            </div>
            <div>
                {trend_tags}
            </div>
        </div>

        <!-- Main Body -->
        <div style="padding: 30px 20px;">
            <div style="font-size: 16px; color: #222; line-height: 1.6;">
                {formatted_content}
            </div>
        </div>

        <!-- Footer -->
        <div style="border-top: 1px solid #000; padding: 20px; font-size: 12px; color: #666; background-color: #f9f9f9;">
            <p style="margin: 0 0 10px 0;">
                <strong>Jafar</strong> &mdash; The villain to BlackRock's Aladdin.
            </p>
            <p style="margin: 0 0 10px 0; font-style: italic;">
                "The only thing BlackRock manages better than assets is their conflict of interest."
            </p>
            <p style="margin: 0; font-family: monospace; font-size: 10px; color: #999;">
                POWERED BY {provider_info.upper()} | NOT FINANCIAL ADVICE
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
    ) -> bool:
        """
        Send the economic digest via email.

        Args:
            report_content: The LLM-generated analysis.
            trends: List of trending topics analyzed.
            tweet_count: Total number of tweets analyzed.
            provider_info: Description of LLM used.
            signal_strength: Signal strength rating (high/medium/low/none).

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
            report_content, trends, tweet_count, provider_info, signal_strength
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
