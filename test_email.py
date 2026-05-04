"""
Test Email Script & SMTP Diagnoser.

Sends a test email using Jafar's actual HTML template with sample data,
then falls back to SMTP diagnostics if the connection fails.
"""

import sys
import logging
import smtplib
import socket
import time
from src.config import config
from src.reporter import EmailReporter, EmailConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jafar.diagnose")


# Sample data that exercises every section of the HTML template
SAMPLE_REPORT_CONTENT = """\
**Consumer Spending Shift Detected**: Social discussion volume around "too expensive" \
has surged 340% week-over-week, concentrated in grocery and dining categories. \
Multiple organic posts reference egg prices exceeding $8/dozen in urban markets, \
consistent with USDA wholesale data showing a 22% monthly increase.

**GPU Supply Crunch**: Mentions of RTX 5090 "sold out" and "scalper pricing" have \
crossed the anomaly threshold. Verified via Best Buy and Newegg — different SKUs show \
30-45% markups above MSRP on secondary markets. This mirrors the 2020-2021 GPU \
shortage pattern, though current volumes are lower.

**Uranium Spot Price Movement**: Chatter around uranium spiking was *partially \
exaggerated*. Spot price moved +3.2% to $78.40/lb (verified via Numerco), \
but social posts claimed "parabolic" moves. The Agent flagged this as \
**Overstated** after cross-referencing market data.

Nothing actionable detected in labor markets or housing this cycle. \
Seasonal hiring posts are within normal range for March."""

SAMPLE_TRENDS = [
    "Egg Prices",
    "RTX 5090 Shortage",
    "Uranium Spot",
    "Grocery Inflation",
    "GPU Scalping",
]

SAMPLE_ML_RESULTS = {
    "bot_suspects": [
        {"username": "crypto_signals_99", "bot_score": 0.92},
        {"username": "buygold_now", "bot_score": 0.85},
        {"username": "breaking_mkt_news", "bot_score": 0.71},
        {"username": "real_trader_42", "bot_score": 0.45},
    ],
    "top_accounts": [
        {"username": "grocery_watchdog", "signal_score": 0.94, "cluster": "consumer-staples", "tweet_count": 12},
        {"username": "chip_analyst", "signal_score": 0.88, "cluster": "semiconductors", "tweet_count": 8},
        {"username": "energy_insider", "signal_score": 0.76, "cluster": "commodities", "tweet_count": 5},
        {"username": "midwest_shopper", "signal_score": 0.71, "cluster": "consumer-staples", "tweet_count": 15},
    ],
    "bot_judgments": [
        {
            "username": "crypto_signals_99",
            "classification": "Bot",
            "confidence": 0.95,
            "reasoning": "Account posts identical phrasing across multiple tokens at regular 30-minute intervals. No organic engagement pattern.",
        },
        {
            "username": "buygold_now",
            "classification": "Bot",
            "confidence": 0.88,
            "reasoning": "Repetitive promotional content with affiliate links. Created within the last 30 days with abnormal follower growth.",
        },
        {
            "username": "breaking_mkt_news",
            "classification": "Likely Bot",
            "confidence": 0.72,
            "reasoning": "Aggregator-style account that reposts headlines verbatim. May be automated but provides some signal value.",
        },
        {
            "username": "real_trader_42",
            "classification": "Human",
            "confidence": 0.81,
            "reasoning": "Conversational style, varied topics, engagement with replies. ML score likely elevated due to high posting frequency.",
        },
    ],
    "ml_evaluation": {
        "agreement_rate": 0.75,
        "precision": 0.82,
        "recall": 0.67,
        "f1_score": 0.74,
        "disagreements": [
            {
                "username": "real_trader_42",
                "ml_score": 0.45,
                "llm_classification": "Human",
                "reason": "ML flagged due to high volume; LLM identified organic conversational patterns and varied content.",
            },
            {
                "username": "breaking_mkt_news",
                "ml_score": 0.71,
                "llm_classification": "Likely Bot",
                "reason": "Both agree on suspicion but differ on severity. Account may be semi-automated news aggregator.",
            },
        ],
    },
}


def test_connection(host: str, port: int, method: str, username: str, password: str, timeout: int = 10) -> bool:
    """
    Test a specific SMTP connection method.

    Args:
        method: "STARTTLS" (Explicit SSL) or "SSL/TLS" (Implicit SSL)
    """
    logger.info(f"Probing {host}:{port} using {method}...")

    try:
        start_time = time.time()

        if method == "SSL/TLS":
            server = smtplib.SMTP_SSL(host, port, timeout=timeout)
            logger.info(f"  Connected (Time: {time.time() - start_time:.2f}s)")

            logger.info(f"  Logging in as {username}...")
            server.login(username, password)
            logger.info("  Login successful")
            server.quit()

        elif method == "STARTTLS":
            server = smtplib.SMTP(host, port, timeout=timeout)
            logger.info(f"  Connected (Time: {time.time() - start_time:.2f}s)")
            server.ehlo()

            logger.info("  Sending STARTTLS...")
            server.starttls()
            logger.info("  STARTTLS accepted")

            logger.info(f"  Logging in as {username}...")
            server.login(username, password)
            logger.info("  Login successful")
            server.quit()

        return True

    except socket.timeout:
        logger.error(f"  Connection timed out after {timeout}s")
        logger.error("    (Firewall might be blocking this port)")
    except smtplib.SMTPAuthenticationError:
        logger.error("  Authentication failed (Wrong username/password)")
    except smtplib.SMTPConnectError:
        logger.error("  Could not connect (Port closed or unreachable)")
    except smtplib.SMTPException as e:
        logger.error(f"  SMTP Error: {e}")
    except Exception as e:
        logger.error(f"  Unexpected error: {e}")

    return False


def main():
    logger.info("=== Jafar SMTP Diagnostic Tool ===")

    if not config.smtp.username or not config.smtp.password:
        logger.error("Missing SMTP credentials in .env")
        sys.exit(1)

    current_protocol = "STARTTLS" if config.smtp.use_tls else "SSL/TLS"
    logger.info(f"Current Config: {config.smtp.host}:{config.smtp.port} ({current_protocol})")
    logger.info(f"User: {config.smtp.username}")

    logger.info("\n--- Step 1: Testing Configured Settings ---")
    if test_connection(
        config.smtp.host,
        config.smtp.port,
        current_protocol,
        config.smtp.username,
        config.smtp.password
    ):
        logger.info("\nConfiguration is VALID.")
        logger.info("Sending test email using the full HTML template with sample data...")

        email_config = EmailConfig(
            host=config.smtp.host,
            port=config.smtp.port,
            username=config.smtp.username,
            password=config.smtp.password,
            use_tls=config.smtp.use_tls,
            email_from=config.smtp.email_from,
            email_from_name=config.smtp.email_from_name,
            email_to=config.smtp.email_to
        )
        reporter = EmailReporter(email_config)
        success = reporter.send_email(
            report_content=SAMPLE_REPORT_CONTENT,
            trends=SAMPLE_TRENDS,
            tweet_count=1247,
            provider_info="Test Diagnostic (Sample Data)",
            signal_strength="medium",
            subject_line="[TEST] Jafar Template Verification",
            ml_results=SAMPLE_ML_RESULTS,
        )
        if success:
            logger.info("Email sent successfully.")
            sys.exit(0)
        else:
            logger.error("Connection worked but sending failed.")
            sys.exit(1)

    logger.info("\n--- Step 2: Diagnosing Alternatives ---")
    logger.warning("Current configuration failed. Probing common alternatives...")

    alternatives = [
        (587, "STARTTLS", "Standard Submission Port"),
        (465, "SSL/TLS", "Legacy Secure Port"),
        (2525, "STARTTLS", "Alternative Port"),
        (2525, "SSL/TLS", "Alternative Port"),
    ]

    for port, proto, desc in alternatives:
        if port == config.smtp.port and proto == current_protocol:
            continue

        logger.info(f"\nTrying {desc}: {config.smtp.host}:{port} ({proto})")
        if test_connection(config.smtp.host, port, proto, config.smtp.username, config.smtp.password):
            logger.info("\n" + "="*50)
            logger.info("FOUND WORKING CONFIGURATION!")
            logger.info("Update your config.yaml to:")
            logger.info("smtp:")
            logger.info(f"  host: {config.smtp.host}")
            logger.info(f"  port: {port}")
            logger.info(f"  use_tls: {str(proto == 'STARTTLS').lower()}")
            logger.info("="*50)
            sys.exit(0)

    logger.error("\nAll attempts failed.")
    logger.error("Possible causes:")
    logger.error("1. VPS Firewall is blocking outgoing SMTP ports (check ufw/iptables)")
    logger.error("2. VPS Provider (e.g., DigitalOcean, Vultr) blocks SMTP by default")
    logger.error("3. Hostname is incorrect")
    logger.error("4. Credentials are incorrect")

if __name__ == "__main__":
    main()
