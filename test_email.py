"""
Test Email Script.

Verifies SMTP settings and email delivery in isolation.
"""

import sys
import logging
from src.config import config
from src.reporter import EmailReporter, EmailConfig

def main():
    # Setup logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("jafar.test_email")

    logger.info("Initializing configuration...")
    
    # Validate SMTP config
    if not config.smtp.username or not config.smtp.password:
        logger.error("Missing SMTP credentials. Please check your .env file.")
        logger.error("Required: SMTP_USERNAME, SMTP_PASSWORD")
        sys.exit(1)
        
    if not config.smtp.email_to:
        logger.error("No recipients defined. Please check 'email.to' in config.yaml.")
        sys.exit(1)

    logger.info(f"Host: {config.smtp.host}:{config.smtp.port}")
    logger.info(f"User: {config.smtp.username}")
    logger.info(f"From: {config.smtp.email_from}")
    logger.info(f"To:   {config.smtp.email_to}")

    # Initialize reporter
    email_config = EmailConfig(
        host=config.smtp.host,
        port=config.smtp.port,
        username=config.smtp.username,
        password=config.smtp.password,
        use_tls=config.smtp.use_tls,
        email_from=config.smtp.email_from,
        email_to=config.smtp.email_to
    )
    
    reporter = EmailReporter(email_config)

    # Test payload
    logger.info("Sending test email...")
    success = reporter.send_email(
        report_content="This is a test email from the Jafar CLI to verify SMTP settings.\n\nIf you are reading this, the email configuration is correct.",
        trends=["Test Trend 1", "Test Trend 2"],
        tweet_count=42,
        provider_info="Test Script",
        signal_strength="low"
    )

    if success:
        logger.info("Test completed successfully.")
    else:
        logger.error("Test failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
