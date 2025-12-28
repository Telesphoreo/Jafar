"""
Test Email Script & SMTP Diagnoser.

Verifies SMTP settings, tests connectivity, and helps debug configuration issues
by probing common email ports and protocols.
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
            # Implicit SSL (usually port 465)
            server = smtplib.SMTP_SSL(host, port, timeout=timeout)
            logger.info(f"  ✓ Connected (Time: {time.time() - start_time:.2f}s)")
            
            logger.info(f"  → Logging in as {username}...")
            server.login(username, password)
            logger.info("  ✓ Login successful")
            server.quit()
            
        elif method == "STARTTLS":
            # Explicit SSL (usually port 587)
            server = smtplib.SMTP(host, port, timeout=timeout)
            logger.info(f"  ✓ Connected (Time: {time.time() - start_time:.2f}s)")
            server.ehlo()
            
            logger.info("  → Sending STARTTLS...")
            server.starttls()
            logger.info("  ✓ STARTTLS accepted")
            
            logger.info(f"  → Logging in as {username}...")
            server.login(username, password)
            logger.info("  ✓ Login successful")
            server.quit()
            
        return True

    except socket.timeout:
        logger.error(f"  ✗ Connection timed out after {timeout}s")
        logger.error("    (Firewall might be blocking this port)")
    except smtplib.SMTPAuthenticationError:
        logger.error("  ✗ Authentication failed (Wrong username/password)")
    except smtplib.SMTPConnectError:
        logger.error("  ✗ Could not connect (Port closed or unreachable)")
    except smtplib.SMTPException as e:
        logger.error(f"  ✗ SMTP Error: {e}")
    except Exception as e:
        logger.error(f"  ✗ Unexpected error: {e}")
        
    return False

def main():
    logger.info("=== Jafar SMTP Diagnostic Tool ===")
    
    # 1. Validate Basic Config
    if not config.smtp.username or not config.smtp.password:
        logger.error("Missing SMTP credentials in .env")
        sys.exit(1)

    current_protocol = "STARTTLS" if config.smtp.use_tls else "SSL/TLS"
    logger.info(f"Current Config: {config.smtp.host}:{config.smtp.port} ({current_protocol})")
    logger.info(f"User: {config.smtp.username}")

    # 2. Test Current Configuration
    logger.info("\n--- Step 1: Testing Configured Settings ---")
    test_connection(
        config.smtp.host, 
        config.smtp.port, 
        current_protocol, 
        config.smtp.username, 
        config.smtp.password
    ):
        logger.info("\n✓ Configuration is VALID.")
        logger.info("Attempting to send actual test email...")
        
        # Run the actual reporter
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
        success = reporter.send_email(
            report_content="Diagnostic test email.",
            trends=["Diagnostic"],
            tweet_count=0,
            provider_info="Diagnostic Tool",
            signal_strength="low"
        )
        if success:
            logger.info("✓ Email sent successfully.")
            sys.exit(0)
        else:
            logger.error("✗ Connection worked but sending failed.")
            sys.exit(1)
            
    # 3. Diagnostics if failure
    logger.info("\n--- Step 2: Diagnosing Alternatives ---")
    logger.warning("Current configuration failed. Probing common alternatives...")

    # Define common combinations to try
    alternatives = [
        # (Port, Protocol, Description)
        (587, "STARTTLS", "Standard Submission Port"),
        (465, "SSL/TLS", "Legacy Secure Port"),
        (2525, "STARTTLS", "Alternative Port"),
        (2525, "SSL/TLS", "Alternative Port"),
    ]

    for port, proto, desc in alternatives:
        # Skip if we just tested this exact combo
        if port == config.smtp.port and proto == current_protocol:
            continue
            
        logger.info(f"\nTrying {desc}: {config.smtp.host}:{port} ({proto})")
        if test_connection(config.smtp.host, port, proto, config.smtp.username, config.smtp.password):
            logger.info("\n" + "="*50)
            logger.info("✓ FOUND WORKING CONFIGURATION!")
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
