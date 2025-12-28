"""
Configuration module for Twitter Sentiment Analysis.

Loads environment variables and provides typed configuration access.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TwitterConfig:
    """Twitter/X credentials configuration."""
    username: str = field(default_factory=lambda: os.getenv("TWITTER_USERNAME", ""))
    password: str = field(default_factory=lambda: os.getenv("TWITTER_PASSWORD", ""))
    email: str = field(default_factory=lambda: os.getenv("TWITTER_EMAIL", ""))
    email_password: str = field(default_factory=lambda: os.getenv("TWITTER_EMAIL_PASSWORD", ""))
    db_path: str = field(default_factory=lambda: os.getenv("TWSCRAPE_DB_PATH", "accounts.db"))


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))


@dataclass
class GoogleConfig:
    """Google Generative AI configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-1.5-pro"))


@dataclass
class SMTPConfig:
    """SMTP email configuration."""
    host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.gmail.com"))
    port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    username: str = field(default_factory=lambda: os.getenv("SMTP_USERNAME", ""))
    password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    use_tls: bool = field(default_factory=lambda: os.getenv("SMTP_USE_TLS", "true").lower() == "true")
    email_from: str = field(default_factory=lambda: os.getenv("EMAIL_FROM", ""))
    email_to: list[str] = field(default_factory=lambda: [
        e.strip() for e in os.getenv("EMAIL_TO", "").split(",") if e.strip()
    ])


@dataclass
class AppConfig:
    """Application settings."""
    broad_tweet_limit: int = field(default_factory=lambda: int(os.getenv("BROAD_TWEET_LIMIT", "50")))
    specific_tweet_limit: int = field(default_factory=lambda: int(os.getenv("SPECIFIC_TWEET_LIMIT", "20")))
    top_trends_count: int = field(default_factory=lambda: int(os.getenv("TOP_TRENDS_COUNT", "5")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    llm_provider: Literal["openai", "google"] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower()  # type: ignore
    )

    # Broad search topics for initial discovery
    broad_topics: list[str] = field(default_factory=lambda: [
        "economy",
        "recession",
        "markets",
        "inflation",
        "federal reserve",
    ])


@dataclass
class Config:
    """Main configuration container."""
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    app: AppConfig = field(default_factory=AppConfig)

    def setup_logging(self) -> logging.Logger:
        """Configure and return the application logger."""
        logging.basicConfig(
            level=getattr(logging, self.app.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger("twitter_sentiment")

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of missing/invalid settings.

        Returns:
            List of validation error messages, empty if all valid.
        """
        errors = []

        # Check LLM provider configuration
        if self.app.llm_provider == "openai" and not self.openai.api_key:
            errors.append("OPENAI_API_KEY is required when using OpenAI provider")
        elif self.app.llm_provider == "google" and not self.google.api_key:
            errors.append("GOOGLE_API_KEY is required when using Google provider")

        # Check SMTP configuration for email sending
        if not self.smtp.username or not self.smtp.password:
            errors.append("SMTP_USERNAME and SMTP_PASSWORD are required for email")
        if not self.smtp.email_to:
            errors.append("EMAIL_TO is required (at least one recipient)")

        return errors


# Global configuration instance
config = Config()
