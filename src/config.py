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
    model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))


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
    # Higher limits = more data = better discovery (but slower due to rate limits)
    # With multiple accounts, you can increase these significantly
    broad_tweet_limit: int = field(default_factory=lambda: int(os.getenv("BROAD_TWEET_LIMIT", "200")))
    specific_tweet_limit: int = field(default_factory=lambda: int(os.getenv("SPECIFIC_TWEET_LIMIT", "100")))
    top_trends_count: int = field(default_factory=lambda: int(os.getenv("TOP_TRENDS_COUNT", "10")))

    # Minimum thresholds for trend detection (filters spam/noise)
    min_trend_mentions: int = field(default_factory=lambda: int(os.getenv("MIN_TREND_MENTIONS", "3")))
    min_trend_authors: int = field(default_factory=lambda: int(os.getenv("MIN_TREND_AUTHORS", "2")))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    llm_provider: Literal["openai", "google"] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower()  # type: ignore
    )

    # Broad search topics for DISCOVERY - cast the widest possible net
    # The goal is to capture ALL financial chatter, then let the analyzer find anomalies
    broad_topics: list[str] = field(default_factory=lambda: [
        # General financial Twitter communities - high volume, diverse content
        "fintwit",
        "stock market today",
        "trading",
        "markets",

        # Breaking/emerging signals
        "breaking market",
        "just announced",
        "shortage OR surplus",
        "supply chain",

        # Commodities broad sweep
        "commodities",
        "futures",
        "spot price",

        # Options flow - often leads price
        "unusual volume",
        "options flow",
        "dark pool",
        "whale alert",

        # Sector rotation signals
        "sector rotation",
        "money flowing",
        "outperform OR underperform",

        # Macro signals
        "inflation data",
        "yield curve",
        "dollar index",

        # International markets (often lead US)
        "asia markets",
        "europe open",
        "emerging markets",

        # Sentiment extremes
        "oversold OR overbought",
        "capitulation",
        "FOMO OR panic",

        # Earnings/events
        "earnings surprise",
        "guidance raised OR lowered",
        "FDA approval OR rejection",

        # Physical markets
        "physical delivery",
        "warehouse inventory",
        "shipping rates",
        "freight",
    ])


@dataclass
class MemoryConfig:
    """Vector memory configuration for historical context."""
    # Store type: "chroma" (local) or "pgvector" (production)
    store_type: Literal["chroma", "pgvector"] = field(
        default_factory=lambda: os.getenv("MEMORY_STORE_TYPE", "chroma")  # type: ignore
    )
    # Embedding provider: "openai" (better) or "local" (free)
    embedding_provider: Literal["openai", "local"] = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai")  # type: ignore
    )
    # ChromaDB storage path (for local store)
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./memory_store"))
    # PostgreSQL connection string (for pgvector)
    postgres_url: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", ""))
    # Whether to enable memory system
    enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    )
    # Minimum similarity for parallel detection
    min_similarity: float = field(
        default_factory=lambda: float(os.getenv("MEMORY_MIN_SIMILARITY", "0.6"))
    )


@dataclass
class Config:
    """Main configuration container."""
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    app: AppConfig = field(default_factory=AppConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

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
