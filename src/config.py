"""
Configuration module for Twitter Sentiment Analysis.

Loads application settings from config.yaml and secrets from environment variables.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Default config file path
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"


def _load_yaml_config() -> dict:
    """Load configuration from YAML file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml_config = _load_yaml_config()


def _get_yaml(section: str, key: str, default=None):
    """Get a value from the YAML config."""
    return _yaml_config.get(section, {}).get(key, default)


def _get_yaml_section(section: str, default=None):
    """Get an entire section from the YAML config."""
    return _yaml_config.get(section, default or {})


@dataclass
class TwitterConfig:
    """Twitter/X credentials and settings."""
    # Secrets from .env
    username: str = field(default_factory=lambda: os.getenv("TWITTER_USERNAME", ""))
    password: str = field(default_factory=lambda: os.getenv("TWITTER_PASSWORD", ""))
    email: str = field(default_factory=lambda: os.getenv("TWITTER_EMAIL", ""))
    email_password: str = field(default_factory=lambda: os.getenv("TWITTER_EMAIL_PASSWORD", ""))

    # Settings from YAML (with .env fallback for proxies with credentials)
    db_path: str = field(default_factory=lambda: _get_yaml("twitter", "db_path", "accounts.db"))
    proxies: list[str] = field(default_factory=lambda: _get_proxies())  # pylint: disable=unnecessary-lambda


def _get_proxies() -> list[str]:
    """Get proxies from YAML or .env (for proxies with credentials)."""
    # First check .env (for proxies with credentials)
    env_proxies = os.getenv("TWITTER_PROXIES", "")
    if env_proxies:
        return [p.strip() for p in env_proxies.split(",") if p.strip()]
    # Fall back to YAML
    return _get_yaml("twitter", "proxies", []) or []


@dataclass
class GoogleConfig:
    """Google Generative AI configuration (LLM + embeddings)."""
    # Secret from .env
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    # LLM model
    model: str = field(default_factory=lambda: _get_yaml("llm", "google_model", "gemini-3.1-flash-lite"))
    # Embedding model
    embedding_model: str = field(
        default_factory=lambda: _get_yaml("llm", "embedding_model", "gemini-embedding-2-preview")
    )
    # Embedding dimensions (128-3072, recommended: 768, 1536, or 3072)
    embedding_dimensions: int = field(
        default_factory=lambda: _get_yaml("llm", "embedding_dimensions", 3072)
    )


@dataclass
class AdminEmailConfig:
    """Admin diagnostics email configuration."""
    enabled: bool = field(
        default_factory=lambda: _get_yaml_section("email").get("admin", {}).get("enabled", True)
    )
    send_on_success: bool = field(
        default_factory=lambda: _get_yaml_section("email").get("admin", {}).get("send_on_success", False)
    )
    recipients: list[str] = field(
        default_factory=lambda: _get_yaml_section("email").get("admin", {}).get("recipients", []) or []
    )
    log_retention_count: int = field(
        default_factory=lambda: _get_yaml_section("email").get("admin", {}).get("log_retention_count", 10)
    )


@dataclass
class SMTPConfig:
    """SMTP email configuration."""
    # Settings from YAML
    host: str = field(default_factory=lambda: _get_yaml("smtp", "host", "smtp.gmail.com"))
    port: int = field(default_factory=lambda: _get_yaml("smtp", "port", 587))
    use_tls: bool = field(default_factory=lambda: _get_yaml("smtp", "use_tls", True))

    # Secrets from .env
    username: str = field(default_factory=lambda: os.getenv("SMTP_USERNAME", ""))
    password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))

    # Settings from YAML
    email_from: str = field(default_factory=lambda: _get_yaml("email", "from", ""))
    email_from_name: str = field(
        default_factory=lambda: _get_yaml_section("email").get("from_name", "Jafar Intelligence System")
    )
    email_to: list[str] = field(default_factory=lambda: _get_yaml("email", "to", []) or [])

    # Admin diagnostics
    admin: AdminEmailConfig = field(default_factory=AdminEmailConfig)


@dataclass
class AppConfig:
    """Application settings from YAML."""
    # Scraping limits
    broad_tweet_limit: int = field(
        default_factory=lambda: _get_yaml("scraping", "broad_tweet_limit", 200)
    )
    specific_tweet_limit: int = field(
        default_factory=lambda: _get_yaml("scraping", "specific_tweet_limit", 100)
    )
    top_trends_count: int = field(
        default_factory=lambda: _get_yaml("scraping", "top_trends_count", 10)
    )
    min_trend_mentions: int = field(
        default_factory=lambda: _get_yaml("scraping", "min_trend_mentions", 3)
    )
    min_trend_authors: int = field(
        default_factory=lambda: _get_yaml("scraping", "min_trend_authors", 2)
    )
    # Logging
    log_level: str = field(
        default_factory=lambda: _get_yaml("logging", "level", "INFO")
    )

    # NLP Model
    spacy_model: str = field(
        default_factory=lambda: _get_yaml("app", "spacy_model", "en_core_web_sm")
    )

    # Broad search topics
    broad_topics: list[str] = field(default_factory=lambda: _get_broad_topics())  # pylint: disable=unnecessary-lambda


def _get_broad_topics() -> list[str]:
    """Get broad topics from YAML or use defaults."""
    topics = _get_yaml_section("broad_topics")
    if topics:
        return topics

    # Default topics if not in YAML
    return [
        "fintwit",
        "stock market today",
        "trading",
        "markets",
        "breaking market",
        "just announced",
        "shortage OR surplus",
        "supply chain",
        "commodities",
        "futures",
        "spot price",
        "unusual volume",
        "options flow",
        "dark pool",
        "whale alert",
        "sector rotation",
        "money flowing",
        "outperform OR underperform",
        "inflation data",
        "yield curve",
        "dollar index",
        "asia markets",
        "europe open",
        "emerging markets",
        "oversold OR overbought",
        "capitulation",
        "FOMO OR panic",
        "earnings surprise",
        "guidance raised OR lowered",
        "FDA approval OR rejection",
        "physical delivery",
        "warehouse inventory",
        "shipping rates",
        "freight",
    ]


@dataclass
class DatabaseConfig:
    """Postgres database configuration."""
    # Secret from .env (contains credentials)
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    # Memory/vector search settings
    memory_enabled: bool = field(
        default_factory=lambda: _get_yaml("memory", "enabled", True)
    )
    min_similarity: float = field(
        default_factory=lambda: _get_yaml("memory", "min_similarity", 0.6)
    )


@dataclass
class FactCheckerConfig:
    """Market data fact-checking configuration."""
    enabled: bool = field(
        default_factory=lambda: _get_yaml("fact_checker", "enabled", True)
    )
    # Cache market data for N minutes to avoid excessive API calls
    cache_ttl_minutes: int = field(
        default_factory=lambda: _get_yaml("fact_checker", "cache_ttl_minutes", 5)
    )
    # Allowed variance for "price at X" claims
    price_tolerance_pct: float = field(
        default_factory=lambda: _get_yaml("fact_checker", "price_tolerance_pct", 2.0)
    )


@dataclass
class TemporalConfig:
    """Temporal trend tracking configuration."""
    consecutive_threshold: int = field(
        default_factory=lambda: _get_yaml("temporal", "consecutive_threshold", 3)
    )
    gap_threshold_days: int = field(
        default_factory=lambda: _get_yaml("temporal", "gap_threshold_days", 14)
    )


@dataclass
class Config:
    """Main configuration container."""
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    app: AppConfig = field(default_factory=AppConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    fact_checker: FactCheckerConfig = field(default_factory=FactCheckerConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    def setup_logging(self) -> logging.Logger:
        """Configure and return the application logger."""
        # Reset existing handlers to ensure clean configuration
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)

        logging.basicConfig(
            level=getattr(logging, self.app.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        return logging.getLogger("jafar")

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of missing/invalid settings.

        Returns:
            List of validation error messages, empty if all valid.
        """
        errors = []

        # Check if config.yaml exists
        if not CONFIG_FILE.exists():
            errors.append(f"Config file not found: {CONFIG_FILE} (copy config.yaml.example to config.yaml)")

        # Check Google API key
        if not self.google.api_key:
            errors.append("GOOGLE_API_KEY is required (used for LLM and embeddings)")

        # Check SMTP configuration for email sending
        if not self.smtp.username or not self.smtp.password:
            errors.append("SMTP_USERNAME and SMTP_PASSWORD are required for email")
        if not self.smtp.email_to:
            errors.append("email.to is required in config.yaml (at least one recipient)")

        return errors


# Global configuration instance
config = Config()
