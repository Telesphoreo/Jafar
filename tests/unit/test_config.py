"""
Unit tests for src/config.py

Tests configuration loading, validation, and dataclass behavior.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestYamlConfigLoading:
    """Tests for YAML configuration loading."""

    def test_get_yaml_returns_default_for_missing_key(self):
        """Test that _get_yaml returns default when key is missing."""
        from src.config import _get_yaml

        result = _get_yaml("nonexistent", "key", "default_value")
        assert result == "default_value"

    def test_get_yaml_section_returns_empty_dict_for_missing(self):
        """Test that _get_yaml_section returns empty dict for missing section."""
        from src.config import _get_yaml_section

        result = _get_yaml_section("nonexistent")
        assert result == {}


class TestTwitterConfig:
    """Tests for TwitterConfig dataclass."""

    def test_twitter_config_defaults(self, mock_env_vars):
        """Test TwitterConfig uses environment variables."""
        from src.config import TwitterConfig

        config = TwitterConfig()
        # Environment values should be used when available
        assert config.db_path == "accounts.db"  # Default from YAML

    def test_twitter_config_proxies_from_env(self, monkeypatch):
        """Test proxies can be loaded from environment."""
        monkeypatch.setenv("TWITTER_PROXIES", "http://proxy1:8080,http://proxy2:8080")

        from src.config import _get_proxies

        proxies = _get_proxies()
        assert len(proxies) == 2
        assert "http://proxy1:8080" in proxies


class TestOpenAIConfig:
    """Tests for OpenAIConfig dataclass."""

    def test_openai_config_has_model(self):
        """Test OpenAIConfig has a model configured."""
        from src.config import OpenAIConfig

        config = OpenAIConfig()
        # Model should be set (either from config.yaml or default)
        assert config.model is not None
        assert len(config.model) > 0

    def test_openai_config_uses_env_api_key(self, monkeypatch):
        """Test that API key comes from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

        from src.config import OpenAIConfig

        config = OpenAIConfig()
        assert config.api_key == "test-key-123"


class TestGoogleConfig:
    """Tests for GoogleConfig dataclass."""

    def test_google_config_has_model(self):
        """Test GoogleConfig has a model configured."""
        from src.config import GoogleConfig

        config = GoogleConfig()
        # Model should be set (either from config.yaml or default)
        assert config.model is not None
        assert "gemini" in config.model.lower()

    def test_google_config_uses_env_api_key(self, monkeypatch):
        """Test that API key comes from environment."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")

        from src.config import GoogleConfig

        config = GoogleConfig()
        assert config.api_key == "google-test-key"


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_app_config_has_values(self):
        """Test AppConfig has valid values."""
        from src.config import AppConfig

        config = AppConfig()
        # Values should be positive integers
        assert config.broad_tweet_limit > 0
        assert config.specific_tweet_limit > 0
        assert config.top_trends_count > 0
        assert config.min_trend_mentions > 0
        assert config.min_trend_authors > 0
        assert config.llm_provider in ["openai", "google"]

    def test_broad_topics_has_entries(self):
        """Test that broad_topics has entries."""
        from src.config import AppConfig

        config = AppConfig()
        assert len(config.broad_topics) > 0
        # Should have at least some financial-related topics
        topics_str = " ".join(config.broad_topics).lower()
        assert "market" in topics_str or "trading" in topics_str or "fintwit" in topics_str


class TestFactCheckerConfig:
    """Tests for FactCheckerConfig dataclass."""

    def test_fact_checker_defaults(self):
        """Test FactCheckerConfig defaults."""
        from src.config import FactCheckerConfig

        config = FactCheckerConfig()
        assert config.enabled is True
        assert config.cache_ttl_minutes == 5
        assert config.price_tolerance_pct == 2.0


class TestTemporalConfig:
    """Tests for TemporalConfig dataclass."""

    def test_temporal_config_defaults(self):
        """Test TemporalConfig defaults."""
        from src.config import TemporalConfig

        config = TemporalConfig()
        assert config.consecutive_threshold == 3
        assert config.gap_threshold_days == 14


class TestConfigValidation:
    """Tests for Config.validate() method."""

    def test_validate_missing_openai_key(self, monkeypatch, sample_config_yaml):
        """Test validation fails when OpenAI provider selected without key."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Need to reload config module to pick up changes
        with patch("src.config._yaml_config", {"llm": {"provider": "openai"}}):
            from src.config import Config

            config = Config()
            # Force OpenAI provider
            config.app.llm_provider = "openai"
            config.openai.api_key = ""

            errors = config.validate()
            assert any("OPENAI_API_KEY" in e for e in errors)

    def test_validate_missing_google_key(self, monkeypatch):
        """Test validation fails when Google provider selected without key."""
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        from src.config import Config

        config = Config()
        config.app.llm_provider = "google"
        config.google.api_key = ""

        errors = config.validate()
        assert any("GOOGLE_API_KEY" in e for e in errors)

    def test_validate_missing_smtp_credentials(self, monkeypatch):
        """Test validation fails without SMTP credentials."""
        monkeypatch.setenv("SMTP_USERNAME", "")
        monkeypatch.setenv("SMTP_PASSWORD", "")

        from src.config import Config

        config = Config()
        config.smtp.username = ""
        config.smtp.password = ""

        errors = config.validate()
        assert any("SMTP" in e for e in errors)


class TestWorkerLogFilter:
    """Tests for WorkerLogFilter."""

    def test_filter_adds_worker_info(self):
        """Test that filter adds worker_info to log records."""
        import logging

        from src.config import WorkerLogFilter, worker_context

        filter_instance = WorkerLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Without worker context
        filter_instance.filter(record)
        assert record.worker_info == ""

        # With worker context
        worker_context.set(5)
        filter_instance.filter(record)
        assert record.worker_info == " [Worker 5]"

        # Clean up
        worker_context.set(None)
