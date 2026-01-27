"""
Unit tests for src/fact_checker.py

Tests market data extraction, caching, and fact-checking with mocked yfinance.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import asyncio

import pytest

from src.fact_checker import (
    COMMODITY_MAP,
    CRYPTO_MAP,
    INDEX_MAP,
    MarketDataPoint,
    MarketFactChecker,
)


class TestMarketDataPoint:
    """Tests for MarketDataPoint dataclass."""

    def test_basic_creation(self):
        """Test creating a MarketDataPoint."""
        data = MarketDataPoint(
            symbol="NVDA",
            name="NVIDIA Corporation",
            current_price=150.0,
            price_change_1d=5.0,
            price_change_1d_pct=3.45,
            price_change_5d_pct=8.2,
            volume=50000000,
            avg_volume=30000000,
            high_52w=160.0,
            low_52w=80.0,
            market_cap=1500000000000,
            fetched_at=datetime.now(),
            category="stock",
        )

        assert data.symbol == "NVDA"
        assert data.current_price == 150.0

    def test_is_near_52w_high(self):
        """Test 52-week high detection."""
        data = MarketDataPoint(
            symbol="TEST",
            name="Test Stock",
            current_price=155.0,  # Within 5% of high
            price_change_1d=0.0,
            price_change_1d_pct=0.0,
            price_change_5d_pct=0.0,
            volume=1000000,
            avg_volume=1000000,
            high_52w=160.0,
            low_52w=80.0,
            market_cap=None,
            fetched_at=datetime.now(),
            category="stock",
        )

        assert data.is_near_52w_high is True
        assert data.is_near_52w_low is False

    def test_is_near_52w_low(self):
        """Test 52-week low detection."""
        data = MarketDataPoint(
            symbol="TEST",
            name="Test Stock",
            current_price=82.0,  # Within 5% of low
            price_change_1d=0.0,
            price_change_1d_pct=0.0,
            price_change_5d_pct=0.0,
            volume=1000000,
            avg_volume=1000000,
            high_52w=160.0,
            low_52w=80.0,
            market_cap=None,
            fetched_at=datetime.now(),
            category="stock",
        )

        assert data.is_near_52w_low is True
        assert data.is_near_52w_high is False

    def test_is_unusual_volume(self):
        """Test unusual volume detection."""
        data = MarketDataPoint(
            symbol="TEST",
            name="Test Stock",
            current_price=100.0,
            price_change_1d=0.0,
            price_change_1d_pct=0.0,
            price_change_5d_pct=0.0,
            volume=5000000,  # 2.5x average
            avg_volume=2000000,
            high_52w=120.0,
            low_52w=80.0,
            market_cap=None,
            fetched_at=datetime.now(),
            category="stock",
        )

        assert data.is_unusual_volume is True
        assert data.volume_ratio == 2.5

    def test_volume_ratio_handles_zero_avg(self):
        """Test that zero average volume doesn't cause division error."""
        data = MarketDataPoint(
            symbol="TEST",
            name="Test Stock",
            current_price=100.0,
            price_change_1d=0.0,
            price_change_1d_pct=0.0,
            price_change_5d_pct=0.0,
            volume=1000000,
            avg_volume=0,  # Zero average
            high_52w=120.0,
            low_52w=80.0,
            market_cap=None,
            fetched_at=datetime.now(),
            category="stock",
        )

        assert data.volume_ratio == 0.0
        assert data.is_unusual_volume is False


class TestMarketFactChecker:
    """Tests for MarketFactChecker class."""

    def test_init(self):
        """Test fact checker initialization."""
        checker = MarketFactChecker(
            cache_ttl_minutes=10,
            price_tolerance_pct=3.0,
        )

        assert checker.cache_ttl == timedelta(minutes=10)
        assert checker.price_tolerance_pct == 3.0

    def test_extract_cashtags(self):
        """Test extracting cashtags from trends."""
        checker = MarketFactChecker()
        trends = ["$NVDA", "$AAPL", "random text"]

        symbols = checker.extract_symbols_from_trends(trends)

        assert "NVDA" in symbols
        assert "AAPL" in symbols

    def test_extract_commodity_symbols(self):
        """Test extracting commodity symbols from names."""
        checker = MarketFactChecker()
        trends = ["gold", "silver prices", "oil market"]

        symbols = checker.extract_symbols_from_trends(trends)

        assert "GC=F" in symbols  # Gold futures
        assert "SI=F" in symbols  # Silver futures
        assert "CL=F" in symbols  # Oil futures

    def test_extract_crypto_symbols(self):
        """Test extracting crypto symbols."""
        checker = MarketFactChecker()
        trends = ["bitcoin", "ethereum rally"]

        symbols = checker.extract_symbols_from_trends(trends)

        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols

    def test_extract_index_symbols(self):
        """Test extracting index symbols."""
        checker = MarketFactChecker()
        trends = ["s&p 500", "nasdaq", "dow jones"]

        symbols = checker.extract_symbols_from_trends(trends)

        assert "^GSPC" in symbols  # S&P 500
        assert "^IXIC" in symbols  # NASDAQ
        assert "^DJI" in symbols  # Dow Jones

    def test_get_category(self):
        """Test category detection for symbols."""
        checker = MarketFactChecker()

        assert checker._get_category("GC=F") == "commodity"
        assert checker._get_category("BTC-USD") == "crypto"
        assert checker._get_category("^GSPC") == "index"
        assert checker._get_category("XLE") == "etf"
        assert checker._get_category("NVDA") == "stock"

    def test_cache_validity(self):
        """Test cache validity checking."""
        checker = MarketFactChecker(cache_ttl_minutes=5)

        # Not in cache
        assert checker._is_cache_valid("NVDA") is False

        # Add to cache
        mock_data = MagicMock()
        checker._cache["NVDA"] = (mock_data, datetime.now())
        assert checker._is_cache_valid("NVDA") is True

        # Expired cache
        checker._cache["OLD"] = (mock_data, datetime.now() - timedelta(minutes=10))
        assert checker._is_cache_valid("OLD") is False

    @pytest.mark.asyncio
    async def test_fetch_market_data_uses_cache(self, mock_yfinance):
        """Test that cached data is reused."""
        checker = MarketFactChecker()

        # Pre-populate cache
        mock_data = MarketDataPoint(
            symbol="NVDA",
            name="NVIDIA",
            current_price=150.0,
            price_change_1d=0.0,
            price_change_1d_pct=0.0,
            price_change_5d_pct=0.0,
            volume=1000000,
            avg_volume=1000000,
            high_52w=160.0,
            low_52w=80.0,
            market_cap=None,
            fetched_at=datetime.now(),
            category="stock",
        )
        checker._cache["NVDA"] = (mock_data, datetime.now())

        # Fetch should use cache
        result = await checker.fetch_market_data({"NVDA"}, include_common=False)

        assert "NVDA" in result
        assert result["NVDA"].current_price == 150.0

    def test_format_for_llm(self):
        """Test LLM formatting of market data."""
        checker = MarketFactChecker()

        market_data = {
            "NVDA": MarketDataPoint(
                symbol="NVDA",
                name="NVIDIA Corporation",
                current_price=150.0,
                price_change_1d=5.0,
                price_change_1d_pct=3.45,
                price_change_5d_pct=8.2,
                volume=50000000,
                avg_volume=30000000,
                high_52w=160.0,
                low_52w=80.0,
                market_cap=1500000000000,
                fetched_at=datetime.now(),
                category="stock",
            ),
            "GC=F": MarketDataPoint(
                symbol="GC=F",
                name="Gold",
                current_price=2050.0,
                price_change_1d=15.0,
                price_change_1d_pct=0.74,
                price_change_5d_pct=2.1,
                volume=100000,
                avg_volume=80000,
                high_52w=2100.0,
                low_52w=1800.0,
                market_cap=None,
                fetched_at=datetime.now(),
                category="commodity",
            ),
        }

        formatted = checker.format_for_llm(market_data, ["$NVDA", "gold"])

        assert "VERIFIED MARKET DATA" in formatted
        assert "NVIDIA" in formatted or "NVDA" in formatted
        assert "Gold" in formatted or "GC=F" in formatted
        assert "FACT-CHECK INSTRUCTIONS" in formatted

    def test_format_for_llm_empty(self):
        """Test LLM formatting with empty data."""
        checker = MarketFactChecker()
        formatted = checker.format_for_llm({}, [])
        assert formatted == ""

    def test_get_notes(self):
        """Test notes generation for data points."""
        checker = MarketFactChecker()

        # Near 52w high with unusual volume
        data = MarketDataPoint(
            symbol="TEST",
            name="Test",
            current_price=158.0,
            price_change_1d=0.0,
            price_change_1d_pct=0.0,
            price_change_5d_pct=0.0,
            volume=5000000,
            avg_volume=2000000,
            high_52w=160.0,
            low_52w=80.0,
            market_cap=None,
            fetched_at=datetime.now(),
            category="stock",
        )

        notes = checker._get_notes(data)
        assert "52w HIGH" in notes
        assert "HIGH VOLUME" in notes

    def test_commodity_map_completeness(self):
        """Test that common commodities are mapped."""
        assert "gold" in COMMODITY_MAP
        assert "silver" in COMMODITY_MAP
        assert "oil" in COMMODITY_MAP
        assert "natural gas" in COMMODITY_MAP
        assert "copper" in COMMODITY_MAP

    def test_crypto_map_completeness(self):
        """Test that common cryptos are mapped."""
        assert "bitcoin" in CRYPTO_MAP
        assert "btc" in CRYPTO_MAP
        assert "ethereum" in CRYPTO_MAP
        assert "eth" in CRYPTO_MAP

    def test_index_map_completeness(self):
        """Test that major indices are mapped."""
        assert "s&p" in INDEX_MAP
        assert "nasdaq" in INDEX_MAP
        assert "dow" in INDEX_MAP
        assert "vix" in INDEX_MAP
