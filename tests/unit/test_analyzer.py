"""
Unit tests for src/analyzer.py

Tests statistical trend discovery, n-gram extraction, and quality filtering.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer import (
    NOISE_TERMS,
    FINANCIAL_CONTEXT_TERMS,
    DiscoveredTrend,
    StatisticalTrendAnalyzer,
)
from src.scraper import ScrapedTweet


class TestDiscoveredTrend:
    """Tests for DiscoveredTrend dataclass."""

    def test_basic_creation(self):
        """Test creating a DiscoveredTrend."""
        trend = DiscoveredTrend(
            term="$NVDA",
            term_type="cashtag",
            mention_count=50,
            unique_authors=30,
            total_engagement=10000.0,
            avg_engagement=200.0,
        )

        assert trend.term == "$NVDA"
        assert trend.term_type == "cashtag"
        assert trend.mention_count == 50

    def test_financial_context_ratio(self):
        """Test financial context ratio calculation."""
        trend = DiscoveredTrend(
            term="Test",
            term_type="ngram",
            mention_count=100,
            unique_authors=50,
            total_engagement=5000.0,
            avg_engagement=50.0,
            financial_context_count=80,
        )

        assert trend.financial_context_ratio == 0.8

    def test_cashtag_cooccurrence_ratio(self):
        """Test cashtag co-occurrence ratio."""
        trend = DiscoveredTrend(
            term="Silver",
            term_type="ngram",
            mention_count=100,
            unique_authors=50,
            total_engagement=5000.0,
            avg_engagement=50.0,
            cashtag_cooccurrence_count=30,
        )

        assert trend.cashtag_cooccurrence_ratio == 0.3

    def test_velocity_score(self):
        """Test velocity score calculation."""
        trend = DiscoveredTrend(
            term="$AAPL",
            term_type="cashtag",
            mention_count=100,
            unique_authors=80,  # High diversity
            total_engagement=10000.0,
            avg_engagement=100.0,
        )

        # High author diversity should boost score
        assert trend.velocity_score > 0

    def test_composite_score_high_context(self):
        """Test composite score with high financial context."""
        trend = DiscoveredTrend(
            term="Shortage",
            term_type="ngram",
            mention_count=50,
            unique_authors=40,
            total_engagement=8000.0,
            avg_engagement=160.0,
            financial_context_count=45,  # 90% context
            cashtag_cooccurrence_count=20,  # 40% co-occurrence
        )

        score = trend.composite_score
        assert score > 0

    def test_composite_score_low_context(self):
        """Test composite score with low financial context."""
        trend = DiscoveredTrend(
            term="Movie",
            term_type="ngram",
            mention_count=100,
            unique_authors=80,
            total_engagement=20000.0,
            avg_engagement=200.0,
            financial_context_count=10,  # Only 10% context
            cashtag_cooccurrence_count=0,
        )

        # Low context should heavily penalize score
        score = trend.composite_score
        # Score should be much lower than raw engagement would suggest
        assert score < 5000

    def test_passes_quality_threshold_cashtag(self):
        """Test that cashtags pass quality threshold easily."""
        trend = DiscoveredTrend(
            term="$NVDA",
            term_type="cashtag",
            mention_count=20,
            unique_authors=15,
            total_engagement=5000.0,
            avg_engagement=250.0,
            financial_context_count=20,  # 100% - inherent for cashtags
        )

        assert trend.passes_quality_threshold(min_authors=10) is True

    def test_passes_quality_threshold_ngram_requires_context(self):
        """Test that n-grams require financial context."""
        trend = DiscoveredTrend(
            term="Shortage",
            term_type="ngram",
            mention_count=50,
            unique_authors=30,
            total_engagement=10000.0,
            avg_engagement=200.0,
            financial_context_count=40,  # 80% context
            cashtag_cooccurrence_count=15,  # 30% co-occurrence
        )

        assert trend.passes_quality_threshold(min_authors=10, min_context=0.65) is True

    def test_fails_quality_threshold_low_authors(self):
        """Test failure with too few authors."""
        trend = DiscoveredTrend(
            term="$LOW",
            term_type="cashtag",
            mention_count=50,
            unique_authors=5,  # Too few
            total_engagement=5000.0,
            avg_engagement=100.0,
            financial_context_count=50,
        )

        assert trend.passes_quality_threshold(min_authors=10) is False

    def test_fails_quality_threshold_low_context(self):
        """Test failure with low financial context."""
        trend = DiscoveredTrend(
            term="Entertainment",
            term_type="ngram",
            mention_count=100,
            unique_authors=50,
            total_engagement=20000.0,
            avg_engagement=200.0,
            financial_context_count=20,  # Only 20% context
            cashtag_cooccurrence_count=5,
        )

        assert trend.passes_quality_threshold(min_context=0.65) is False

    def test_str_representation(self):
        """Test string representation."""
        trend = DiscoveredTrend(
            term="$NVDA",
            term_type="cashtag",
            mention_count=50,
            unique_authors=30,
            total_engagement=10000.0,
            avg_engagement=200.0,
            financial_context_count=50,
            cashtag_cooccurrence_count=50,
        )

        str_repr = str(trend)
        assert "$NVDA" in str_repr
        assert "cashtag" in str_repr
        assert "50 mentions" in str_repr


class TestStatisticalTrendAnalyzer:
    """Tests for StatisticalTrendAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = StatisticalTrendAnalyzer(model_name="en_core_web_sm")
        assert analyzer.model_name == "en_core_web_sm"
        assert analyzer._nlp is None  # Lazy loaded

    def test_is_noise_filters_common_words(self):
        """Test that common noise words are filtered."""
        analyzer = StatisticalTrendAnalyzer()

        assert analyzer._is_noise("the") is True
        assert analyzer._is_noise("trump") is True
        assert analyzer._is_noise("congress") is True
        assert analyzer._is_noise("stock market") is True

    def test_is_noise_allows_financial_terms(self):
        """Test that financial terms are not filtered."""
        analyzer = StatisticalTrendAnalyzer()

        # Note: These are NOT in NOISE_TERMS
        assert analyzer._is_noise("shortage") is False
        assert analyzer._is_noise("supply chain") is False

    def test_is_noise_filters_urls(self):
        """Test that URLs are filtered."""
        analyzer = StatisticalTrendAnalyzer()

        assert analyzer._is_noise("https://example.com") is True
        assert analyzer._is_noise("www.test.com") is True

    def test_is_noise_filters_pure_numbers(self):
        """Test that pure numbers are filtered."""
        analyzer = StatisticalTrendAnalyzer()

        assert analyzer._is_noise("12345") is True
        assert analyzer._is_noise("99.99%") is True
        assert analyzer._is_noise("$1,000") is True

    def test_calculate_engagement(self):
        """Test engagement calculation."""
        analyzer = StatisticalTrendAnalyzer()

        tweet = ScrapedTweet(
            id=1,
            text="Test",
            username="user",
            display_name="User",
            created_at=datetime.now(),
            likes=100,
            retweets=50,
            replies=30,
            views=1000,
            language="en",
            is_retweet=False,
        )

        engagement = analyzer._calculate_engagement(tweet)
        # 100*1.0 + 50*0.5 + 30*0.3 = 100 + 25 + 9 = 134
        assert engagement == 134.0

    def test_has_financial_context(self):
        """Test financial context detection."""
        analyzer = StatisticalTrendAnalyzer()

        # Tweet with cashtag
        tweet_with_cashtag = ScrapedTweet(
            id=1,
            text="$NVDA breaking out!",
            username="user",
            display_name="User",
            created_at=datetime.now(),
            likes=100,
            retweets=50,
            replies=10,
            views=1000,
            language="en",
            is_retweet=False,
        )
        assert analyzer._has_financial_context(tweet_with_cashtag) is True

        # Tweet with financial terms
        tweet_with_terms = ScrapedTweet(
            id=2,
            text="Market rally continues, bullish momentum building",
            username="user",
            display_name="User",
            created_at=datetime.now(),
            likes=100,
            retweets=50,
            replies=10,
            views=1000,
            language="en",
            is_retweet=False,
        )
        assert analyzer._has_financial_context(tweet_with_terms) is True

        # Tweet without financial context
        tweet_no_context = ScrapedTweet(
            id=3,
            text="Just watched an amazing movie!",
            username="user",
            display_name="User",
            created_at=datetime.now(),
            likes=100,
            retweets=50,
            replies=10,
            views=1000,
            language="en",
            is_retweet=False,
        )
        assert analyzer._has_financial_context(tweet_no_context) is False

    def test_clean_text(self):
        """Test text cleaning."""
        analyzer = StatisticalTrendAnalyzer()

        text = "Check this https://example.com @user $NVDA #stocks wow!"
        cleaned = analyzer._clean_text(text)

        assert "https" not in cleaned
        assert "@user" not in cleaned
        assert "$NVDA" in cleaned  # Cashtags preserved
        assert "#stocks" in cleaned  # Hashtags preserved

    def test_extract_cashtags(self, financial_tweets):
        """Test cashtag extraction."""
        analyzer = StatisticalTrendAnalyzer()
        trends = analyzer._extract_cashtags(financial_tweets)

        # Should find cashtags from financial tweets
        assert len(trends) > 0
        # All should be cashtag type
        for trend in trends.values():
            assert trend.term_type == "cashtag"
            assert trend.term.startswith("$")

    def test_extract_hashtags(self, sample_tweets):
        """Test hashtag extraction."""
        analyzer = StatisticalTrendAnalyzer()
        trends = analyzer._extract_hashtags(sample_tweets)

        # Should find hashtags
        for trend in trends.values():
            assert trend.term_type == "hashtag"
            assert trend.term.startswith("#")

    @pytest.mark.slow
    def test_extract_ngrams_with_spacy(self, spacy_nlp, financial_tweets):
        """Test n-gram extraction with real spaCy model."""
        analyzer = StatisticalTrendAnalyzer()
        analyzer._nlp = spacy_nlp  # Use pre-loaded model

        trends = analyzer._extract_ngrams(financial_tweets)

        # Should find meaningful terms
        assert len(trends) > 0
        for trend in trends.values():
            assert trend.term_type == "ngram"

    def test_extract_trends_returns_list(self, sample_tweets):
        """Test that extract_trends returns term list."""
        with patch.object(StatisticalTrendAnalyzer, "_extract_ngrams") as mock_ngrams:
            mock_ngrams.return_value = {}

            analyzer = StatisticalTrendAnalyzer()
            trends = analyzer.extract_trends(
                sample_tweets,
                top_n=10,
                min_mentions=1,
                min_authors=1,
            )

            assert isinstance(trends, list)

    def test_extract_trends_with_details_returns_tuple(self, sample_tweets):
        """Test that extract_trends_with_details returns qualified and all candidates."""
        with patch.object(StatisticalTrendAnalyzer, "_extract_ngrams") as mock_ngrams:
            mock_ngrams.return_value = {}

            analyzer = StatisticalTrendAnalyzer()
            qualified, candidates = analyzer.extract_trends_with_details(
                sample_tweets,
                top_n=10,
                min_mentions=1,
                min_authors=1,
            )

            assert isinstance(qualified, list)
            assert isinstance(candidates, list)


class TestNoiseTerms:
    """Tests for noise term filtering."""

    def test_political_terms_filtered(self):
        """Test that political terms are in noise list."""
        political = ["trump", "biden", "congress", "senate", "republican", "democrat"]
        for term in political:
            assert term in NOISE_TERMS, f"{term} should be in NOISE_TERMS"

    def test_countries_filtered(self):
        """Test that country names are filtered."""
        # Only test the ones that are actually in the noise list
        countries = ["us", "usa", "china", "russia", "uk", "eu"]
        for term in countries:
            assert term in NOISE_TERMS, f"{term} should be in NOISE_TERMS"

    def test_generic_market_terms_filtered(self):
        """Test that generic market terms are filtered."""
        generic = ["stock market", "stocks", "trading", "investor"]
        for term in generic:
            assert term in NOISE_TERMS, f"{term} should be in NOISE_TERMS"


class TestFinancialContextTerms:
    """Tests for financial context term detection."""

    def test_price_terms_present(self):
        """Test that price-related terms are in context set."""
        price_terms = ["price", "prices", "expensive", "cheap", "inflation"]
        for term in price_terms:
            assert term in FINANCIAL_CONTEXT_TERMS, f"{term} should be in FINANCIAL_CONTEXT_TERMS"

    def test_supply_demand_terms_present(self):
        """Test that supply/demand terms are in context set."""
        supply_terms = ["shortage", "surplus", "supply", "demand", "sold out"]
        for term in supply_terms:
            assert term in FINANCIAL_CONTEXT_TERMS, f"{term} should be in FINANCIAL_CONTEXT_TERMS"

    def test_market_terms_present(self):
        """Test that market action terms are in context set."""
        market_terms = ["bullish", "bearish", "rally", "crash", "breakout"]
        for term in market_terms:
            assert term in FINANCIAL_CONTEXT_TERMS, f"{term} should be in FINANCIAL_CONTEXT_TERMS"
