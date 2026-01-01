"""
Statistical Trend Discovery Module.

COMPLETELY DIFFERENT APPROACH: Instead of keyword matching, this module:
1. Extracts ALL n-grams (1-3 word phrases) from tweets
2. Scores them by engagement velocity and frequency
3. Filters out eternal noise (Fed, Trump, etc.)
4. Surfaces whatever is STATISTICALLY ANOMALOUS right now

This finds trucks, silver, shipping containers, whatever - we don't need to know
what to look for in advance.

SETUP REQUIRED:
    python -m spacy download en_core_web_sm
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .scraper import ScrapedTweet

logger = logging.getLogger("jafar.analyzer")

# =============================================================================
# NOISE FILTER - Terms that are ALWAYS mentioned (no signal value)
# =============================================================================
NOISE_TERMS = {
    # Government/Political (always present, never actionable)
    "federal reserve", "fed", "the fed", "fomc", "powell", "jerome powell",
    "congress", "senate", "house", "white house", "government",
    "trump", "donald trump", "donald t", "don t", "don trump", "donaldtrump",
    "biden", "joe biden", "harris", "kamala", "obama", "desantis", "musk", "elon",
    "gop", "republican", "republicans", "democrat", "democrats", "democratic",
    "president", "vice president", "treasury", "sec", "ftc", "doj", "cftc",
    "yellen", "janet yellen",

    # Countries (too generic)
    "us", "usa", "u.s.", "u.s", "america", "united states", "american",
    "china", "chinese", "russia", "russian", "uk", "eu", "european union",
    "canada", "canadian", "mexico", "mexican", "japan", "japanese",
    "germany", "german", "france", "french", "india", "indian",

    # Generic market structure terms (always discussed, no signal value)
    "wall street", "main street", "stock market", "market", "markets",
    "stocks", "stock", "shares", "share", "equity", "equities",
    "trade", "trading", "trader", "traders",
    "investor", "investors", "investing", "investment",
    "portfolio", "position", "positions",

    # NOTE: Sentiment terms like "bullish", "bearish", "rally", "crash", "recession",
    # "inflation" are intentionally NOT filtered - spikes in these indicate market mood.
    # We want to catch when bearish sentiment is unusually high, etc.

    # Time words
    "today", "yesterday", "tomorrow", "week", "month", "year", "daily",
    "weekly", "monthly", "yearly", "annual", "quarter", "quarterly",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night",

    # Common filler words
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "this", "that", "these", "those",
    "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "i", "me", "my", "he", "she", "his", "her", "who", "what", "when",
    "where", "why", "how", "which", "all", "each", "every", "both",
    "new", "old", "big", "small", "high", "low", "up", "down", "out", "in",
    "more", "less", "most", "least", "very", "just", "only", "even",
    "now", "then", "here", "there", "also", "too", "so", "yet", "still",
    "first", "last", "next", "other", "another", "same", "different",

    # Generic nouns that appear everywhere (no signal value)
    "thing", "things", "stuff", "something", "anything", "everything", "nothing",
    "world", "life", "way", "ways", "day", "days", "night", "nights",
    "time", "times", "moment", "moments", "hour", "hours", "minute", "minutes",
    "people", "person", "man", "men", "woman", "women", "guy", "guys", "kid", "kids",
    "group", "groups", "team", "teams", "part", "parts", "side", "sides",
    "place", "places", "area", "areas", "point", "points", "end", "ends",
    "case", "cases", "fact", "facts", "reason", "reasons", "example", "examples",
    "kind", "kinds", "type", "types", "sort", "sorts", "lot", "lots",
    "step", "steps", "move", "moves", "action", "actions", "change", "changes",
    "experience", "experiences", "story", "stories", "situation", "situations",
    "problem", "problems", "issue", "issues", "question", "questions", "answer", "answers",
    "idea", "ideas", "thought", "thoughts", "opinion", "opinions", "view", "views",
    "work", "works", "job", "jobs", "business", "businesses", "service", "services",
    "system", "systems", "process", "processes", "plan", "plans", "project", "projects",
    "result", "results", "effect", "effects", "impact", "impacts", "outcome", "outcomes",
    "level", "levels", "rate", "rates", "number", "numbers", "amount", "amounts",
    "worth", "value", "values", "cost", "costs", "money", "cash", "funds",
    "right", "rights", "left", "wrong", "good", "bad", "best", "worst", "better", "worse",
    "start", "beginning", "middle", "end", "top", "bottom", "front", "back",
    "head", "hand", "hands", "eye", "eyes", "face", "body", "mind", "heart",
    "home", "house", "room", "office", "school", "city", "country", "state",
    "game", "games", "play", "video", "media", "content", "post", "posts",
    "chat", "message", "messages", "call", "calls", "email", "emails",
    "word", "words", "name", "names", "title", "titles", "term", "terms",
    "don", "dont", "don t", "won", "wont", "won t", "cant", "can t",
    "didnt", "didn t", "doesnt", "doesn t", "isnt", "isn t", "arent", "aren t",
    "wouldn", "wouldn t", "couldn", "couldn t", "shouldn", "shouldn t",
    "hasn", "hasn t", "haven", "haven t", "hadn", "hadn t",
    "gonna", "gotta", "wanna", "kinda", "sorta", "dunno", "idk", "imo", "imho",
    "lol", "lmao", "omg", "wtf", "smh", "tbh", "ngl", "fr", "rn", "af",

    # Common first names (no signal value)
    "donald", "mark", "john", "james", "michael", "david", "robert", "william",
    "richard", "joseph", "thomas", "charles", "chris", "daniel", "matt", "matthew",
    "anthony", "brian", "kevin", "steve", "steven", "paul", "andrew", "josh",
    "mary", "jennifer", "lisa", "sarah", "karen", "nancy", "betty", "susan",

    # Seasonal/holiday noise
    "christmas", "xmas", "holiday", "holidays", "thanksgiving", "easter",
    "new year", "new years", "halloween", "valentine", "valentines",
    "birthday", "anniversary", "weekend", "vacation",

    # Social media noise
    "rt", "via", "breaking", "just in", "news", "update", "thread",
    "http", "https", "www", "com", "link", "click", "follow", "retweet",
    "like", "share", "comment", "subscribe", "dm", "tweet", "twitter",

    # Media outlets
    "cnbc", "bloomberg", "reuters", "wsj", "cnn", "fox", "msnbc",
    "yahoo", "marketwatch", "seeking alpha", "zerohedge", "financial times",
    "wall street journal", "barrons", "investors business daily",

    # Generic company terms
    "inc", "corp", "llc", "ltd", "co", "company", "companies", "corporation",
    "ceo", "cfo", "coo", "executive", "management", "board", "director",

    # Overused tech buzzwords (REDUCED - these ARE relevant when spiking alongside hardware/supply discussion)
    # Removed: "ai", "openai" - now conditionally allowed if high financial context
    "chatgpt", "gpt", "machine learning",

    # Crypto (unless specifically wanted - can be toggled)
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency", "blockchain",

    # Numbers and percentages (not meaningful alone)
    "percent", "percentage", "%", "million", "billion", "trillion",
    "thousand", "hundred", "dollar", "dollars", "usd", "price", "prices",
}

# Compile patterns for faster matching
CASHTAG_PATTERN = re.compile(r'\$([A-Za-z]{1,5})\b')
HASHTAG_PATTERN = re.compile(r'#(\w+)')
URL_PATTERN = re.compile(r'https?://\S+')
MENTION_PATTERN = re.compile(r'@\w+')

# Economic context indicators - if a tweet contains these, its terms are economically relevant
FINANCIAL_CONTEXT_TERMS = {
    # Traditional market terms
    "market", "trading", "stock", "buy", "sell", "long", "short",
    "bullish", "bearish", "rally", "crash", "earnings", "revenue", "profit",
    "loss", "dividend", "yield", "bond", "treasury", "fed",
    "recession", "gdp", "commodity", "futures", "options", "puts", "calls",
    "breakout", "support", "resistance", "volume", "volatility", "etf",
    "portfolio", "hedge", "risk", "sector", "index", "dow", "nasdaq", "s&p",
    "gold", "silver", "oil", "copper", "wheat", "corn", "lumber",

    # CONSUMER PRICES (critical inflation/margin signals)
    "price", "prices", "pricing", "cost", "costs", "expensive", "cheap",
    "afford", "affordability", "msrp", "retail price", "launch price",
    "price increase", "price hike", "price cut", "discount", "sale",
    "inflation", "deflation", "cost of living",

    # SUPPLY/DEMAND (availability = pricing power)
    "supply", "demand", "shortage", "surplus", "allocation", "backlog",
    "sold out", "in stock", "available", "wait list", "backorder",
    "production", "capacity", "inventory", "stockpile",

    # CONSUMER SPENDING (recession/boom signals)
    "spending", "sales", "revenue", "consumer", "retail",
    "buying", "purchased", "splurge", "cutting back", "budget",
    "worth it", "value", "deal",

    # EMPLOYMENT/WAGES (income side)
    "hiring", "layoffs", "jobs", "unemployment", "employment",
    "wage", "wages", "salary", "pay", "raise", "pay cut",

    # BUSINESS OPERATIONS (margin/profitability signals)
    "capex", "margin", "margins", "gross margin", "operating margin",
    "shipments", "deliveries", "orders", "guidance",

    # TECH/HARDWARE (AI/semiconductor economy)
    "foundry", "wafer", "chip", "semiconductor", "gpu", "hardware",
    "datacenter", "cloud", "infrastructure",
}


@dataclass
class DiscoveredTrend:
    """A statistically discovered trend with scoring metrics."""
    term: str
    term_type: str  # 'cashtag', 'hashtag', 'ngram', 'entity'
    mention_count: int
    unique_authors: int
    total_engagement: float
    avg_engagement: float
    sample_tweets: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    financial_context_count: int = 0  # How many mentions had financial context
    cashtag_cooccurrence_count: int = 0  # How many mentions appeared WITH a cashtag

    @property
    def financial_context_ratio(self) -> float:
        """Ratio of mentions that appeared in financial context."""
        return self.financial_context_count / max(self.mention_count, 1)

    @property
    def cashtag_cooccurrence_ratio(self) -> float:
        """Ratio of mentions that appeared alongside a cashtag."""
        return self.cashtag_cooccurrence_count / max(self.mention_count, 1)

    @property
    def velocity_score(self) -> float:
        """
        Score based on engagement velocity.
        High unique authors + high engagement = organic trend, not spam.
        """
        # Penalize if same accounts posting repeatedly (potential spam)
        author_ratio = self.unique_authors / max(self.mention_count, 1)

        # Base score from engagement
        engagement_score = self.total_engagement

        # Boost for diverse authorship (organic spread)
        diversity_bonus = author_ratio * 2

        return engagement_score * (1 + diversity_bonus)

    @property
    def composite_score(self) -> float:
        """
        Final ranking score combining all factors.
        Financial context is heavily weighted - trends without financial context
        are heavily penalized to filter entertainment/noise.
        """
        # Frequency matters but engagement matters more
        freq_component = self.mention_count * 0.2
        engagement_component = self.velocity_score * 0.8

        base_score = freq_component + engagement_component

        # Financial context multiplier: 0.01x to 1.5x
        # Stricter thresholds to filter generic words that occasionally appear in financial tweets
        # < 40% financial context = heavily penalized (0.01x - 0.2x)
        # 40-70% = moderate (0.2x - 0.8x)
        # > 70% financial context = boosted (0.8x - 1.5x)
        if self.financial_context_ratio < 0.4:
            context_multiplier = 0.01 + (self.financial_context_ratio * 0.5)
        elif self.financial_context_ratio < 0.7:
            context_multiplier = 0.2 + (self.financial_context_ratio * 0.85)
        else:
            context_multiplier = 0.8 + (self.financial_context_ratio * 0.7)

        # Cashtag co-occurrence bonus: 1.0x to 1.5x
        # Terms that appear alongside tickers are more likely to be financially relevant
        cooccurrence_bonus = 1.0 + (self.cashtag_cooccurrence_ratio * 0.5)

        return base_score * context_multiplier * cooccurrence_bonus

    def passes_quality_threshold(self, min_authors: int = 10, min_context: float = 0.65) -> bool:
        """
        Check if this trend passes minimum quality requirements for deep dive.

        Requirements:
        - At least min_authors unique authors (default 10)
        - At least min_context financial context ratio (default 65% - lowered from 85% to catch hardware/infrastructure trends)
        - For non-cashtags: should have SOME cashtag co-occurrence (proves financial relevance)
        """
        # Basic thresholds
        if self.unique_authors < min_authors:
            return False
        if self.financial_context_ratio < min_context:
            return False

        # Cashtags always pass (they ARE the financial signal)
        if self.term_type == "cashtag":
            return True

        # For n-grams and hashtags: require some cashtag co-occurrence
        # This proves the term is actually discussed in financial contexts
        # Exception: if it has very high engagement and perfect financial context
        if self.cashtag_cooccurrence_ratio < 0.1:
            # Allow if it has overwhelming signal (100% financial, high engagement)
            if self.financial_context_ratio >= 0.95 and self.total_engagement > 20000:
                return True
            return False

        return True

    def __str__(self) -> str:
        return (
            f"{self.term} ({self.term_type}): "
            f"{self.mention_count} mentions by {self.unique_authors} authors, "
            f"engagement={self.total_engagement:.0f}, "
            f"fin_ctx={self.financial_context_ratio:.0%}, "
            f"cashtag_co={self.cashtag_cooccurrence_ratio:.0%}"
        )


class StatisticalTrendAnalyzer:
    """
    Statistical anomaly detector for Twitter trends.

    Instead of keyword matching, this extracts ALL meaningful terms and
    ranks them by engagement velocity to find what's HOT right now.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp: Language | None = None
        logger.info("StatisticalTrendAnalyzer initialized")

    def _get_nlp(self) -> Language:
        """Lazy load spaCy model."""
        if self._nlp is None:
            logger.info(f"Loading spaCy model: {self.model_name}")
            try:
                self._nlp = spacy.load(self.model_name)
                logger.info("spaCy model loaded successfully")
            except OSError as e:
                logger.error(
                    f"Failed to load spaCy model. Run: python -m spacy download {self.model_name}"
                )
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Please run: python -m spacy download {self.model_name}"
                ) from e
        return self._nlp

    def _is_noise(self, term: str) -> bool:
        """Check if term should be filtered out."""
        normalized = term.lower().strip()

        # Too short
        if len(normalized) < 2:
            return True

        # Direct match with noise terms
        if normalized in NOISE_TERMS:
            return True

        # Pure numbers
        if normalized.replace('.', '').replace(',', '').replace('%', '').replace('$', '').isdigit():
            return True

        # URLs
        if 'http' in normalized or '.com' in normalized or 'www' in normalized:
            return True

        # Check partial matches for multi-word noise
        for noise in NOISE_TERMS:
            if len(noise) > 3 and noise in normalized:
                return True

        return False

    def _calculate_engagement(self, tweet: ScrapedTweet) -> float:
        """Calculate engagement score for a tweet."""
        return (tweet.likes * 1.0) + (tweet.retweets * 0.5) + (tweet.replies * 0.3)

    def _has_financial_context(self, tweet: ScrapedTweet) -> bool:
        """Check if tweet contains financial context indicators."""
        text_lower = tweet.text.lower()
        # Has cashtag = definitely financial
        if CASHTAG_PATTERN.search(tweet.text):
            return True
        # Contains financial terms
        return any(term in text_lower for term in FINANCIAL_CONTEXT_TERMS)

    def _has_cashtag(self, tweet: ScrapedTweet) -> bool:
        """Check if tweet contains any cashtag ($TICKER)."""
        return bool(CASHTAG_PATTERN.search(tweet.text))

    def _clean_text(self, text: str) -> str:
        """Remove URLs, mentions, and clean up text for n-gram extraction."""
        text = URL_PATTERN.sub('', text)
        text = MENTION_PATTERN.sub('', text)
        # Keep cashtags and hashtags but clean other noise
        text = re.sub(r'[^\w\s$#\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_cashtags(self, tweets: list[ScrapedTweet]) -> dict[str, DiscoveredTrend]:
        """Extract cashtags - highest signal for markets."""
        trends: dict[str, DiscoveredTrend] = {}
        author_tracker: dict[str, set[str]] = defaultdict(set)

        for tweet in tweets:
            if tweet.is_retweet:
                continue

            matches = CASHTAG_PATTERN.findall(tweet.text)
            engagement = self._calculate_engagement(tweet)

            for match in matches:
                ticker = match.upper()

                if self._is_noise(ticker):
                    continue

                key = ticker
                author_tracker[key].add(tweet.username)

                if key not in trends:
                    trends[key] = DiscoveredTrend(
                        term=f"${ticker}",
                        term_type="cashtag",
                        mention_count=0,
                        unique_authors=0,
                        total_engagement=0,
                        avg_engagement=0,
                        sample_tweets=[],
                        first_seen=tweet.created_at,
                        last_seen=tweet.created_at,
                    )

                trends[key].mention_count += 1
                trends[key].total_engagement += engagement
                trends[key].last_seen = tweet.created_at
                # Cashtags are inherently financial context
                trends[key].financial_context_count += 1

                if len(trends[key].sample_tweets) < 3:
                    trends[key].sample_tweets.append(tweet.text[:200])

        # Update unique author counts
        for key, trend in trends.items():
            trend.unique_authors = len(author_tracker[key])
            trend.avg_engagement = trend.total_engagement / max(trend.mention_count, 1)

        return trends

    def _extract_hashtags(self, tweets: list[ScrapedTweet]) -> dict[str, DiscoveredTrend]:
        """Extract meaningful hashtags."""
        trends: dict[str, DiscoveredTrend] = {}
        author_tracker: dict[str, set[str]] = defaultdict(set)

        for tweet in tweets:
            if tweet.is_retweet:
                continue

            engagement = self._calculate_engagement(tweet)
            has_financial_context = self._has_financial_context(tweet)
            has_cashtag = self._has_cashtag(tweet)

            for hashtag in tweet.hashtags:
                tag = hashtag.lower().strip('#')

                if self._is_noise(tag) or len(tag) < 3 or len(tag) > 25:
                    continue

                author_tracker[tag].add(tweet.username)

                if tag not in trends:
                    trends[tag] = DiscoveredTrend(
                        term=f"#{tag}",
                        term_type="hashtag",
                        mention_count=0,
                        unique_authors=0,
                        total_engagement=0,
                        avg_engagement=0,
                        sample_tweets=[],
                        first_seen=tweet.created_at,
                        last_seen=tweet.created_at,
                    )

                trends[tag].mention_count += 1
                trends[tag].total_engagement += engagement
                trends[tag].last_seen = tweet.created_at
                if has_financial_context:
                    trends[tag].financial_context_count += 1
                if has_cashtag:
                    trends[tag].cashtag_cooccurrence_count += 1

                if len(trends[tag].sample_tweets) < 3:
                    trends[tag].sample_tweets.append(tweet.text[:200])

        for key, trend in trends.items():
            trend.unique_authors = len(author_tracker[key])
            trend.avg_engagement = trend.total_engagement / max(trend.mention_count, 1)

        return trends

    def _extract_ngrams(self, tweets: list[ScrapedTweet]) -> dict[str, DiscoveredTrend]:
        """
        Extract meaningful n-grams (1-3 words) using spaCy for better tokenization.
        This is the CORE discovery mechanism - finds terms we didn't know to look for.
        """
        nlp = self._get_nlp()
        trends: dict[str, DiscoveredTrend] = {}
        author_tracker: dict[str, set[str]] = defaultdict(set)

        # Process tweets in batches for efficiency
        texts = []
        tweet_data = []

        for tweet in tweets:
            if tweet.is_retweet:
                continue
            cleaned = self._clean_text(tweet.text)
            if cleaned:
                texts.append(cleaned)
                tweet_data.append(tweet)

        logger.info(f"Processing {len(texts)} tweets for n-gram extraction...")

        # Process with spaCy
        for doc, tweet in zip(nlp.pipe(texts, batch_size=100), tweet_data):
            engagement = self._calculate_engagement(tweet)
            has_financial_context = self._has_financial_context(tweet)
            has_cashtag = self._has_cashtag(tweet)

            # Extract noun phrases and named entities - these are the meaningful terms
            meaningful_terms = set()

            # Named entities (ORG, PRODUCT, EVENT, etc.)
            # Note: WORK_OF_ART excluded - catches entertainment (movies, anime, songs)
            for ent in doc.ents:
                if ent.label_ in {"ORG", "PRODUCT", "EVENT", "FAC", "GPE", "LOC"}:
                    term = ent.text.strip()
                    if len(term) >= 2 and not self._is_noise(term):
                        meaningful_terms.add(term.lower())

            # Noun chunks (noun phrases)
            for chunk in doc.noun_chunks:
                # Filter out chunks that are just determiners or pronouns
                root = chunk.root
                if root.pos_ in {"NOUN", "PROPN"}:
                    term = chunk.text.strip()
                    if len(term) >= 2 and not self._is_noise(term):
                        meaningful_terms.add(term.lower())

            # Individual nouns and proper nouns (single important words)
            for token in doc:
                if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                    term = token.text.strip()
                    if len(term) >= 3 and not self._is_noise(term):
                        meaningful_terms.add(term.lower())

            # Add each meaningful term
            for term in meaningful_terms:
                author_tracker[term].add(tweet.username)

                if term not in trends:
                    # Use title case for display
                    display_term = term.title() if not term.startswith(('$', '#')) else term
                    trends[term] = DiscoveredTrend(
                        term=display_term,
                        term_type="ngram",
                        mention_count=0,
                        unique_authors=0,
                        total_engagement=0,
                        avg_engagement=0,
                        sample_tweets=[],
                        first_seen=tweet.created_at,
                        last_seen=tweet.created_at,
                    )

                trends[term].mention_count += 1
                trends[term].total_engagement += engagement
                trends[term].last_seen = tweet.created_at
                if has_financial_context:
                    trends[term].financial_context_count += 1
                if has_cashtag:
                    trends[term].cashtag_cooccurrence_count += 1

                if len(trends[term].sample_tweets) < 3:
                    trends[term].sample_tweets.append(tweet.text[:200])

        for key, trend in trends.items():
            trend.unique_authors = len(author_tracker[key])
            trend.avg_engagement = trend.total_engagement / max(trend.mention_count, 1)

        return trends

    def extract_trends_with_details(
        self,
        tweets: list[ScrapedTweet],
        top_n: int = 10,
        min_mentions: int = 3,
        min_authors: int = 2,
        apply_quality_filter: bool = True,
    ) -> tuple[list[DiscoveredTrend], list[DiscoveredTrend]]:
        """
        Extract trending topics with full details and quality filtering.

        Args:
            tweets: List of scraped tweets to analyze
            top_n: Maximum number of trends to consider
            min_mentions: Minimum mention count to be considered
            min_authors: Minimum unique authors (filters spam)
            apply_quality_filter: If True, filter by quality threshold

        Returns:
            Tuple of (qualified_trends, all_candidates)
            - qualified_trends: Trends that pass quality threshold (for deep dive)
            - all_candidates: Top N trends before quality filter (for logging/debugging)
        """
        logger.info(f"Analyzing {len(tweets)} tweets for emerging trends...")

        all_trends: list[DiscoveredTrend] = []

        # 1. Cashtags (highest signal)
        cashtags = self._extract_cashtags(tweets)
        logger.info(f"Found {len(cashtags)} unique cashtags")
        all_trends.extend(cashtags.values())

        # 2. Hashtags
        hashtags = self._extract_hashtags(tweets)
        logger.info(f"Found {len(hashtags)} unique hashtags")
        all_trends.extend(hashtags.values())

        # 3. N-grams (discovered terms)
        ngrams = self._extract_ngrams(tweets)
        logger.info(f"Found {len(ngrams)} unique n-grams/entities")
        all_trends.extend(ngrams.values())

        # Filter by minimum thresholds
        filtered = [
            t for t in all_trends
            if t.mention_count >= min_mentions and t.unique_authors >= min_authors
        ]
        logger.info(f"{len(filtered)} trends pass minimum thresholds")

        # Sort by composite score
        filtered.sort(key=lambda x: x.composite_score, reverse=True)

        # Deduplicate (same term from different sources)
        seen_terms = set()
        unique_trends: list[DiscoveredTrend] = []
        for trend in filtered:
            normalized = trend.term.lower().strip('$#')
            if normalized not in seen_terms:
                seen_terms.add(normalized)
                unique_trends.append(trend)

        # Get top N candidates
        candidates = unique_trends[:top_n]

        # Log all candidates
        logger.info("=" * 60)
        logger.info("TOP EMERGING TRENDS (by engagement velocity):")
        for i, trend in enumerate(candidates, 1):
            logger.info(f"  {i}. {trend}")
        logger.info("=" * 60)

        # Apply quality filter
        if apply_quality_filter:
            qualified = [t for t in candidates if t.passes_quality_threshold()]
            logger.info(f"Quality filter: {len(qualified)}/{len(candidates)} trends pass threshold")

            if qualified:
                logger.info("QUALIFIED FOR DEEP DIVE:")
                for i, trend in enumerate(qualified, 1):
                    logger.info(f"  {i}. {trend.term}")
            else:
                logger.warning("No trends passed quality threshold - quiet day or noisy data")
        else:
            qualified = candidates

        return qualified, candidates

    def extract_trends(
        self,
        tweets: list[ScrapedTweet],
        top_n: int = 10,
        min_mentions: int = 3,
        min_authors: int = 2,
    ) -> list[str]:
        """
        Extract top trending topics using statistical analysis.

        Args:
            tweets: List of scraped tweets to analyze
            top_n: Number of top trends to return
            min_mentions: Minimum mention count to be considered
            min_authors: Minimum unique authors (filters spam)

        Returns:
            List of top trending terms (only those passing quality threshold)
        """
        qualified, _ = self.extract_trends_with_details(
            tweets=tweets,
            top_n=top_n,
            min_mentions=min_mentions,
            min_authors=min_authors,
            apply_quality_filter=True,
        )
        return [t.term for t in qualified]

    def get_detailed_analysis(self, tweets: list[ScrapedTweet], top_n: int = 15) -> str:
        """Get detailed breakdown of all discovered signals."""
        cashtags = self._extract_cashtags(tweets)
        hashtags = self._extract_hashtags(tweets)
        ngrams = self._extract_ngrams(tweets)

        lines = ["=== STATISTICAL TREND DISCOVERY REPORT ===\n"]
        lines.append(f"Analyzed {len(tweets)} tweets\n")

        lines.append("\n[CASHTAGS] - Highest signal for market moves:")
        sorted_cashtags = sorted(cashtags.values(), key=lambda x: x.composite_score, reverse=True)
        for trend in sorted_cashtags[:top_n]:
            lines.append(
                f"  {trend.term}: {trend.mention_count} mentions, "
                f"{trend.unique_authors} authors, "
                f"engagement={trend.total_engagement:.0f}"
            )

        lines.append("\n[HASHTAGS] - Topic clustering:")
        sorted_hashtags = sorted(hashtags.values(), key=lambda x: x.composite_score, reverse=True)
        for trend in sorted_hashtags[:top_n]:
            lines.append(
                f"  {trend.term}: {trend.mention_count} mentions, "
                f"{trend.unique_authors} authors, "
                f"engagement={trend.total_engagement:.0f}"
            )

        lines.append("\n[DISCOVERED TERMS] - Statistically significant phrases:")
        sorted_ngrams = sorted(ngrams.values(), key=lambda x: x.composite_score, reverse=True)
        for trend in sorted_ngrams[:top_n]:
            lines.append(
                f"  {trend.term}: {trend.mention_count} mentions, "
                f"{trend.unique_authors} authors, "
                f"engagement={trend.total_engagement:.0f}"
            )

        return "\n".join(lines)


# Keep backward compatibility alias
TrendAnalyzer = StatisticalTrendAnalyzer
