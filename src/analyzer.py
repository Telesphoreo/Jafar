"""
NLP Analyzer Module using spaCy.

Performs Named Entity Recognition (NER) to identify trending topics
from broad Twitter searches. This is the "Investigator" phase that
extracts Organizations, Geopolitical Entities, and Products.

SETUP REQUIRED:
Before running, download the spaCy model:
    python -m spacy download en_core_web_sm
"""

import logging
from collections import Counter
from dataclasses import dataclass

import spacy
from spacy.language import Language

from .scraper import ScrapedTweet

logger = logging.getLogger("twitter_sentiment.analyzer")

# Entity types we're interested in for financial/economic analysis
RELEVANT_ENTITY_TYPES = {
    "ORG",      # Organizations (companies, agencies, institutions)
    "GPE",      # Geopolitical entities (countries, cities, states)
    "PRODUCT",  # Products (can include commodities, financial products)
    "NORP",     # Nationalities, religious/political groups
    "EVENT",    # Named events (could be financial events)
    "MONEY",    # Monetary values
    "PERCENT",  # Percentages
}

# Entities to exclude (too common or not useful)
EXCLUDED_ENTITIES = {
    "the",
    "a",
    "an",
    "us",
    "usa",
    "u.s.",
    "u.s",
    "america",
    "rt",
    "via",
    "today",
    "yesterday",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "this",
    "that",
    "these",
    "those",
    "http",
    "https",
}


@dataclass
class ExtractedEntity:
    """Represents an extracted named entity."""
    text: str
    label: str
    count: int
    sample_contexts: list[str]

    def __str__(self) -> str:
        return f"{self.text} ({self.label}): {self.count} mentions"


class TrendAnalyzer:
    """
    Analyzes tweets using spaCy NER to identify trending entities.

    This is the "Investigator" that processes broad search results
    to find specific trending topics for deeper analysis.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the analyzer with a spaCy model.

        Args:
            model_name: Name of the spaCy model to load.
                       Run `python -m spacy download en_core_web_sm` first.
        """
        self.model_name = model_name
        self._nlp: Language | None = None
        logger.info(f"TrendAnalyzer initialized with model: {model_name}")

    def _get_nlp(self) -> Language:
        """Lazy load the spaCy model."""
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

    def _clean_entity_text(self, text: str) -> str:
        """Clean and normalize entity text."""
        # Remove leading/trailing whitespace and convert to title case
        cleaned = text.strip()

        # Remove @ mentions and # symbols
        if cleaned.startswith("@") or cleaned.startswith("#"):
            cleaned = cleaned[1:]

        # Remove common prefixes
        for prefix in ["$", "the ", "a ", "an "]:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]

        return cleaned.strip()

    def _is_valid_entity(self, text: str, label: str) -> bool:
        """Check if an entity is valid and relevant."""
        cleaned = text.lower().strip()

        # Skip if too short
        if len(cleaned) < 2:
            return False

        # Skip excluded entities
        if cleaned in EXCLUDED_ENTITIES:
            return False

        # Skip if mostly numbers (except for MONEY/PERCENT)
        if label not in {"MONEY", "PERCENT"}:
            alpha_chars = sum(1 for c in cleaned if c.isalpha())
            if alpha_chars < len(cleaned) * 0.5:
                return False

        # Skip URLs
        if "http" in cleaned or ".com" in cleaned or "www" in cleaned:
            return False

        return True

    def extract_entities(
        self,
        tweets: list[ScrapedTweet],
        entity_types: set[str] | None = None,
    ) -> list[ExtractedEntity]:
        """
        Extract named entities from a list of tweets.

        Args:
            tweets: List of tweets to analyze.
            entity_types: Set of entity types to extract.
                         Defaults to RELEVANT_ENTITY_TYPES.

        Returns:
            List of ExtractedEntity objects sorted by count.
        """
        if entity_types is None:
            entity_types = {"ORG", "GPE", "PRODUCT"}

        nlp = self._get_nlp()
        entity_counter: Counter[tuple[str, str]] = Counter()
        entity_contexts: dict[tuple[str, str], list[str]] = {}

        logger.info(f"Extracting entities from {len(tweets)} tweets")

        for tweet in tweets:
            # Skip retweets to avoid counting duplicates
            if tweet.is_retweet:
                continue

            # Process the tweet text
            doc = nlp(tweet.text)

            for ent in doc.ents:
                # Filter by entity type
                if ent.label_ not in entity_types:
                    continue

                # Clean and validate the entity
                cleaned_text = self._clean_entity_text(ent.text)
                if not self._is_valid_entity(cleaned_text, ent.label_):
                    continue

                # Use title case for consistency
                normalized = cleaned_text.title()
                key = (normalized, ent.label_)

                entity_counter[key] += 1

                # Store sample contexts (up to 3)
                if key not in entity_contexts:
                    entity_contexts[key] = []
                if len(entity_contexts[key]) < 3:
                    entity_contexts[key].append(tweet.text[:200])

        # Convert to ExtractedEntity objects
        entities = [
            ExtractedEntity(
                text=text,
                label=label,
                count=count,
                sample_contexts=entity_contexts.get((text, label), []),
            )
            for (text, label), count in entity_counter.most_common()
        ]

        logger.info(f"Extracted {len(entities)} unique entities")
        return entities

    def extract_trends(
        self,
        tweets: list[ScrapedTweet],
        top_n: int = 5,
    ) -> list[str]:
        """
        Extract the top trending entities from tweets.

        This is the main method for the "Investigator" phase.
        It identifies the most frequently mentioned organizations,
        locations, and products for deeper sentiment analysis.

        Args:
            tweets: List of tweets from broad search.
            top_n: Number of top trends to return.

        Returns:
            List of top trending entity names.
        """
        logger.info(f"Extracting top {top_n} trends from {len(tweets)} tweets")

        # Extract entities focusing on ORG, GPE, and PRODUCT
        entities = self.extract_entities(
            tweets,
            entity_types={"ORG", "GPE", "PRODUCT"},
        )

        # Get top N entities by count
        top_entities = entities[:top_n]

        # Log the trends found
        for entity in top_entities:
            logger.info(f"Trend found: {entity}")

        # Return just the entity names
        trends = [entity.text for entity in top_entities]

        logger.info(f"Top trends: {trends}")
        return trends

    def get_entity_summary(self, entities: list[ExtractedEntity]) -> str:
        """
        Generate a summary of extracted entities.

        Args:
            entities: List of extracted entities.

        Returns:
            Formatted string summary.
        """
        if not entities:
            return "No entities found."

        summary_parts = ["=== Entity Analysis ===\n"]

        # Group by entity type
        by_type: dict[str, list[ExtractedEntity]] = {}
        for entity in entities:
            if entity.label not in by_type:
                by_type[entity.label] = []
            by_type[entity.label].append(entity)

        # Format each type
        type_labels = {
            "ORG": "Organizations",
            "GPE": "Locations",
            "PRODUCT": "Products",
            "NORP": "Groups",
            "EVENT": "Events",
            "MONEY": "Monetary Values",
            "PERCENT": "Percentages",
        }

        for label, type_entities in by_type.items():
            label_name = type_labels.get(label, label)
            summary_parts.append(f"\n{label_name}:")
            for entity in type_entities[:5]:  # Top 5 per type
                summary_parts.append(f"  - {entity.text}: {entity.count} mentions")

        return "\n".join(summary_parts)
