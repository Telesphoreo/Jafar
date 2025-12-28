"""
Base interfaces and data structures for vector memory.

Defines the abstract contracts that different vector store
backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MemoryRecord:
    """
    A structured memory of a single day's market analysis.

    This is what gets embedded and stored - designed to capture
    the ESSENCE of what happened, not just raw text.
    """
    # Identity
    id: str  # Unique ID (usually date-based)
    date: datetime

    # What was discovered
    trends: list[str]  # Top trends found
    trend_categories: list[str]  # Categorized: commodity, sector, stock, etc.

    # Market context
    signal_strength: str  # high, medium, low, none
    sentiment: str  # bullish, bearish, neutral, mixed
    top_engagement: float

    # The narrative - this is key for semantic search
    themes: list[str]  # Key themes/narratives (e.g., "supply shortage", "fed pivot")
    summary: str  # 2-3 sentence summary of what happened

    # Full content (for reference, not embedding)
    full_digest: str

    # Metadata
    notable: bool = False
    tweet_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_embedding_text(self) -> str:
        """
        Create the text that will be embedded for similarity search.

        This is carefully structured to enable semantic matching on
        what actually matters - themes, trends, and market conditions.
        """
        parts = [
            f"Date: {self.date.strftime('%Y-%m-%d')}",
            f"Signal: {self.signal_strength}",
            f"Sentiment: {self.sentiment}",
            f"Trends: {', '.join(self.trends)}",
            f"Categories: {', '.join(self.trend_categories)}",
            f"Themes: {', '.join(self.themes)}",
            f"Summary: {self.summary}",
        ]
        return "\n".join(parts)

    def to_context_string(self) -> str:
        """
        Format this memory for inclusion in LLM context.
        """
        notable_marker = " [NOTABLE DAY]" if self.notable else ""
        return f"""
### {self.date.strftime('%B %d, %Y')}{notable_marker}
- **Signal**: {self.signal_strength.upper()}
- **Sentiment**: {self.sentiment}
- **Top Trends**: {', '.join(self.trends[:5])}
- **Themes**: {', '.join(self.themes)}
- **Summary**: {self.summary}
"""


@dataclass
class SearchResult:
    """A search result from the vector store."""
    record: MemoryRecord
    similarity: float  # 0-1, higher is more similar
    distance: float  # Raw distance metric

    @property
    def is_strong_match(self) -> bool:
        """Is this a strong enough match to mention?"""
        return self.similarity > 0.75

    @property
    def is_moderate_match(self) -> bool:
        """Is this a moderate match worth considering?"""
        return self.similarity > 0.6


class VectorStore(ABC):
    """
    Abstract interface for vector storage backends.

    Implementations: ChromaDB (local), pgvector (production)
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store (create collections, etc.)."""
        pass

    @abstractmethod
    async def store(self, record: MemoryRecord, embedding: list[float]) -> str:
        """
        Store a memory record with its embedding.

        Args:
            record: The structured memory record
            embedding: The vector embedding

        Returns:
            The ID of the stored record
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """
        Search for similar memories.

        Args:
            query_embedding: The embedding to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results, ordered by similarity
        """
        pass

    @abstractmethod
    async def get_by_date(self, date: datetime) -> Optional[MemoryRecord]:
        """Get a specific memory by date."""
        pass

    @abstractmethod
    async def get_recent(self, days: int = 7) -> list[MemoryRecord]:
        """Get recent memories."""
        pass

    @abstractmethod
    async def get_notable(self, limit: int = 10) -> list[MemoryRecord]:
        """Get notable/significant memories."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total number of stored memories."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass
