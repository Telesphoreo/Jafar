"""
Memory Manager - Orchestrates the vector memory system.

This is the high-level interface that the main pipeline uses.
It handles:
- Creating structured memories from raw analysis
- Generating embeddings
- Searching for historical parallels
- Formatting context for the LLM
"""

import logging
import re
from datetime import datetime
from typing import Literal, Optional

from .base import MemoryRecord, SearchResult, VectorStore
from .embeddings import EmbeddingService, create_embedding_service
from .chroma_store import ChromaVectorStore

logger = logging.getLogger("twitter_sentiment.memory.manager")


class MemoryManager:
    """
    High-level memory management for the sentiment analysis system.

    Provides semantic search over historical market conditions
    to find genuine parallels - not just keyword matches.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self._initialized = False
        logger.info("MemoryManager created")

    async def initialize(self) -> None:
        """Initialize the memory system."""
        await self.vector_store.initialize()
        self._initialized = True
        count = await self.vector_store.count()
        logger.info(f"MemoryManager initialized with {count} stored memories")

    def _ensure_initialized(self) -> None:
        """Ensure the system is initialized."""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

    def _extract_themes(self, analysis: str) -> list[str]:
        """
        Extract key themes from the LLM analysis.

        Looks for patterns like:
        - Bullet points under "Themes" section
        - Key phrases in the summary
        - Trend descriptions
        """
        themes = []

        # Look for explicit themes section
        themes_match = re.search(
            r'\*\*TRENDS OBSERVED\*\*:?\s*\n(.*?)(?=\n\*\*|\Z)',
            analysis,
            re.DOTALL | re.IGNORECASE,
        )
        if themes_match:
            section = themes_match.group(1)
            # Extract bullet points
            bullets = re.findall(r'[-•]\s*(.+?)(?:\n|$)', section)
            themes.extend([b.strip() for b in bullets if len(b.strip()) > 3])

        # Also look for key phrases in summary
        summary_match = re.search(
            r'\*\*(?:BOTTOM LINE|SUMMARY)\*\*:?\s*(.+?)(?:\n\*\*|\Z)',
            analysis,
            re.DOTALL | re.IGNORECASE,
        )
        if summary_match:
            # Extract noun phrases (simplified)
            summary = summary_match.group(1)
            # Add key terms from summary
            key_terms = re.findall(r'(?:about|regarding|concerning)\s+([^,.]+)', summary, re.IGNORECASE)
            themes.extend([t.strip() for t in key_terms])

        # Deduplicate while preserving order
        seen = set()
        unique_themes = []
        for theme in themes:
            normalized = theme.lower()
            if normalized not in seen and len(theme) > 3:
                seen.add(normalized)
                unique_themes.append(theme)

        return unique_themes[:10]  # Limit to top 10 themes

    def _extract_sentiment(self, analysis: str) -> str:
        """Extract overall sentiment from analysis."""
        analysis_lower = analysis.lower()

        # Look for explicit sentiment
        if "bullish" in analysis_lower and "bearish" not in analysis_lower:
            return "bullish"
        elif "bearish" in analysis_lower and "bullish" not in analysis_lower:
            return "bearish"
        elif "mixed" in analysis_lower or ("bullish" in analysis_lower and "bearish" in analysis_lower):
            return "mixed"
        else:
            return "neutral"

    def _categorize_trends(self, trends: list[str]) -> list[str]:
        """Categorize trends into types."""
        categories = set()

        commodity_keywords = {"gold", "silver", "oil", "copper", "wheat", "corn", "uranium", "lithium"}
        sector_keywords = {"tech", "biotech", "pharma", "energy", "solar", "ev", "semiconductor", "defense"}

        for trend in trends:
            trend_lower = trend.lower()

            if trend.startswith("$"):
                categories.add("stock")
            elif trend.startswith("#"):
                categories.add("hashtag")
            elif any(kw in trend_lower for kw in commodity_keywords):
                categories.add("commodity")
            elif any(kw in trend_lower for kw in sector_keywords):
                categories.add("sector")
            else:
                categories.add("general")

        return list(categories)

    def _extract_summary(self, analysis: str) -> str:
        """Extract the bottom line summary from analysis."""
        # Look for explicit summary sections
        patterns = [
            r'\*\*BOTTOM LINE\*\*:?\s*(.+?)(?:\n\*\*|\Z)',
            r'\*\*SUMMARY\*\*:?\s*(.+?)(?:\n\*\*|\Z)',
            r'\*\*ASSESSMENT\*\*:?\s*(.+?)(?:\n\*\*|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                # Clean up and limit length
                summary = re.sub(r'\s+', ' ', summary)
                return summary[:500]

        # Fallback: first paragraph
        paragraphs = analysis.split('\n\n')
        if paragraphs:
            return paragraphs[0][:500]

        return "No summary available"

    async def create_memory(
        self,
        trends: list[str],
        analysis: str,
        signal_strength: str,
        top_engagement: float,
        tweet_count: int,
        notable: bool = False,
        date: Optional[datetime] = None,
    ) -> MemoryRecord:
        """
        Create a structured memory from analysis results.

        Args:
            trends: List of discovered trends
            analysis: The full LLM analysis
            signal_strength: Signal strength rating
            top_engagement: Highest engagement seen
            tweet_count: Total tweets analyzed
            notable: Whether this day was notable
            date: Date of analysis (defaults to now)

        Returns:
            A structured MemoryRecord ready for embedding
        """
        self._ensure_initialized()

        if date is None:
            date = datetime.now()

        # Create structured memory
        record = MemoryRecord(
            id=f"memory_{date.strftime('%Y%m%d')}",
            date=date,
            trends=trends,
            trend_categories=self._categorize_trends(trends),
            signal_strength=signal_strength,
            sentiment=self._extract_sentiment(analysis),
            top_engagement=top_engagement,
            themes=self._extract_themes(analysis),
            summary=self._extract_summary(analysis),
            full_digest=analysis,
            notable=notable,
            tweet_count=tweet_count,
        )

        return record

    async def store_memory(self, record: MemoryRecord) -> str:
        """
        Store a memory with its embedding.

        Args:
            record: The structured memory record

        Returns:
            The ID of the stored memory
        """
        self._ensure_initialized()

        # Generate embedding from the structured text
        embedding_text = record.to_embedding_text()
        embedding = await self.embedding_service.embed(embedding_text)

        # Store in vector database
        record_id = await self.vector_store.store(record, embedding)

        logger.info(f"Stored memory {record_id} with {len(embedding)}-dim embedding")
        return record_id

    async def find_parallels(
        self,
        trends: list[str],
        themes: list[str],
        sentiment: str,
        signal_strength: str,
        limit: int = 5,
        min_similarity: float = 0.6,
    ) -> list[SearchResult]:
        """
        Find historical parallels to current market conditions.

        Args:
            trends: Current trends discovered
            themes: Current themes/narratives
            sentiment: Current sentiment
            signal_strength: Current signal strength
            limit: Maximum parallels to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar historical memories
        """
        self._ensure_initialized()

        # Construct query from current conditions
        query_text = "\n".join([
            f"Trends: {', '.join(trends)}",
            f"Themes: {', '.join(themes)}",
            f"Sentiment: {sentiment}",
            f"Signal: {signal_strength}",
        ])

        # Generate embedding for query
        query_embedding = await self.embedding_service.embed(query_text)

        # Search for similar memories
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )

        # Filter out today's memory if present
        today_id = f"memory_{datetime.now().strftime('%Y%m%d')}"
        results = [r for r in results if r.record.id != today_id]

        logger.info(f"Found {len(results)} historical parallels")
        for r in results:
            logger.debug(f"  - {r.record.date.strftime('%Y-%m-%d')}: similarity={r.similarity:.3f}")

        return results

    async def format_parallels_for_llm(
        self,
        parallels: list[SearchResult],
        max_parallels: int = 3,
    ) -> str:
        """
        Format historical parallels for inclusion in LLM context.

        This is carefully designed to encourage thoughtful comparison,
        not force-fitting parallels where none exist.
        """
        if not parallels:
            return """
## Historical Context
No strong historical parallels found in the database.
This could mean:
- This is a genuinely novel situation
- The memory database is still building up
- Current conditions are unique enough to not match past patterns

Proceed with analysis based on current data only.
"""

        strong_matches = [p for p in parallels if p.is_strong_match]
        moderate_matches = [p for p in parallels if p.is_moderate_match and not p.is_strong_match]

        lines = [
            "## Historical Context",
            "",
            "The following historical periods show POTENTIAL similarity to current conditions.",
            "**IMPORTANT**: Only mention these if the parallel is genuinely meaningful.",
            "History rhymes, but doesn't always repeat. Don't force a connection.",
            "",
        ]

        if strong_matches:
            lines.append("### Strong Parallels (>75% similarity)")
            for match in strong_matches[:max_parallels]:
                lines.append(match.record.to_context_string())
                lines.append(f"*Similarity: {match.similarity:.1%}*")
                lines.append("")

        if moderate_matches and len(strong_matches) < max_parallels:
            lines.append("### Moderate Parallels (60-75% similarity)")
            remaining = max_parallels - len(strong_matches)
            for match in moderate_matches[:remaining]:
                lines.append(match.record.to_context_string())
                lines.append(f"*Similarity: {match.similarity:.1%}*")
                lines.append("")

        lines.extend([
            "---",
            "When considering these parallels, ask:",
            "- Is the similarity superficial (same keywords) or substantive (similar dynamics)?",
            "- What happened AFTER these historical periods?",
            "- Are the underlying conditions truly comparable?",
            "",
        ])

        return "\n".join(lines)

    async def get_recent_context(self, days: int = 7) -> str:
        """
        Get recent memories for baseline context.

        This provides continuity - what has the system been tracking?
        """
        self._ensure_initialized()

        recent = await self.vector_store.get_recent(days=days)
        count = await self.vector_store.count()

        if not recent:
            return f"""
## Memory Status
Total memories stored: {count}
Recent memories (last {days} days): 0

The system is still building its memory. After several days of operation,
it will be able to provide historical context and identify parallels.
"""

        lines = [
            f"## Memory Status",
            f"Total memories stored: {count}",
            f"Recent memories (last {days} days): {len(recent)}",
            "",
            "### Recent Days:",
        ]

        for memory in recent[:5]:
            notable = " [NOTABLE]" if memory.notable else ""
            lines.append(
                f"- {memory.date.strftime('%Y-%m-%d')}: "
                f"Signal={memory.signal_strength.upper()}{notable} - "
                f"{', '.join(memory.trends[:3])}"
            )

        return "\n".join(lines)

    async def close(self) -> None:
        """Clean up resources."""
        await self.vector_store.close()
        logger.info("MemoryManager closed")


async def create_memory_manager(
    store_type: Literal["chroma", "pgvector"] = "chroma",
    embedding_provider: Literal["openai", "local"] = "openai",
    openai_api_key: str = "",
    postgres_url: str = "",
    chroma_path: str = "./memory_store",
) -> MemoryManager:
    """
    Factory function to create a configured MemoryManager.

    Args:
        store_type: "chroma" for local, "pgvector" for production
        embedding_provider: "openai" or "local"
        openai_api_key: Required for OpenAI embeddings
        postgres_url: Required for pgvector store
        chroma_path: Path for ChromaDB storage

    Returns:
        Initialized MemoryManager
    """
    # Create embedding service
    embedding_service = create_embedding_service(
        provider=embedding_provider,
        api_key=openai_api_key,
    )

    # Create vector store
    if store_type == "chroma":
        vector_store = ChromaVectorStore(persist_directory=chroma_path)
    elif store_type == "pgvector":
        if not postgres_url:
            raise ValueError("postgres_url required for pgvector store")
        from .pgvector_store import PgVectorStore
        vector_store = PgVectorStore(
            connection_string=postgres_url,
            embedding_dimension=embedding_service.dimension,
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")

    # Create and initialize manager
    manager = MemoryManager(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    await manager.initialize()
    return manager
