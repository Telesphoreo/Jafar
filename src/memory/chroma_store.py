"""
ChromaDB Vector Store Implementation.

ChromaDB is perfect for local/development use:
- No server required
- Stores everything in a local directory
- Built-in persistence
- Good performance for moderate scale (< 1M vectors)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .base import MemoryRecord, SearchResult, VectorStore

logger = logging.getLogger("twitter_sentiment.memory.chroma")


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of the vector store.

    Stores memories locally with full persistence.
    """

    def __init__(
        self,
        persist_directory: str = "./memory_store",
        collection_name: str = "market_memories",
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        logger.info(f"ChromaVectorStore configured with directory: {persist_directory}")

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise RuntimeError(
                "chromadb not installed. Install with: pip install chromadb"
            )

        # Create persist directory if needed
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Market sentiment memory store"},
        )

        count = self._collection.count()
        logger.info(f"ChromaDB initialized with {count} existing memories")

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if self._collection is None:
            raise RuntimeError("ChromaVectorStore not initialized. Call initialize() first.")

    def _record_to_metadata(self, record: MemoryRecord) -> dict:
        """Convert a MemoryRecord to ChromaDB metadata."""
        return {
            "date": record.date.isoformat(),
            "trends": json.dumps(record.trends),
            "trend_categories": json.dumps(record.trend_categories),
            "signal_strength": record.signal_strength,
            "sentiment": record.sentiment,
            "top_engagement": record.top_engagement,
            "themes": json.dumps(record.themes),
            "summary": record.summary,
            "notable": record.notable,
            "tweet_count": record.tweet_count,
            "created_at": record.created_at.isoformat(),
        }

    def _metadata_to_record(self, id: str, metadata: dict, document: str) -> MemoryRecord:
        """Convert ChromaDB metadata back to a MemoryRecord."""
        return MemoryRecord(
            id=id,
            date=datetime.fromisoformat(metadata["date"]),
            trends=json.loads(metadata["trends"]),
            trend_categories=json.loads(metadata["trend_categories"]),
            signal_strength=metadata["signal_strength"],
            sentiment=metadata["sentiment"],
            top_engagement=metadata["top_engagement"],
            themes=json.loads(metadata["themes"]),
            summary=metadata["summary"],
            full_digest=document,
            notable=metadata["notable"],
            tweet_count=metadata["tweet_count"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
        )

    async def store(self, record: MemoryRecord, embedding: list[float]) -> str:
        """Store a memory record with its embedding."""
        self._ensure_initialized()

        # Check if record for this date already exists
        existing = self._collection.get(ids=[record.id])
        if existing["ids"]:
            # Update existing record
            self._collection.update(
                ids=[record.id],
                embeddings=[embedding],
                documents=[record.full_digest],
                metadatas=[self._record_to_metadata(record)],
            )
            logger.info(f"Updated existing memory: {record.id}")
        else:
            # Add new record
            self._collection.add(
                ids=[record.id],
                embeddings=[embedding],
                documents=[record.full_digest],
                metadatas=[self._record_to_metadata(record)],
            )
            logger.info(f"Stored new memory: {record.id}")

        return record.id

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """Search for similar memories."""
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        # ChromaDB uses distance (lower is better), we want similarity (higher is better)
        # For cosine distance: similarity = 1 - distance
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit * 2, self._collection.count()),  # Get extra to filter
            include=["documents", "metadatas", "distances"],
        )

        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - distance

                if similarity >= min_similarity:
                    record = self._metadata_to_record(
                        id=id,
                        metadata=results["metadatas"][0][i],
                        document=results["documents"][0][i],
                    )
                    search_results.append(SearchResult(
                        record=record,
                        similarity=similarity,
                        distance=distance,
                    ))

        # Sort by similarity and limit
        search_results.sort(key=lambda x: x.similarity, reverse=True)
        return search_results[:limit]

    async def get_by_date(self, date: datetime) -> Optional[MemoryRecord]:
        """Get a specific memory by date."""
        self._ensure_initialized()

        date_id = f"memory_{date.strftime('%Y%m%d')}"
        results = self._collection.get(
            ids=[date_id],
            include=["documents", "metadatas"],
        )

        if results["ids"]:
            return self._metadata_to_record(
                id=results["ids"][0],
                metadata=results["metadatas"][0],
                document=results["documents"][0],
            )
        return None

    async def get_recent(self, days: int = 7) -> list[MemoryRecord]:
        """Get recent memories."""
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        # Get all and filter (ChromaDB doesn't support date range queries well)
        results = self._collection.get(
            include=["documents", "metadatas"],
        )

        records = []
        for i, id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            if metadata["date"] >= cutoff_str:
                records.append(self._metadata_to_record(
                    id=id,
                    metadata=metadata,
                    document=results["documents"][i],
                ))

        # Sort by date descending
        records.sort(key=lambda x: x.date, reverse=True)
        return records

    async def get_notable(self, limit: int = 10) -> list[MemoryRecord]:
        """Get notable/significant memories."""
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        # Get all notable memories
        results = self._collection.get(
            where={"notable": True},
            include=["documents", "metadatas"],
        )

        records = []
        for i, id in enumerate(results["ids"]):
            records.append(self._metadata_to_record(
                id=id,
                metadata=results["metadatas"][i],
                document=results["documents"][i],
            ))

        # Sort by date descending and limit
        records.sort(key=lambda x: x.date, reverse=True)
        return records[:limit]

    async def count(self) -> int:
        """Get total number of stored memories."""
        self._ensure_initialized()
        return self._collection.count()

    async def close(self) -> None:
        """Clean up resources."""
        # ChromaDB PersistentClient handles cleanup automatically
        self._client = None
        self._collection = None
        logger.info("ChromaDB connection closed")
