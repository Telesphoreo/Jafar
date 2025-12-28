"""
pgvector PostgreSQL Vector Store Implementation.

Production-ready vector storage using PostgreSQL with pgvector extension.

Advantages over ChromaDB:
- Battle-tested PostgreSQL infrastructure
- Full SQL querying capabilities
- Better for large scale (> 1M vectors)
- ACID compliance
- Easy to integrate with existing Postgres infrastructure

Requirements:
- PostgreSQL with pgvector extension installed
- asyncpg for async PostgreSQL access
- DATABASE_URL environment variable or explicit connection string
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from .base import MemoryRecord, SearchResult, VectorStore

logger = logging.getLogger("twitter_sentiment.memory.pgvector")


class PgVectorStore(VectorStore):
    """
    PostgreSQL + pgvector implementation of the vector store.

    Production-ready vector storage with full SQL capabilities.
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "market_memories",
        embedding_dimension: int = 1536,
    ):
        """
        Initialize pgvector store.

        Args:
            connection_string: PostgreSQL connection string
                e.g., "postgresql://user:pass@localhost:5432/dbname"
            table_name: Name of the table to store memories
            embedding_dimension: Dimension of embeddings (1536 for OpenAI small)
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self._pool = None
        logger.info(f"PgVectorStore configured with table: {table_name}")

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool and create tables."""
        try:
            import asyncpg
        except ImportError:
            raise RuntimeError(
                "asyncpg not installed. Install with: pip install asyncpg"
            )

        # Create connection pool
        self._pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=1,
            max_size=10,
        )

        # Create pgvector extension and table
        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create memories table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    date TIMESTAMP NOT NULL,
                    embedding vector({self.embedding_dimension}),
                    trends JSONB NOT NULL,
                    trend_categories JSONB NOT NULL,
                    signal_strength TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    top_engagement FLOAT NOT NULL,
                    themes JSONB NOT NULL,
                    summary TEXT NOT NULL,
                    full_digest TEXT NOT NULL,
                    notable BOOLEAN NOT NULL DEFAULT FALSE,
                    tweet_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)

            # Create indexes for common queries
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_date
                ON {self.table_name}(date DESC)
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_notable
                ON {self.table_name}(notable) WHERE notable = TRUE
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_signal
                ON {self.table_name}(signal_strength)
            """)

            # Create vector index for similarity search using HNSW
            # HNSW is faster, more accurate, and supports higher dimensions than ivfflat
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding_hnsw
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

        count = await self.count()
        logger.info(f"pgvector initialized with {count} existing memories")

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if self._pool is None:
            raise RuntimeError("PgVectorStore not initialized. Call initialize() first.")

    async def store(self, record: MemoryRecord, embedding: list[float]) -> str:
        """Store a memory record with its embedding."""
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            # Upsert the record
            await conn.execute(f"""
                INSERT INTO {self.table_name} (
                    id, date, embedding, trends, trend_categories,
                    signal_strength, sentiment, top_engagement,
                    themes, summary, full_digest, notable, tweet_count, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    date = EXCLUDED.date,
                    embedding = EXCLUDED.embedding,
                    trends = EXCLUDED.trends,
                    trend_categories = EXCLUDED.trend_categories,
                    signal_strength = EXCLUDED.signal_strength,
                    sentiment = EXCLUDED.sentiment,
                    top_engagement = EXCLUDED.top_engagement,
                    themes = EXCLUDED.themes,
                    summary = EXCLUDED.summary,
                    full_digest = EXCLUDED.full_digest,
                    notable = EXCLUDED.notable,
                    tweet_count = EXCLUDED.tweet_count
            """,
                record.id,
                record.date,
                str(embedding),  # pgvector accepts string representation
                json.dumps(record.trends),
                json.dumps(record.trend_categories),
                record.signal_strength,
                record.sentiment,
                record.top_engagement,
                json.dumps(record.themes),
                record.summary,
                record.full_digest,
                record.notable,
                record.tweet_count,
                record.created_at,
            )

        logger.info(f"Stored memory: {record.id}")
        return record.id

    def _row_to_record(self, row) -> MemoryRecord:
        """Convert a database row to a MemoryRecord."""
        return MemoryRecord(
            id=row["id"],
            date=row["date"],
            trends=json.loads(row["trends"]) if isinstance(row["trends"], str) else row["trends"],
            trend_categories=json.loads(row["trend_categories"]) if isinstance(row["trend_categories"], str) else row["trend_categories"],
            signal_strength=row["signal_strength"],
            sentiment=row["sentiment"],
            top_engagement=row["top_engagement"],
            themes=json.loads(row["themes"]) if isinstance(row["themes"], str) else row["themes"],
            summary=row["summary"],
            full_digest=row["full_digest"],
            notable=row["notable"],
            tweet_count=row["tweet_count"],
            created_at=row["created_at"],
        )

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """Search for similar memories using cosine similarity."""
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            # Use cosine distance (1 - similarity)
            # pgvector <=> operator returns cosine distance
            rows = await conn.fetch(f"""
                SELECT *,
                    1 - (embedding <=> $1::vector) as similarity,
                    embedding <=> $1::vector as distance
                FROM {self.table_name}
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """,
                str(query_embedding),
                min_similarity,
                limit,
            )

        results = []
        for row in rows:
            record = self._row_to_record(row)
            results.append(SearchResult(
                record=record,
                similarity=row["similarity"],
                distance=row["distance"],
            ))

        return results

    async def get_by_date(self, date: datetime) -> Optional[MemoryRecord]:
        """Get a specific memory by date."""
        self._ensure_initialized()

        date_id = f"memory_{date.strftime('%Y%m%d')}"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT * FROM {self.table_name} WHERE id = $1
            """, date_id)

        if row:
            return self._row_to_record(row)
        return None

    async def get_recent(self, days: int = 7) -> list[MemoryRecord]:
        """Get recent memories."""
        self._ensure_initialized()

        cutoff = datetime.now() - timedelta(days=days)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM {self.table_name}
                WHERE date >= $1
                ORDER BY date DESC
            """, cutoff)

        return [self._row_to_record(row) for row in rows]

    async def get_notable(self, limit: int = 10) -> list[MemoryRecord]:
        """Get notable/significant memories."""
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM {self.table_name}
                WHERE notable = TRUE
                ORDER BY date DESC
                LIMIT $1
            """, limit)

        return [self._row_to_record(row) for row in rows]

    async def count(self) -> int:
        """Get total number of stored memories."""
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT COUNT(*) as count FROM {self.table_name}
            """)

        return row["count"]

    async def close(self) -> None:
        """Clean up resources."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        logger.info("pgvector connection closed")
