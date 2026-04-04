# pylint: disable=not-callable
"""
VectorChord PostgreSQL Vector Store Implementation.

Uses SQLAlchemy async with VectorChord (pgvector + vchord) for vector similarity search.
"""

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func, select, text

from src.database import create_tables, get_engine, get_session
from src.models import MemoryRecord as MemoryRecordModel

from .base import MemoryRecord, SearchResult, VectorStore

logger = logging.getLogger("jafar.memory.pgvector")


class PgVectorStore(VectorStore):
    """
    PostgreSQL + VectorChord implementation of the vector store.

    Uses SQLAlchemy async sessions from src.database instead of
    managing its own connection pool.
    """

    def __init__(self, embedding_dimension: int = 3072):
        """
        Initialize pgvector store.

        Args:
            embedding_dimension: Dimension of embeddings (default 3072).
        """
        self.embedding_dimension = embedding_dimension

    async def initialize(self) -> None:
        """Create the VectorChord extensions, tables, and index."""
        engine = get_engine()

        # Create pgvector + vchord extensions (VectorChord depends on pgvector)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vchord CASCADE"))

        # Create ORM tables
        await create_tables()

        # Create VectorChord RaBitQ index on the embedding column
        async with engine.begin() as conn:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_memory_embedding "
                "ON memory_records USING vchordrq (embedding vector_cosine_ops)"
            ))

        count = await self.count()
        logger.info(f"VectorChord initialized with {count} existing memories")

    async def store(self, record: MemoryRecord, embedding: list[float]) -> str:
        """Store a memory record with its embedding via SQLAlchemy merge."""
        session = await get_session()
        async with session.begin():
            orm_record = MemoryRecordModel(
                id=record.id,
                date=record.date,
                content=record.full_digest,
                summary=record.summary,
                trends=record.trends,
                signal_strength=record.signal_strength,
                notable=record.notable,
                metadata_json={
                    "sentiment": record.sentiment,
                    "themes": record.themes,
                    "trend_categories": record.trend_categories,
                    "tweet_count": record.tweet_count,
                    "top_engagement": record.top_engagement,
                },
                embedding=embedding,
            )
            await session.merge(orm_record)

        await session.close()
        logger.info(f"Stored memory: {record.id}")
        return record.id

    def _to_dataclass(self, row: MemoryRecordModel) -> MemoryRecord:
        """Convert an ORM MemoryRecord model instance to the base MemoryRecord dataclass."""
        meta = row.metadata_json or {}
        return MemoryRecord(
            id=row.id,
            date=row.date,
            trends=row.trends or [],
            trend_categories=meta.get("trend_categories", []),
            signal_strength=row.signal_strength or "none",
            sentiment=meta.get("sentiment", "neutral"),
            top_engagement=meta.get("top_engagement", 0.0),
            themes=meta.get("themes", []),
            summary=row.summary or "",
            full_digest=row.content or "",
            notable=row.notable,
            tweet_count=meta.get("tweet_count", 0),
        )

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.6,
    ) -> list[SearchResult]:
        """Search for similar memories using cosine similarity."""
        query_vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        sql = text("""
            SELECT *,
                1 - (embedding <=> :query_vec::vector) AS similarity
            FROM memory_records
            WHERE 1 - (embedding <=> :query_vec::vector) >= :min_sim
            ORDER BY similarity DESC
            LIMIT :lim
        """)

        session = await get_session()
        try:
            result = await session.execute(
                sql,
                {"query_vec": query_vec_str, "min_sim": min_similarity, "lim": limit},
            )
            rows = result.mappings().all()
        finally:
            await session.close()

        results = []
        for row in rows:
            record = MemoryRecord(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                id=row["id"],
                date=row["date"],
                content=row["content"],
                summary=row["summary"] or "",
                trends=row["trends"] or [],
                signal_strength=row["signal_strength"] or "none",
                notable=row["notable"],
                metadata=row["metadata_json"],
            )
            similarity = float(row["similarity"])
            results.append(SearchResult(
                record=record,
                similarity=similarity,
                distance=1.0 - similarity,
            ))

        return results

    async def get_by_date(self, date_str: str) -> Optional[MemoryRecord]:  # pylint: disable=arguments-renamed
        """Get a specific memory by id (date string like '20240115')."""
        session = await get_session()
        try:
            result = await session.get(MemoryRecordModel, date_str)
        finally:
            await session.close()

        if result is None:
            return None
        return self._to_dataclass(result)

    async def get_recent(self, days: int = 7) -> list[MemoryRecord]:
        """Get recent memories."""
        cutoff = date.today() - timedelta(days=days)

        session = await get_session()
        try:
            stmt = (
                select(MemoryRecordModel)
                .where(MemoryRecordModel.date >= cutoff)
                .order_by(MemoryRecordModel.date.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
        finally:
            await session.close()

        return [self._to_dataclass(row) for row in rows]

    async def get_notable(self, limit: int = 10) -> list[MemoryRecord]:
        """Get notable/significant memories."""
        session = await get_session()
        try:
            stmt = (
                select(MemoryRecordModel)
                .where(MemoryRecordModel.notable.is_(True))
                .order_by(MemoryRecordModel.date.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
        finally:
            await session.close()

        return [self._to_dataclass(row) for row in rows]

    async def count(self) -> int:
        """Get total number of stored memories."""
        session = await get_session()
        try:
            result = await session.execute(
                select(func.count()).select_from(MemoryRecordModel)
            )
            return result.scalar_one()
        finally:
            await session.close()

    async def close(self) -> None:
        """No-op — engine lifecycle is managed by src.database."""
