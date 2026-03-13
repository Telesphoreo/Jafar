"""SQLAlchemy ORM models for Jafar."""

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

EMBEDDING_DIM = 3072


class VectorType(UserDefinedType):
    """Custom SQLAlchemy type for pgvector/VectorChord vector columns."""

    cache_ok = True

    def __init__(self, dim: int):
        self.dim = dim

    def get_col_spec(self) -> str:
        return f"vector({self.dim})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            return f"[{','.join(str(v) for v in value)}]"

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, list):
                return value
            return [float(x) for x in value.strip("[]").split(",")]

        return process


class Base(DeclarativeBase):
    pass


class Digest(Base):
    """Stores daily digest reports."""

    __tablename__ = "digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    trends: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    tweet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    digest_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    signal_strength: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    top_engagement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notable: Mapped[bool] = mapped_column(Boolean, default=False)
    trend_details: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class TrendHistory(Base):
    """Tracks trend mentions and engagement over time."""

    __tablename__ = "trend_history"
    __table_args__ = (UniqueConstraint("trend_term", "date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trend_term: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    mentions: Mapped[int] = mapped_column(Integer, default=0)
    engagement: Mapped[float] = mapped_column(Float, default=0.0)


class Tweet(Base):
    """Stores scraped tweets for ML training data (bot detection, account scoring)."""

    __tablename__ = "tweets"
    __table_args__ = (
        Index("ix_tweets_username_created_at", "username", "created_at"),
        Index("ix_tweets_pipeline_run_source_query", "pipeline_run", "source_query"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    retweets: Mapped[int] = mapped_column(Integer, default=0)
    replies: Mapped[int] = mapped_column(Integer, default=0)
    views: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    hashtags: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_query: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    pipeline_run: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)


class BotJudgment(Base):
    """Stores LLM bot judgments for building labeled datasets."""

    __tablename__ = "bot_judgments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    bot_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    classification: Mapped[str] = mapped_column(String(20), nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    signals: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    ml_bot_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    judged_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    pipeline_run: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)


class AppState(Base):
    """Simple key-value store for pipeline state."""

    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class MemoryRecord(Base):
    """Stores digest memories with vector embeddings for semantic search."""

    __tablename__ = "memory_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trends: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    signal_strength: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    notable: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    embedding: Mapped[Optional[Any]] = mapped_column(
        VectorType(EMBEDDING_DIM), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
