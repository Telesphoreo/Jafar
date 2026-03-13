"""Shared async SQLAlchemy engine and session factory."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(url: str) -> None:
    """Initialize the async engine and session factory.

    Converts postgresql:// URLs to postgresql+asyncpg:// automatically.
    """
    global _engine, _session_factory

    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    _engine = create_async_engine(url)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Return a new async session from the factory."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


def get_engine() -> AsyncEngine:
    """Return the async engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


async def close_db() -> None:
    """Dispose the engine and release connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def create_tables() -> None:
    """Create all tables defined in the ORM models."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
