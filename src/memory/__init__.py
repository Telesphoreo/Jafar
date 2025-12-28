"""
Vector Memory System for Historical Context.

This module provides semantic search over historical digests,
enabling the system to find true parallels - not just keyword matches.

"History doesn't repeat itself, but it often rhymes." - Mark Twain
"""

from .base import MemoryRecord, VectorStore, SearchResult
from .embeddings import EmbeddingService, create_embedding_service
from .chroma_store import ChromaVectorStore
from .memory_manager import MemoryManager, create_memory_manager

__all__ = [
    "MemoryRecord",
    "VectorStore",
    "SearchResult",
    "EmbeddingService",
    "create_embedding_service",
    "ChromaVectorStore",
    "MemoryManager",
    "create_memory_manager",
]
