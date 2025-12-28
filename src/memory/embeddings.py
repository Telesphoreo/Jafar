"""
Embedding Service for generating vector representations.

Uses OpenAI's embedding models by default, with support for
local models via sentence-transformers as a fallback.
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal

logger = logging.getLogger("twitter_sentiment.memory.embeddings")


class EmbeddingService(ABC):
    """Abstract interface for embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbeddingService(EmbeddingService):
    """
    OpenAI embedding service using text-embedding-3-small.

    This model provides excellent quality at low cost:
    - 1536 dimensions
    - $0.02 per 1M tokens
    - Fast inference
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._dimension = 1536 if "small" in model else 3072
        logger.info(f"OpenAIEmbeddingService initialized with model: {model}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        client = self._get_client()

        response = await client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []

        client = self._get_client()

        response = await client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


class LocalEmbeddingService(EmbeddingService):
    """
    Local embedding service using sentence-transformers.

    Falls back to this if no API key is provided.
    Uses all-MiniLM-L6-v2 by default (384 dimensions, fast, good quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # Default for MiniLM
        logger.info(f"LocalEmbeddingService initialized with model: {model_name}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded local embedding model: {self.model_name}")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


def create_embedding_service(
    provider: Literal["openai", "local"] = "openai",
    api_key: str = "",
    model: str = "",
) -> EmbeddingService:
    """
    Factory function to create the appropriate embedding service.

    Args:
        provider: "openai" or "local"
        api_key: OpenAI API key (required for openai provider)
        model: Model name (optional, uses defaults)

    Returns:
        Configured EmbeddingService instance
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required for openai embedding provider")
        return OpenAIEmbeddingService(
            api_key=api_key,
            model=model or "text-embedding-3-small",
        )
    elif provider == "local":
        return LocalEmbeddingService(
            model_name=model or "all-MiniLM-L6-v2",
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
