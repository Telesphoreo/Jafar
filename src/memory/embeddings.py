"""
Embedding Service for generating vector representations.

Uses OpenAI's embedding models by default, with support for
local models via sentence-transformers as a fallback.
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal

logger = logging.getLogger("jafar.memory.embeddings")


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
    OpenAI embedding service using text-embedding-3 models.

    Supports native dimension reduction via the dimensions parameter.
    This allows using text-embedding-3-large with reduced dimensions
    for better quality while fitting within database limits (e.g., pgvector's 2000 dim limit).

    Models:
    - text-embedding-3-small: default 1536 dimensions
    - text-embedding-3-large: default 3072 dimensions (can be reduced)
    """

    # Default dimensions for each model
    MODEL_DEFAULT_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        """
        Initialize OpenAI embedding service.

        Args:
            api_key: OpenAI API key
            model: Model name (text-embedding-3-small or text-embedding-3-large)
            dimensions: Override output dimensions (useful for pgvector's 2000 dim limit).
                        If None, uses model's default dimensions.
        """
        self.api_key = api_key
        self.model = model
        self._client = None

        # Determine dimensions
        default_dim = self.MODEL_DEFAULT_DIMENSIONS.get(model, 1536)
        if dimensions is not None:
            if dimensions > default_dim:
                logger.warning(
                    f"Requested dimensions ({dimensions}) exceeds model default ({default_dim}). "
                    f"Using {default_dim}."
                )
                self._dimension = default_dim
                self._requested_dimensions = None
            else:
                self._dimension = dimensions
                self._requested_dimensions = dimensions
        else:
            self._dimension = default_dim
            self._requested_dimensions = None

        logger.info(
            f"OpenAIEmbeddingService initialized: model={model}, dimensions={self._dimension}"
        )

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

        kwargs = {
            "model": self.model,
            "input": text,
        }
        if self._requested_dimensions is not None:
            kwargs["dimensions"] = self._requested_dimensions

        response = await client.embeddings.create(**kwargs)

        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []

        client = self._get_client()

        kwargs = {
            "model": self.model,
            "input": texts,
        }
        if self._requested_dimensions is not None:
            kwargs["dimensions"] = self._requested_dimensions

        response = await client.embeddings.create(**kwargs)

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
    dimensions: int | None = None,
) -> EmbeddingService:
    """
    Factory function to create the appropriate embedding service.

    Args:
        provider: "openai" or "local"
        api_key: OpenAI API key (required for openai provider)
        model: Model name (optional, uses defaults)
        dimensions: Override output dimensions for OpenAI embeddings.
                    Useful for pgvector's 2000 dimension limit.
                    Use 1536 or 2000 with text-embedding-3-large for best quality
                    within database constraints.

    Returns:
        Configured EmbeddingService instance
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required for openai embedding provider")
        return OpenAIEmbeddingService(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            dimensions=dimensions,
        )
    elif provider == "local":
        return LocalEmbeddingService(
            model_name=model or "all-MiniLM-L6-v2",
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
