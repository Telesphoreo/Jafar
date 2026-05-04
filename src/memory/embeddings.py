"""
Embedding Service using Google Gemini.

Uses Gemini's embedding models with task-specific optimization:
- SEMANTIC_SIMILARITY: Historical parallels
- FACT_VERIFICATION: Claim checking
- CLASSIFICATION: Bot detection, account scoring
- CLUSTERING: Trend grouping
"""

import logging
from abc import ABC, abstractmethod

from google import genai

logger = logging.getLogger("jafar.memory.embeddings")


class EmbeddingService(ABC):
    """Abstract interface for embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced."""

    @abstractmethod
    async def embed(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> list[float]:
        """Generate embedding for a single text."""


class GeminiEmbeddingService(EmbeddingService):
    """
    Gemini embedding service using gemini-embedding-2-preview.

    Supports task-specific embeddings for optimal retrieval quality:
    - SEMANTIC_SIMILARITY: Best for historical parallel search
    - FACT_VERIFICATION: Best for claim verification
    - CLASSIFICATION: Best for bot detection and account scoring
    - CLUSTERING: Best for trend grouping
    - RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY: Best for document search
    """

    VALID_TASK_TYPES = {
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "CODE_RETRIEVAL_QUERY",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2-preview",
        dimensions: int = 3072,
    ):
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._dimension = dimensions
        logger.info(
            f"GeminiEmbeddingService initialized: model={model}, dimensions={dimensions}"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> list[float]:
        """Generate embedding for a single text with task-specific optimization."""
        if task_type not in self.VALID_TASK_TYPES:
            logger.warning(f"Unknown task type '{task_type}', falling back to SEMANTIC_SIMILARITY")
            task_type = "SEMANTIC_SIMILARITY"

        result = self._client.models.embed_content(
            model=self.model,
            contents=text,
            config={
                "task_type": task_type,
                "output_dimensionality": self._dimension,
            },
        )
        return result.embeddings[0].values
