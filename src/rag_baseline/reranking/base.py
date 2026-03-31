"""Abstract base reranker (PRD 1 §9.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_baseline.schemas.rerank import RerankOutput
from rag_baseline.schemas.retrieval import RetrievedPassage


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        passages: list[RetrievedPassage],
        example_id: str = "",
    ) -> RerankOutput:
        """Rerank passages for a query.

        Args:
            query: The query string.
            passages: List of retrieved passages to rerank.
            example_id: Example identifier.

        Returns:
            RerankOutput with reranked passages.
        """
        ...
