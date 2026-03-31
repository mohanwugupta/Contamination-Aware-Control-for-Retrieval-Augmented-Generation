"""Abstract base retriever (PRD 1 §9.2).

All retriever implementations must subclass this and implement
index() and retrieve().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_baseline.schemas.retrieval import RetrievalOutput


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def index(self, corpus: list[dict]) -> None:
        """Index a corpus of passages.

        Args:
            corpus: List of dicts with keys 'passage_id', 'text', 'source'.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalOutput:
        """Retrieve top-k passages for a query.

        Args:
            query: The query string.
            top_k: Number of passages to retrieve.

        Returns:
            RetrievalOutput with ranked passages.
        """
        ...
