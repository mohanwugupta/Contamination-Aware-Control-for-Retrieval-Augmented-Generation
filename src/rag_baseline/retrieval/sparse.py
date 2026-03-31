"""Sparse (BM25) retriever (PRD 1 §9.2).

Uses rank_bm25 for lexical retrieval.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from rag_baseline.retrieval.base import BaseRetriever
from rag_baseline.schemas.retrieval import RetrievalOutput, RetrievedPassage


class SparseRetriever(BaseRetriever):
    """BM25-based sparse retriever."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []

    def index(self, corpus: list[dict]) -> None:
        """Index corpus using BM25."""
        self._corpus = corpus
        tokenized = [doc["text"].lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalOutput:
        """Retrieve top-k passages via BM25 scoring."""
        if self._bm25 is None:
            raise RuntimeError("Must call index() before retrieve()")

        top_k = min(top_k, len(self._corpus))
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = scores.argsort()[::-1][:top_k]

        passages = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = self._corpus[idx]
            passages.append(
                RetrievedPassage(
                    passage_id=doc["passage_id"],
                    text=doc["text"],
                    source=doc["source"],
                    retrieval_score=float(scores[idx]),
                    rank=rank,
                )
            )

        return RetrievalOutput(
            example_id="",  # Set by caller
            retrieved_passages=passages,
        )
