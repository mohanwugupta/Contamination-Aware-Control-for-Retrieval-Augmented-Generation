"""Hybrid retriever combining dense + sparse with reciprocal rank fusion (PRD 1 §9.2).

Combines results from DenseRetriever and SparseRetriever using
reciprocal rank fusion (RRF) for score combination.
"""

from __future__ import annotations

from rag_baseline.retrieval.base import BaseRetriever
from rag_baseline.retrieval.dense import DenseRetriever
from rag_baseline.retrieval.sparse import SparseRetriever
from rag_baseline.schemas.retrieval import RetrievalOutput, RetrievedPassage


class HybridRetriever(BaseRetriever):
    """Hybrid retriever using reciprocal rank fusion."""

    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-base-en-v1.5",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
    ) -> None:
        self.dense = DenseRetriever(model_name=dense_model_name)
        self.sparse = SparseRetriever()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self._corpus: list[dict] = []

    def index(self, corpus: list[dict]) -> None:
        """Index corpus in both dense and sparse retrievers."""
        self._corpus = corpus
        self.dense.index(corpus)
        self.sparse.index(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalOutput:
        """Retrieve using reciprocal rank fusion of dense and sparse results."""
        # Retrieve more from each to have enough for fusion
        fetch_k = min(top_k * 3, len(self._corpus))
        dense_results = self.dense.retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse.retrieve(query, top_k=fetch_k)

        # Build RRF scores
        rrf_scores: dict[str, float] = {}
        passage_lookup: dict[str, RetrievedPassage] = {}

        for p in dense_results.retrieved_passages:
            rrf_scores[p.passage_id] = rrf_scores.get(p.passage_id, 0.0)
            rrf_scores[p.passage_id] += self.dense_weight / (self.rrf_k + p.rank)
            passage_lookup[p.passage_id] = p

        for p in sparse_results.retrieved_passages:
            rrf_scores[p.passage_id] = rrf_scores.get(p.passage_id, 0.0)
            rrf_scores[p.passage_id] += self.sparse_weight / (self.rrf_k + p.rank)
            if p.passage_id not in passage_lookup:
                passage_lookup[p.passage_id] = p

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)[:top_k]

        passages = []
        for rank, pid in enumerate(sorted_ids, start=1):
            p = passage_lookup[pid]
            passages.append(
                RetrievedPassage(
                    passage_id=p.passage_id,
                    text=p.text,
                    source=p.source,
                    retrieval_score=rrf_scores[pid],
                    rank=rank,
                )
            )

        return RetrievalOutput(
            example_id="",  # Set by caller
            retrieved_passages=passages,
        )
