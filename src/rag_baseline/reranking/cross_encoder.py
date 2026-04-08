"""Cross-encoder reranker (PRD 1 §9.3).

Uses a cross-encoder model to score query-passage pairs.
"""

from __future__ import annotations

import os

from sentence_transformers import CrossEncoder

from rag_baseline.reranking.base import BaseReranker
from rag_baseline.schemas.rerank import RerankOutput, RerankedPassage
from rag_baseline.schemas.retrieval import RetrievedPassage


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            # Pass cache_folder so sentence-transformers looks in HF_HOME
            # (set by SLURM scripts) rather than its own default
            # ~/.cache/torch/sentence_transformers/.  Without this, models
            # downloaded by precache_models.sh are invisible at inference time.
            cache_folder = os.environ.get("HF_HOME")
            self._model = CrossEncoder(
                self.model_name,
                cache_folder=cache_folder,
            )
        return self._model

    def rerank(
        self,
        query: str,
        passages: list[RetrievedPassage],
        example_id: str = "",
    ) -> RerankOutput:
        """Rerank passages using cross-encoder scoring."""
        if not passages:
            return RerankOutput(example_id=example_id, reranked_passages=[])

        pairs = [(query, p.text) for p in passages]
        scores = self.model.predict(pairs)

        # Pair passages with scores and sort descending
        scored = list(zip(passages, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = []
        for rank, (p, score) in enumerate(scored, start=1):
            reranked.append(
                RerankedPassage(
                    passage_id=p.passage_id,
                    text=p.text,
                    source=p.source,
                    retrieval_score=p.retrieval_score,
                    rerank_score=float(score),
                    rank_after_rerank=rank,
                )
            )

        return RerankOutput(example_id=example_id, reranked_passages=reranked)
