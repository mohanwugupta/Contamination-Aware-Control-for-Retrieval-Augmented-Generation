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
            # HuggingFace stores model files in {HF_HOME}/hub/, not in
            # HF_HOME itself.  CrossEncoder's cache_folder maps directly to
            # the cache_dir arg of AutoConfig.from_pretrained, so we must
            # point at the hub sub-directory.
            hf_home = os.environ.get("HF_HOME")
            cache_folder = os.path.join(hf_home, "hub") if hf_home else None
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
