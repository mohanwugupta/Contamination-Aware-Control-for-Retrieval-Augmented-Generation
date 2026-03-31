"""Reranking module factory (PRD 1 §9.3)."""

from __future__ import annotations

from rag_baseline.reranking.base import BaseReranker


def create_reranker(reranker_model: str | None = None) -> BaseReranker | None:
    """Factory function to create a reranker.

    Args:
        reranker_model: Model name, or None to disable reranking.

    Returns:
        A BaseReranker instance, or None if disabled.
    """
    if reranker_model is None:
        return None

    from rag_baseline.reranking.cross_encoder import CrossEncoderReranker
    return CrossEncoderReranker(model_name=reranker_model)
