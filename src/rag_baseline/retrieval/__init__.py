"""Retrieval module factory (PRD 1 §9.2)."""

from __future__ import annotations

from rag_baseline.retrieval.base import BaseRetriever


def create_retriever(
    retriever_type: str,
    dense_model: str = "BAAI/bge-base-en-v1.5",
    **kwargs: object,
) -> BaseRetriever | None:
    """Factory function to create a retriever by type.

    Args:
        retriever_type: One of 'dense', 'sparse', 'hybrid', 'none'.
        dense_model: Model name for dense retriever.

    Returns:
        A BaseRetriever instance, or None if type is 'none'.

    Raises:
        ValueError: If retriever_type is not recognized.
    """
    if retriever_type == "dense":
        from rag_baseline.retrieval.dense import DenseRetriever
        return DenseRetriever(model_name=dense_model)
    elif retriever_type == "sparse":
        from rag_baseline.retrieval.sparse import SparseRetriever
        return SparseRetriever()
    elif retriever_type == "hybrid":
        from rag_baseline.retrieval.hybrid import HybridRetriever
        return HybridRetriever(dense_model_name=dense_model)
    elif retriever_type == "none":
        return None
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
