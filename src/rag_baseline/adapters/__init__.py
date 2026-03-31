"""Adapter factory (PRD 1 §18 Task Group A).

Creates the correct dataset adapter by dataset name.
"""

from __future__ import annotations

from rag_baseline.adapters.base import BaseAdapter


def create_adapter(dataset: str) -> BaseAdapter:
    """Create a dataset adapter by name.

    Args:
        dataset: Dataset name (e.g., 'nq_open', 'ambigdocs').

    Returns:
        A BaseAdapter instance for the requested dataset.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if dataset in ("nq_open", "nq", "natural_questions"):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        return NQOpenAdapter()
    elif dataset in ("ambigdocs", "ambig_docs"):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        return AmbigDocsAdapter()
    elif dataset in ("faitheval", "faith_eval"):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        return FaithEvalAdapter()
    elif dataset in ("ramdocs", "ram_docs"):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        return RAMDocsAdapter()
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Supported: nq_open, ambigdocs, faitheval, ramdocs"
        )
