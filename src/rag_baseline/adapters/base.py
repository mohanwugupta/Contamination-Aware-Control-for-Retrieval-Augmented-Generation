"""Abstract base adapter (PRD 1 §9.1).

All dataset adapters must subclass this and implement load() and get_corpus().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_baseline.schemas.input import InputExample


class BaseAdapter(ABC):
    """Abstract base class for all dataset adapters."""

    @abstractmethod
    def load(self, split: str = "validation", **kwargs: object) -> list[InputExample]:
        """Load and normalize a dataset split into InputExamples.

        Args:
            split: Dataset split to load (train, validation, test).

        Returns:
            List of normalized InputExample instances.
        """
        ...

    @abstractmethod
    def load_from_dicts(
        self, rows: list[dict], split: str = "validation"
    ) -> list[InputExample]:
        """Load from raw dicts (for testing without HuggingFace download).

        Args:
            rows: List of raw example dicts matching the HF dataset schema.
            split: Split name for metadata.

        Returns:
            List of normalized InputExample instances.
        """
        ...

    @abstractmethod
    def get_corpus(self) -> list[dict] | None:
        """Return the corpus for retrieval, if the dataset bundles one.

        Returns:
            List of passage dicts with keys (passage_id, text, source),
            or None if the dataset is open-domain (no bundled corpus).
        """
        ...
