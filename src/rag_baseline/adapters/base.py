"""Abstract base adapter (PRD 1 §9.1).

All dataset adapters must subclass this and implement load() and get_corpus().
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from rag_baseline.schemas.input import InputExample


def _load_hf_split(hf_id: str, split: str, disk_name: str) -> Any:
    """Load a dataset split, preferring a pre-cached Arrow disk copy.

    On cluster compute nodes (``HF_HUB_OFFLINE=1``) HuggingFace cannot
    reach the hub, but ``load_from_disk`` reads Arrow data written by
    ``save_to_disk`` during the pre-cache step (``slurm/precache_datasets.sh``).

    Resolution order:

    1. ``$HF_DATASETS_DISK_DIR/<disk_name>`` — explicit disk-cache env var.
    2. ``$HF_DATASETS_CACHE/<disk_name>`` — standard HF cache dir (same
       naming convention, different var name).
    3. ``datasets.load_dataset(hf_id, split=split)`` — network load for
       local development where the pre-cache step has not been run.

    Args:
        hf_id: HuggingFace dataset identifier
               (e.g. ``"google-research-datasets/nq_open"``).
        split: Dataset split (``"train"``, ``"validation"``, or ``"test"``).
        disk_name: Directory name written by ``precache_datasets.sh``
                   (e.g. ``"nq_open_validation"``).

    Returns:
        A HuggingFace :class:`datasets.Dataset` object.
    """
    from datasets import load_dataset, load_from_disk

    for env_var in ("HF_DATASETS_DISK_DIR", "HF_DATASETS_CACHE"):
        base = os.environ.get(env_var)
        if base:
            path = os.path.join(base, disk_name)
            if os.path.isdir(path):
                return load_from_disk(path)

    # Fall back to network load (local dev / CI)
    return load_dataset(hf_id, split=split)


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
