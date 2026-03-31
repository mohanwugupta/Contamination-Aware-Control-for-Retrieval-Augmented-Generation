"""Normalized input example schema (PRD 1 §10.1).

Every benchmark example is normalized into this schema before
entering the pipeline. Supports single-answer, multi-answer,
and unknown/abstain-compatible targets.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


TaskType = Literal["single_answer_qa", "multi_answer_qa"]


class GoldAnswer(BaseModel):
    """Gold answer container supporting single, multi, and unknown modes."""

    single_answer: str | None = None
    multi_answers: list[str] | None = None
    unknown_allowed: bool = False


class ExampleMetadata(BaseModel):
    """Metadata attached to every input example. Must include dataset name."""

    dataset: str
    split: str | None = None
    extra: dict[str, object] | None = None


class InputExample(BaseModel):
    """Top-level normalized input example for the RAG pipeline."""

    example_id: str
    question: str
    task_type: TaskType
    gold: GoldAnswer
    metadata: ExampleMetadata

    # --- validators ---

    @field_validator("example_id")
    @classmethod
    def example_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("example_id must not be empty")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def coerce_metadata(cls, v: object) -> object:
        """Accept a plain dict and validate it contains 'dataset'."""
        if isinstance(v, dict):
            if "dataset" not in v:
                raise ValueError("metadata must contain 'dataset' field")
            return ExampleMetadata(**v)
        return v
