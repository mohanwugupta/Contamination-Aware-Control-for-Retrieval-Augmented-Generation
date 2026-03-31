"""Retrieval output schema (PRD 1 §10.2).

Captures the raw retrieval results before any reranking.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class RetrievedPassage(BaseModel):
    """A single passage returned by the retriever."""

    passage_id: str
    text: str
    source: str
    retrieval_score: float
    rank: int

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("passage text must not be empty")
        return v


class RetrievalOutput(BaseModel):
    """Full retrieval output for one example."""

    example_id: str
    retrieved_passages: list[RetrievedPassage]
