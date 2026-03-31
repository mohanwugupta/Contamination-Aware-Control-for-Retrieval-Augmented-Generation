"""Reranker output schema (PRD 1 §10.3).

Captures passages after reranking, preserving both the original
retrieval score and the rerank score.
"""

from __future__ import annotations

from pydantic import BaseModel


class RerankedPassage(BaseModel):
    """A single passage after reranking."""

    passage_id: str
    text: str
    source: str
    retrieval_score: float
    rerank_score: float
    rank_after_rerank: int


class RerankOutput(BaseModel):
    """Full reranker output for one example."""

    example_id: str
    reranked_passages: list[RerankedPassage]
