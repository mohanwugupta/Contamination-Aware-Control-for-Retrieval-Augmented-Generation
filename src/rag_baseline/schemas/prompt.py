"""Prompt record schema (PRD 1 §10.4).

Captures the exact prompt sent to the generator, including
which passages were used and in what order.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


AnswerMode = Literal["single", "multi", "unknown_or_abstain"]


class PromptMetadata(BaseModel):
    """Metadata about the generation configuration used for this prompt."""

    model_name: str
    temperature: float
    max_context_passages: int


class PromptRecord(BaseModel):
    """Full prompt record for one example."""

    example_id: str
    baseline_name: str
    answer_mode: AnswerMode
    used_passage_ids: list[str]
    prompt_text: str
    prompt_metadata: PromptMetadata
