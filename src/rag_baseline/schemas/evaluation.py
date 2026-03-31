"""Evaluation output schema (PRD 1 §10.6).

Captures evaluation metrics for one example, keyed by dataset
and baseline name.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class Metrics(BaseModel):
    """Evaluation metrics container.

    Different fields are populated depending on the task type.
    """

    exact_match: bool | None = None
    normalized_match: bool | None = None
    multi_answer_score: float | None = None
    answer_category: str | None = None


class EvaluationOutput(BaseModel):
    """Full evaluation output for one example."""

    example_id: str
    dataset: str
    baseline_name: str
    metrics: Metrics

    @field_validator("dataset")
    @classmethod
    def dataset_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset must not be empty")
        return v
