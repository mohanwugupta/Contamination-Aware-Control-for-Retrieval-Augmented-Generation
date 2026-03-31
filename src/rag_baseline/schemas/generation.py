"""Generation output schema (PRD 1 §10.5).

Captures both the raw model output and the parsed structured output.
"""

from __future__ import annotations

from pydantic import BaseModel


class ParsedOutput(BaseModel):
    """Structured parsed answer from the model output."""

    single_answer: str | None = None
    multi_answers: list[str] | None = None
    unknown: bool = False


class GenerationOutput(BaseModel):
    """Full generation output for one example."""

    example_id: str
    raw_model_output: str
    parsed_output: ParsedOutput
