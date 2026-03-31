"""Exact match scorer for NQ-like tasks (PRD 1 §11.1, §13).

Supports:
- Exact match
- Normalized match (case-insensitive, article removal, whitespace normalization)
"""

from __future__ import annotations

import re
import string

from rag_baseline.schemas.evaluation import Metrics


def _normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Lowercase, remove articles, remove punctuation, collapse whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def compute_exact_match(prediction: str, gold: list[str]) -> Metrics:
    """Compute exact match and normalized match metrics.

    Args:
        prediction: Predicted answer string.
        gold: List of acceptable gold answers.

    Returns:
        Metrics with exact_match and normalized_match fields.
    """
    prediction = prediction.strip()
    exact = any(prediction == g for g in gold)
    normalized = any(
        _normalize_answer(prediction) == _normalize_answer(g)
        for g in gold
    )
    return Metrics(
        exact_match=exact,
        normalized_match=normalized,
        multi_answer_score=None,
        answer_category=None,
    )
