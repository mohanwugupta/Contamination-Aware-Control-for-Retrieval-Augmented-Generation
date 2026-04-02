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

    Notes:
        - exact_match: literal string equality (after strip).
        - normalized_match: normalized gold string appears as a substring of
          the normalized prediction (handles verbose model answers that contain
          the correct answer embedded in a full sentence).
    """
    prediction = prediction.strip()
    norm_pred = _normalize_answer(prediction)

    exact = any(prediction == g for g in gold)
    normalized = any(
        _normalize_answer(g) in norm_pred
        for g in gold
        if g  # skip empty gold strings
    )
    return Metrics(
        exact_match=exact,
        normalized_match=normalized,
        multi_answer_score=None,
        answer_category=None,
    )
