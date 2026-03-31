"""Multi-answer scorer for AmbigDocs-like tasks (PRD 1 §11.2, §13).

Computes recall-based multi-answer score and answer category.
"""

from __future__ import annotations

import re
import string

from rag_baseline.schemas.evaluation import Metrics


def _normalize(text: str) -> str:
    """Normalize for comparison."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text.strip()


def compute_multi_answer_score(
    predictions: list[str],
    gold_answers: list[str],
) -> Metrics:
    """Compute multi-answer recall score and category.

    Args:
        predictions: List of predicted answers.
        gold_answers: List of gold answers.

    Returns:
        Metrics with multi_answer_score and answer_category.
    """
    if not predictions or not gold_answers:
        return Metrics(
            exact_match=None,
            normalized_match=None,
            multi_answer_score=0.0,
            answer_category="no_answer",
        )

    normalized_preds = {_normalize(p) for p in predictions}
    normalized_golds = [_normalize(g) for g in gold_answers]

    # Recall: how many gold answers are covered
    covered = sum(1 for g in normalized_golds if g in normalized_preds)
    recall = covered / len(normalized_golds)

    # Determine category
    if recall == 1.0:
        category = "complete"
    elif recall > 0.0:
        category = "partial"
    else:
        category = "no_answer"

    return Metrics(
        exact_match=None,
        normalized_match=None,
        multi_answer_score=recall,
        answer_category=category,
    )
