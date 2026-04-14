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

    normalized_preds = [_normalize(p) for p in predictions]
    normalized_golds = [_normalize(g) for g in gold_answers]

    # Recall: how many gold answers are covered
    # A gold answer is covered if it appears as a substring in ANY of the normalized predictions
    covered = sum(
        1 for g in normalized_golds 
        if g and any(g in p for p in normalized_preds)
    )
    
    # Avoid division by zero if normalized_golds is empty or only contains empty strings
    valid_golds_count = len([g for g in normalized_golds if g])
    if valid_golds_count == 0:
        recall = 0.0
    else:
        recall = covered / valid_golds_count

    # Identify if there is a merged answer (one prediction containing multiple golds)
    # A prediction is "merged" if it covers >1 distinct gold answer
    def _count_golds_in_pred(pred: str) -> int:
        return sum(1 for g in normalized_golds if g and g in pred)
        
    is_merged = any(_count_golds_in_pred(p) > 1 for p in normalized_preds)

    # Determine category
    if is_merged and valid_golds_count > 1:
        category = "merged"
    elif recall == 1.0:
        category = "complete"
    elif recall > 0.0:
        category = "partial"
    else:
        # recall == 0.0 and predictions is non-empty.
        # The model DID produce an answer — it was just entirely wrong.
        # This is distinct from 'no_answer' (abstention/empty output), which is
        # handled by the early-return guard above.  Conflating the two makes it
        # impossible to distinguish confident hallucination from abstention in
        # downstream error analysis, so we label non-empty zero-recall outputs
        # as 'wrong'.
        category = "wrong"
        
    # Check for "ambiguous" - model provided fewer answers than required, or didn't disambiguate
    # If the score is partial or merged, it inherently shows the model was ambiguous or confused.
    # We can refine this: if it didn't find all but found some, or merged them, we map it to ambiguous/merged.
    # The AmbigDocs paper uses 'ambiguous' for returning an ambiguous single answer when multiple exist.
    if category == "partial" and len(predictions) == 1 and valid_golds_count > 1:
         category = "ambiguous"

    return Metrics(
        exact_match=None,
        normalized_match=None,
        multi_answer_score=recall,
        answer_category=category,
    )
