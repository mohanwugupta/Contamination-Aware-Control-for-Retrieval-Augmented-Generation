"""Unified evaluation interface (PRD 1 §9.7).

Dispatches to dataset-specific scorers while presenting a uniform
outer interface. All scorers return EvaluationOutput.
"""

from __future__ import annotations

from rag_baseline.schemas.evaluation import EvaluationOutput
from rag_baseline.schemas.generation import ParsedOutput
from rag_baseline.schemas.input import GoldAnswer


def evaluate_example(
    dataset: str,
    parsed_output: ParsedOutput,
    gold: GoldAnswer,
    example_id: str,
    baseline_name: str,
) -> EvaluationOutput:
    """Evaluate a single example using the appropriate scorer.

    Args:
        dataset: Dataset name (determines which scorer to use).
        parsed_output: Parsed model output.
        gold: Gold answer.
        example_id: Example identifier.
        baseline_name: Baseline name.

    Returns:
        EvaluationOutput with dataset-appropriate metrics.
    """
    if dataset in ("nq_open", "nq", "natural_questions"):
        return _evaluate_exact_match(parsed_output, gold, example_id, dataset, baseline_name)
    elif dataset in ("ambigdocs", "ambig_docs"):
        return _evaluate_multi_answer(parsed_output, gold, example_id, dataset, baseline_name)
    elif dataset in ("faitheval", "faith_eval"):
        return _evaluate_exact_match(parsed_output, gold, example_id, dataset, baseline_name)
    elif dataset in ("ramdocs", "ram_docs"):
        return _evaluate_multi_answer(parsed_output, gold, example_id, dataset, baseline_name)
    else:
        # Default to exact match
        return _evaluate_exact_match(parsed_output, gold, example_id, dataset, baseline_name)


def _evaluate_exact_match(
    parsed: ParsedOutput,
    gold: GoldAnswer,
    example_id: str,
    dataset: str,
    baseline_name: str,
) -> EvaluationOutput:
    """Evaluate using exact match scorer."""
    from rag_baseline.evaluation.exact_match import compute_exact_match

    # Build gold list
    gold_list: list[str] = []
    if gold.single_answer:
        gold_list.append(gold.single_answer)
    if gold.multi_answers:
        gold_list.extend(gold.multi_answers)

    prediction = parsed.single_answer or ""
    metrics = compute_exact_match(prediction, gold_list)

    return EvaluationOutput(
        example_id=example_id,
        dataset=dataset,
        baseline_name=baseline_name,
        metrics=metrics,
    )


def _evaluate_multi_answer(
    parsed: ParsedOutput,
    gold: GoldAnswer,
    example_id: str,
    dataset: str,
    baseline_name: str,
) -> EvaluationOutput:
    """Evaluate using multi-answer scorer."""
    from rag_baseline.evaluation.multi_answer import compute_multi_answer_score

    predictions = parsed.multi_answers or []
    gold_answers = gold.multi_answers or []

    metrics = compute_multi_answer_score(predictions, gold_answers)

    return EvaluationOutput(
        example_id=example_id,
        dataset=dataset,
        baseline_name=baseline_name,
        metrics=metrics,
    )
