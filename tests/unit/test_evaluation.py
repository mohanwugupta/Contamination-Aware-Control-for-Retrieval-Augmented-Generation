"""RED tests for the evaluation layer (PRD 1 §9.7, §13).

Evaluation must dispatch to dataset-specific scorers while
presenting a uniform outer interface. Must support:
- NQ-like: normalized EM
- AmbigDocs-like: multi-answer scoring
- FaithEval-like: context-faithfulness categories
"""

import pytest


# ---------------------------------------------------------------------------
# Exact match scorer (NQ-like)
# ---------------------------------------------------------------------------

class TestExactMatchScorer:
    """Tests for normalized exact match scoring."""

    def test_exact_match_correct(self):
        from rag_baseline.evaluation.exact_match import compute_exact_match

        result = compute_exact_match(prediction="Paris", gold=["Paris"])
        assert result.exact_match is True

    def test_exact_match_case_insensitive(self):
        from rag_baseline.evaluation.exact_match import compute_exact_match

        result = compute_exact_match(prediction="paris", gold=["Paris"])
        assert result.normalized_match is True

    def test_exact_match_with_articles(self):
        from rag_baseline.evaluation.exact_match import compute_exact_match

        result = compute_exact_match(prediction="the Eiffel Tower", gold=["Eiffel Tower"])
        assert result.normalized_match is True

    def test_exact_match_wrong(self):
        from rag_baseline.evaluation.exact_match import compute_exact_match

        result = compute_exact_match(prediction="London", gold=["Paris"])
        assert result.exact_match is False
        assert result.normalized_match is False

    def test_exact_match_multiple_golds(self):
        """If any gold matches, it's correct."""
        from rag_baseline.evaluation.exact_match import compute_exact_match

        result = compute_exact_match(prediction="NYC", gold=["New York City", "NYC", "New York"])
        assert result.normalized_match is True


# ---------------------------------------------------------------------------
# Multi-answer scorer (AmbigDocs-like)
# ---------------------------------------------------------------------------

class TestMultiAnswerScorer:
    """Tests for multi-answer scoring (PRD 1 §11.2)."""

    def test_complete_match(self):
        from rag_baseline.evaluation.multi_answer import compute_multi_answer_score

        result = compute_multi_answer_score(
            predictions=["1963", "1956"],
            gold_answers=["1963", "1956"],
        )
        assert result.multi_answer_score == 1.0
        assert result.answer_category == "complete"

    def test_partial_match(self):
        from rag_baseline.evaluation.multi_answer import compute_multi_answer_score

        result = compute_multi_answer_score(
            predictions=["1963"],
            gold_answers=["1963", "1956"],
        )
        assert 0.0 < result.multi_answer_score < 1.0
        assert result.answer_category == "partial"

    def test_no_match(self):
        from rag_baseline.evaluation.multi_answer import compute_multi_answer_score

        result = compute_multi_answer_score(
            predictions=["2000"],
            gold_answers=["1963", "1956"],
        )
        assert result.multi_answer_score == 0.0
        assert result.answer_category == "no_answer"

    def test_empty_prediction(self):
        from rag_baseline.evaluation.multi_answer import compute_multi_answer_score

        result = compute_multi_answer_score(
            predictions=[],
            gold_answers=["1963"],
        )
        assert result.multi_answer_score == 0.0
        assert result.answer_category == "no_answer"

    def test_over_prediction_not_penalized_for_recall(self):
        """Extra predictions beyond gold don't reduce recall score."""
        from rag_baseline.evaluation.multi_answer import compute_multi_answer_score

        result = compute_multi_answer_score(
            predictions=["1963", "1956", "2000"],
            gold_answers=["1963", "1956"],
        )
        assert result.multi_answer_score == 1.0
        assert result.answer_category == "complete"


# ---------------------------------------------------------------------------
# Scorer dispatch
# ---------------------------------------------------------------------------

class TestScorerDispatch:
    """Tests for the unified evaluation interface."""

    def test_dispatch_nq(self):
        from rag_baseline.evaluation.base import evaluate_example
        from rag_baseline.schemas.generation import ParsedOutput
        from rag_baseline.schemas.input import GoldAnswer

        parsed = ParsedOutput(single_answer="Paris", multi_answers=None, unknown=False)
        gold = GoldAnswer(single_answer="Paris", multi_answers=None, unknown_allowed=False)

        result = evaluate_example(
            dataset="nq_open",
            parsed_output=parsed,
            gold=gold,
            example_id="ex_1",
            baseline_name="test",
        )
        assert result.metrics.normalized_match is True

    def test_dispatch_ambigdocs(self):
        from rag_baseline.evaluation.base import evaluate_example
        from rag_baseline.schemas.generation import ParsedOutput
        from rag_baseline.schemas.input import GoldAnswer

        parsed = ParsedOutput(single_answer=None, multi_answers=["1963", "1956"], unknown=False)
        gold = GoldAnswer(single_answer=None, multi_answers=["1963", "1956"], unknown_allowed=False)

        result = evaluate_example(
            dataset="ambigdocs",
            parsed_output=parsed,
            gold=gold,
            example_id="ex_1",
            baseline_name="test",
        )
        assert result.metrics.multi_answer_score == 1.0
