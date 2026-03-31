"""RED tests for all PRD 1 normalized schemas.

These tests define the contracts from PRD 1 §10.
They should FAIL until the schemas are implemented (red → green).
"""

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# §10.1 — Input schema
# ---------------------------------------------------------------------------

class TestInputExample:
    """Tests for the normalized input example schema (PRD 1 §10.1)."""

    def test_single_answer_example_creates(self):
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        gold = GoldAnswer(
            single_answer="1963",
            multi_answers=None,
            unknown_allowed=False,
        )
        ex = InputExample(
            example_id="ex_0001",
            question="In which year was Michael Jordan born?",
            task_type="single_answer_qa",
            gold=gold,
            metadata={"dataset": "nq_open", "split": "dev"},
        )
        assert ex.example_id == "ex_0001"
        assert ex.gold.single_answer == "1963"

    def test_multi_answer_example_creates(self):
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        gold = GoldAnswer(
            single_answer=None,
            multi_answers=["1963", "1956"],
            unknown_allowed=False,
        )
        ex = InputExample(
            example_id="ex_0002",
            question="In which year was Michael Jordan born?",
            task_type="multi_answer_qa",
            gold=gold,
            metadata={"dataset": "ambigdocs", "split": "dev"},
        )
        assert ex.gold.multi_answers == ["1963", "1956"]
        assert ex.gold.single_answer is None

    def test_unknown_allowed_example(self):
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        gold = GoldAnswer(
            single_answer=None,
            multi_answers=None,
            unknown_allowed=True,
        )
        ex = InputExample(
            example_id="ex_0003",
            question="What is the capital of Atlantis?",
            task_type="single_answer_qa",
            gold=gold,
            metadata={"dataset": "faitheval", "split": "dev"},
        )
        assert ex.gold.unknown_allowed is True

    def test_task_type_must_be_valid(self):
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        gold = GoldAnswer(single_answer="x", multi_answers=None, unknown_allowed=False)
        with pytest.raises(ValidationError):
            InputExample(
                example_id="ex_bad",
                question="Q?",
                task_type="invalid_type",
                gold=gold,
                metadata={},
            )

    def test_example_id_required(self):
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        gold = GoldAnswer(single_answer="x", multi_answers=None, unknown_allowed=False)
        with pytest.raises(ValidationError):
            InputExample(
                example_id="",  # empty not allowed
                question="Q?",
                task_type="single_answer_qa",
                gold=gold,
                metadata={},
            )

    def test_metadata_requires_dataset_field(self):
        """Metadata must contain at least 'dataset'."""
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        gold = GoldAnswer(single_answer="x", multi_answers=None, unknown_allowed=False)
        with pytest.raises(ValidationError):
            InputExample(
                example_id="ex_0004",
                question="Q?",
                task_type="single_answer_qa",
                gold=gold,
                metadata={"split": "dev"},  # missing 'dataset'
            )

    def test_round_trip_json(self):
        """Schema must serialize to JSON and back without loss."""
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        gold = GoldAnswer(single_answer="1963", multi_answers=None, unknown_allowed=False)
        ex = InputExample(
            example_id="ex_rt",
            question="Q?",
            task_type="single_answer_qa",
            gold=gold,
            metadata={"dataset": "nq_open", "split": "dev"},
        )
        json_str = ex.model_dump_json()
        ex2 = InputExample.model_validate_json(json_str)
        assert ex == ex2


# ---------------------------------------------------------------------------
# §10.2 — Retrieval output schema
# ---------------------------------------------------------------------------

class TestRetrievalOutput:
    """Tests for retrieval output schema (PRD 1 §10.2)."""

    def test_retrieval_output_creates(self):
        from rag_baseline.schemas.retrieval import RetrievalOutput, RetrievedPassage

        passage = RetrievedPassage(
            passage_id="p1",
            text="Michael Jordan was born in 1963.",
            source="doc_123",
            retrieval_score=13.2,
            rank=1,
        )
        output = RetrievalOutput(
            example_id="ex_0001",
            retrieved_passages=[passage],
        )
        assert len(output.retrieved_passages) == 1
        assert output.retrieved_passages[0].rank == 1

    def test_multiple_passages_preserve_order(self):
        from rag_baseline.schemas.retrieval import RetrievalOutput, RetrievedPassage

        passages = [
            RetrievedPassage(
                passage_id=f"p{i}",
                text=f"Passage {i}",
                source=f"doc_{i}",
                retrieval_score=10.0 - i,
                rank=i,
            )
            for i in range(1, 6)
        ]
        output = RetrievalOutput(example_id="ex_multi", retrieved_passages=passages)
        assert [p.rank for p in output.retrieved_passages] == [1, 2, 3, 4, 5]

    def test_passage_requires_text(self):
        from rag_baseline.schemas.retrieval import RetrievedPassage

        with pytest.raises(ValidationError):
            RetrievedPassage(
                passage_id="p1",
                text="",  # empty text not allowed
                source="doc_1",
                retrieval_score=1.0,
                rank=1,
            )

    def test_round_trip_json(self):
        from rag_baseline.schemas.retrieval import RetrievalOutput, RetrievedPassage

        passage = RetrievedPassage(
            passage_id="p1", text="text", source="doc_1",
            retrieval_score=5.0, rank=1,
        )
        output = RetrievalOutput(example_id="ex_rt", retrieved_passages=[passage])
        json_str = output.model_dump_json()
        output2 = RetrievalOutput.model_validate_json(json_str)
        assert output == output2


# ---------------------------------------------------------------------------
# §10.3 — Reranker output schema
# ---------------------------------------------------------------------------

class TestRerankOutput:
    """Tests for reranker output schema (PRD 1 §10.3)."""

    def test_rerank_output_creates(self):
        from rag_baseline.schemas.rerank import RerankOutput, RerankedPassage

        passage = RerankedPassage(
            passage_id="p3",
            text="Some passage",
            source="doc_999",
            retrieval_score=11.8,
            rerank_score=0.87,
            rank_after_rerank=1,
        )
        output = RerankOutput(
            example_id="ex_0001",
            reranked_passages=[passage],
        )
        assert output.reranked_passages[0].rerank_score == 0.87

    def test_reranked_passage_has_both_scores(self):
        from rag_baseline.schemas.rerank import RerankedPassage

        p = RerankedPassage(
            passage_id="p1", text="t", source="s",
            retrieval_score=10.0, rerank_score=0.9, rank_after_rerank=1,
        )
        assert p.retrieval_score == 10.0
        assert p.rerank_score == 0.9

    def test_round_trip_json(self):
        from rag_baseline.schemas.rerank import RerankOutput, RerankedPassage

        p = RerankedPassage(
            passage_id="p1", text="t", source="s",
            retrieval_score=10.0, rerank_score=0.9, rank_after_rerank=1,
        )
        output = RerankOutput(example_id="ex_rt", reranked_passages=[p])
        json_str = output.model_dump_json()
        output2 = RerankOutput.model_validate_json(json_str)
        assert output == output2


# ---------------------------------------------------------------------------
# §10.4 — Prompt record schema
# ---------------------------------------------------------------------------

class TestPromptRecord:
    """Tests for prompt record schema (PRD 1 §10.4)."""

    def test_prompt_record_creates(self):
        from rag_baseline.schemas.prompt import PromptRecord, PromptMetadata

        meta = PromptMetadata(
            model_name="Qwen/Qwen2.5-32B-Instruct",
            temperature=0.0,
            max_context_passages=4,
        )
        record = PromptRecord(
            example_id="ex_0001",
            baseline_name="hybrid_rerank_full",
            answer_mode="multi",
            used_passage_ids=["p3", "p1", "p4", "p2"],
            prompt_text="Given the following documents...",
            prompt_metadata=meta,
        )
        assert record.answer_mode == "multi"
        assert len(record.used_passage_ids) == 4

    def test_answer_mode_must_be_valid(self):
        from rag_baseline.schemas.prompt import PromptRecord, PromptMetadata

        meta = PromptMetadata(
            model_name="m", temperature=0.0, max_context_passages=4,
        )
        with pytest.raises(ValidationError):
            PromptRecord(
                example_id="ex",
                baseline_name="b",
                answer_mode="invalid_mode",
                used_passage_ids=[],
                prompt_text="p",
                prompt_metadata=meta,
            )

    def test_round_trip_json(self):
        from rag_baseline.schemas.prompt import PromptRecord, PromptMetadata

        meta = PromptMetadata(model_name="m", temperature=0.0, max_context_passages=4)
        record = PromptRecord(
            example_id="ex_rt", baseline_name="b", answer_mode="single",
            used_passage_ids=["p1"], prompt_text="text", prompt_metadata=meta,
        )
        json_str = record.model_dump_json()
        record2 = PromptRecord.model_validate_json(json_str)
        assert record == record2


# ---------------------------------------------------------------------------
# §10.5 — Generation output schema
# ---------------------------------------------------------------------------

class TestGenerationOutput:
    """Tests for generation output schema (PRD 1 §10.5)."""

    def test_generation_output_single(self):
        from rag_baseline.schemas.generation import GenerationOutput, ParsedOutput

        parsed = ParsedOutput(
            single_answer="1963",
            multi_answers=None,
            unknown=False,
        )
        gen = GenerationOutput(
            example_id="ex_0001",
            raw_model_output="The answer is 1963.",
            parsed_output=parsed,
        )
        assert gen.parsed_output.single_answer == "1963"
        assert gen.parsed_output.unknown is False

    def test_generation_output_multi(self):
        from rag_baseline.schemas.generation import GenerationOutput, ParsedOutput

        parsed = ParsedOutput(
            single_answer=None,
            multi_answers=["1963", "1956"],
            unknown=False,
        )
        gen = GenerationOutput(
            example_id="ex_0001",
            raw_model_output="Multiple answers: 1963 and 1956.",
            parsed_output=parsed,
        )
        assert gen.parsed_output.multi_answers == ["1963", "1956"]

    def test_generation_output_unknown(self):
        from rag_baseline.schemas.generation import GenerationOutput, ParsedOutput

        parsed = ParsedOutput(
            single_answer=None,
            multi_answers=None,
            unknown=True,
        )
        gen = GenerationOutput(
            example_id="ex_0001",
            raw_model_output="I don't know.",
            parsed_output=parsed,
        )
        assert gen.parsed_output.unknown is True

    def test_generation_output_preserves_raw(self):
        from rag_baseline.schemas.generation import GenerationOutput, ParsedOutput

        raw = "The answer is 1963.\nAdditional commentary here."
        parsed = ParsedOutput(single_answer="1963", multi_answers=None, unknown=False)
        gen = GenerationOutput(
            example_id="ex_0001",
            raw_model_output=raw,
            parsed_output=parsed,
        )
        assert gen.raw_model_output == raw

    def test_round_trip_json(self):
        from rag_baseline.schemas.generation import GenerationOutput, ParsedOutput

        parsed = ParsedOutput(single_answer="x", multi_answers=None, unknown=False)
        gen = GenerationOutput(
            example_id="ex_rt", raw_model_output="raw", parsed_output=parsed,
        )
        json_str = gen.model_dump_json()
        gen2 = GenerationOutput.model_validate_json(json_str)
        assert gen == gen2


# ---------------------------------------------------------------------------
# §10.6 — Evaluation output schema
# ---------------------------------------------------------------------------

class TestEvaluationOutput:
    """Tests for evaluation output schema (PRD 1 §10.6)."""

    def test_evaluation_output_creates(self):
        from rag_baseline.schemas.evaluation import EvaluationOutput, Metrics

        metrics = Metrics(
            exact_match=None,
            normalized_match=None,
            multi_answer_score=1.0,
            answer_category="complete",
        )
        ev = EvaluationOutput(
            example_id="ex_0001",
            dataset="ambigdocs",
            baseline_name="hybrid_rerank_full",
            metrics=metrics,
        )
        assert ev.metrics.multi_answer_score == 1.0
        assert ev.metrics.answer_category == "complete"

    def test_evaluation_output_exact_match(self):
        from rag_baseline.schemas.evaluation import EvaluationOutput, Metrics

        metrics = Metrics(
            exact_match=True,
            normalized_match=True,
            multi_answer_score=None,
            answer_category=None,
        )
        ev = EvaluationOutput(
            example_id="ex_0002",
            dataset="nq_open",
            baseline_name="vanilla_rag",
            metrics=metrics,
        )
        assert ev.metrics.exact_match is True

    def test_evaluation_output_requires_dataset(self):
        from rag_baseline.schemas.evaluation import EvaluationOutput, Metrics

        metrics = Metrics(
            exact_match=True, normalized_match=True,
            multi_answer_score=None, answer_category=None,
        )
        with pytest.raises(ValidationError):
            EvaluationOutput(
                example_id="ex_bad",
                dataset="",  # empty not allowed
                baseline_name="b",
                metrics=metrics,
            )

    def test_round_trip_json(self):
        from rag_baseline.schemas.evaluation import EvaluationOutput, Metrics

        metrics = Metrics(
            exact_match=True, normalized_match=True,
            multi_answer_score=None, answer_category=None,
        )
        ev = EvaluationOutput(
            example_id="ex_rt", dataset="nq_open",
            baseline_name="b", metrics=metrics,
        )
        json_str = ev.model_dump_json()
        ev2 = EvaluationOutput.model_validate_json(json_str)
        assert ev == ev2
