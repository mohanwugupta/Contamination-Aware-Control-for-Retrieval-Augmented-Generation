"""RED tests for structured artifact logging (PRD 1 §14).

Logger must save all intermediate artifacts in JSONL format.
"""

import json
import pytest


class TestArtifactLogger:
    """Tests for the structured artifact logger."""

    def test_logger_creates_output_dir(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger

        output_dir = tmp_path / "run_001"
        logger = ArtifactLogger(output_dir=str(output_dir))
        assert output_dir.exists()

    def test_logger_saves_inputs(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        logger = ArtifactLogger(output_dir=str(tmp_path / "run_001"))

        ex = InputExample(
            example_id="ex_1",
            question="Q?",
            task_type="single_answer_qa",
            gold=GoldAnswer(single_answer="A", multi_answers=None, unknown_allowed=False),
            metadata={"dataset": "nq_open", "split": "dev"},
        )
        logger.log_input(ex)
        logger.flush()

        inputs_file = tmp_path / "run_001" / "inputs.jsonl"
        assert inputs_file.exists()
        lines = inputs_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["example_id"] == "ex_1"

    def test_logger_saves_retrievals(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger
        from rag_baseline.schemas.retrieval import RetrievalOutput, RetrievedPassage

        logger = ArtifactLogger(output_dir=str(tmp_path / "run_001"))

        passage = RetrievedPassage(
            passage_id="p1", text="text", source="doc_1",
            retrieval_score=5.0, rank=1,
        )
        output = RetrievalOutput(example_id="ex_1", retrieved_passages=[passage])
        logger.log_retrieval(output)
        logger.flush()

        retrievals_file = tmp_path / "run_001" / "retrievals.jsonl"
        assert retrievals_file.exists()

    def test_logger_saves_predictions(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger
        from rag_baseline.schemas.generation import GenerationOutput, ParsedOutput

        logger = ArtifactLogger(output_dir=str(tmp_path / "run_001"))

        parsed = ParsedOutput(single_answer="A", multi_answers=None, unknown=False)
        gen = GenerationOutput(
            example_id="ex_1", raw_model_output="A", parsed_output=parsed,
        )
        logger.log_prediction(gen)
        logger.flush()

        preds_file = tmp_path / "run_001" / "predictions.jsonl"
        assert preds_file.exists()

    def test_logger_saves_evaluations(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger
        from rag_baseline.schemas.evaluation import EvaluationOutput, Metrics

        logger = ArtifactLogger(output_dir=str(tmp_path / "run_001"))

        metrics = Metrics(exact_match=True, normalized_match=True,
                          multi_answer_score=None, answer_category=None)
        ev = EvaluationOutput(
            example_id="ex_1", dataset="nq_open",
            baseline_name="test", metrics=metrics,
        )
        logger.log_evaluation(ev)
        logger.flush()

        evals_file = tmp_path / "run_001" / "evaluations.jsonl"
        assert evals_file.exists()

    def test_logger_saves_prompts(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger
        from rag_baseline.schemas.prompt import PromptRecord, PromptMetadata

        logger = ArtifactLogger(output_dir=str(tmp_path / "run_001"))

        meta = PromptMetadata(model_name="m", temperature=0.0, max_context_passages=4)
        record = PromptRecord(
            example_id="ex_1", baseline_name="b", answer_mode="single",
            used_passage_ids=["p1"], prompt_text="prompt", prompt_metadata=meta,
        )
        logger.log_prompt(record)
        logger.flush()

        prompts_file = tmp_path / "run_001" / "prompts.jsonl"
        assert prompts_file.exists()

    def test_logger_saves_run_config(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger
        from rag_baseline.config.schema import RunConfig

        logger = ArtifactLogger(output_dir=str(tmp_path / "run_001"))

        cfg = RunConfig(
            dataset="nq_open", split="dev", retriever_type="hybrid",
            reranker_enabled=True, generator_model="m",
            prompt_family="single_answer", top_k_retrieval=10,
            top_k_after_rerank=5, context_strategy="full",
            answer_mode="single", output_dir=str(tmp_path / "run_001"),
            random_seed=42,
        )
        logger.save_run_config(cfg)

        config_file = tmp_path / "run_001" / "run_config.yaml"
        assert config_file.exists()

    def test_logger_saves_summary_metrics(self, tmp_path):
        from rag_baseline.logging.artifact_logger import ArtifactLogger

        logger = ArtifactLogger(output_dir=str(tmp_path / "run_001"))

        summary = {"accuracy": 0.85, "abstention_rate": 0.05}
        logger.save_summary_metrics(summary)

        summary_file = tmp_path / "run_001" / "summary_metrics.json"
        assert summary_file.exists()
        data = json.loads(summary_file.read_text())
        assert data["accuracy"] == 0.85
