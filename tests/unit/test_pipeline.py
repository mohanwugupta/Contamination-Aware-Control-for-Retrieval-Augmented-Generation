"""RED tests for the pipeline runner (PRD 1 §8).

The pipeline must execute the full stage structure:
load → normalize → retrieve → rerank → assemble → prompt → generate →
parse → evaluate → log
"""

import json
import pytest


class TestPipelineRunner:
    """Tests for the end-to-end pipeline runner using mock generator."""

    @pytest.fixture
    def sample_examples(self):
        from rag_baseline.schemas.input import InputExample, GoldAnswer

        return [
            InputExample(
                example_id="ex_001",
                question="When was Michael Jordan born?",
                task_type="single_answer_qa",
                gold=GoldAnswer(single_answer="1963", multi_answers=None, unknown_allowed=False),
                metadata={"dataset": "nq_open", "split": "dev"},
            ),
            InputExample(
                example_id="ex_002",
                question="What is the capital of France?",
                task_type="single_answer_qa",
                gold=GoldAnswer(single_answer="Paris", multi_answers=None, unknown_allowed=False),
                metadata={"dataset": "nq_open", "split": "dev"},
            ),
        ]

    @pytest.fixture
    def sample_corpus(self):
        return [
            {"passage_id": "p1", "text": "Michael Jordan was born in 1963 in Brooklyn.", "source": "doc_1"},
            {"passage_id": "p2", "text": "Michael Jordan is a retired basketball player.", "source": "doc_2"},
            {"passage_id": "p3", "text": "The capital of France is Paris.", "source": "doc_3"},
            {"passage_id": "p4", "text": "Michael B. Jordan is an American actor.", "source": "doc_4"},
            {"passage_id": "p5", "text": "Basketball was invented by James Naismith.", "source": "doc_5"},
        ]

    def test_pipeline_runs_end_to_end(self, tmp_path, sample_examples, sample_corpus):
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="sparse",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=3,
            top_k_after_rerank=3,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "run_001"),
            random_seed=42,
        )

        generator = MockGenerator(default_response="1963")
        runner = PipelineRunner(config=config, generator=generator)
        runner.index_corpus(sample_corpus)

        results = runner.run(sample_examples)

        # Should have results for both examples
        assert len(results) == 2

    def test_pipeline_produces_artifacts(self, tmp_path, sample_examples, sample_corpus):
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="sparse",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=3,
            top_k_after_rerank=3,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "run_001"),
            random_seed=42,
        )

        generator = MockGenerator(default_response="1963")
        runner = PipelineRunner(config=config, generator=generator)
        runner.index_corpus(sample_corpus)
        runner.run(sample_examples)

        run_dir = tmp_path / "run_001"
        # Check all required artifacts exist
        assert (run_dir / "inputs.jsonl").exists()
        assert (run_dir / "retrievals.jsonl").exists()
        assert (run_dir / "prompts.jsonl").exists()
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "evaluations.jsonl").exists()
        assert (run_dir / "run_config.yaml").exists()
        assert (run_dir / "summary_metrics.json").exists()

    def test_pipeline_with_reduced_context(self, tmp_path, sample_examples, sample_corpus):
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="sparse",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=5,
            top_k_after_rerank=2,
            context_strategy="reduced",
            answer_mode="single",
            output_dir=str(tmp_path / "run_reduced"),
            random_seed=42,
        )

        generator = MockGenerator(default_response="1963")
        runner = PipelineRunner(config=config, generator=generator)
        runner.index_corpus(sample_corpus)
        results = runner.run(sample_examples)

        # Check that prompts used reduced context
        prompts_file = tmp_path / "run_reduced" / "prompts.jsonl"
        lines = prompts_file.read_text().strip().split("\n")
        for line in lines:
            record = json.loads(line)
            # Reduced context should use at most 2 passages
            assert len(record["used_passage_ids"]) <= 2

    def test_pipeline_llm_only(self, tmp_path, sample_examples):
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="none",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=0,
            top_k_after_rerank=0,
            context_strategy="none",
            answer_mode="single",
            output_dir=str(tmp_path / "run_llm_only"),
            random_seed=42,
        )

        generator = MockGenerator(default_response="1963")
        runner = PipelineRunner(config=config, generator=generator)
        # No corpus needed for LLM-only
        results = runner.run(sample_examples)
        assert len(results) == 2

    def test_pipeline_summary_metrics(self, tmp_path, sample_examples, sample_corpus):
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="sparse",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=3,
            top_k_after_rerank=3,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "run_metrics"),
            random_seed=42,
        )

        generator = MockGenerator(default_response="1963")
        runner = PipelineRunner(config=config, generator=generator)
        runner.index_corpus(sample_corpus)
        runner.run(sample_examples)

        summary_file = tmp_path / "run_metrics" / "summary_metrics.json"
        summary = json.loads(summary_file.read_text())
        # Summary should contain accuracy metrics
        assert "total_examples" in summary
        assert "accuracy" in summary or "normalized_match_rate" in summary
