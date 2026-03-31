"""Integration tests for the full pipeline with dataset adapters.

These tests verify that adapters + pipeline runner work together
end-to-end using mock generator and BM25 retrieval.
"""

import json

import pytest


class TestNQOpenIntegration:
    """End-to-end: NQ adapter → BM25 retrieval → MockGenerator → evaluation."""

    @pytest.fixture
    def nq_rows(self):
        return [
            {"question": "When was Michael Jordan born?", "answer": ["1963"]},
            {"question": "What is the capital of France?", "answer": ["Paris"]},
            {"question": "Who wrote Romeo and Juliet?", "answer": ["William Shakespeare", "Shakespeare"]},
        ]

    @pytest.fixture
    def corpus(self):
        return [
            {"passage_id": "p1", "text": "Michael Jordan was born in 1963 in Brooklyn, New York.", "source": "wiki_mj"},
            {"passage_id": "p2", "text": "Paris is the capital and largest city of France.", "source": "wiki_paris"},
            {"passage_id": "p3", "text": "Romeo and Juliet is a tragedy written by William Shakespeare.", "source": "wiki_rj"},
            {"passage_id": "p4", "text": "Basketball was invented by James Naismith in 1891.", "source": "wiki_bball"},
            {"passage_id": "p5", "text": "France is a country in Western Europe.", "source": "wiki_france"},
        ]

    def test_nq_adapter_to_pipeline(self, tmp_path, nq_rows, corpus):
        from rag_baseline.adapters.nq_open import NQOpenAdapter
        from rag_baseline.config.schema import RunConfig
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.pipeline.runner import PipelineRunner

        # 1. Adapt
        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(nq_rows, split="validation")
        assert len(examples) == 3

        # 2. Configure
        config = RunConfig(
            dataset="nq_open",
            split="validation",
            retriever_type="sparse",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=3,
            top_k_after_rerank=3,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "nq_integration"),
            random_seed=42,
        )

        # 3. Run
        generator = MockGenerator(default_response="1963")
        runner = PipelineRunner(config=config, generator=generator)
        runner.index_corpus(corpus)
        results = runner.run(examples)

        # 4. Verify
        assert len(results) == 3
        run_dir = tmp_path / "nq_integration"
        assert (run_dir / "summary_metrics.json").exists()

        summary = json.loads((run_dir / "summary_metrics.json").read_text())
        assert summary["total_examples"] == 3
        assert summary["dataset"] == "nq_open"

    def test_nq_llm_only_integration(self, tmp_path, nq_rows):
        """LLM-only baseline (no retrieval) works with NQ adapter."""
        from rag_baseline.adapters.nq_open import NQOpenAdapter
        from rag_baseline.config.schema import RunConfig
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.pipeline.runner import PipelineRunner

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(nq_rows, split="validation")

        config = RunConfig(
            dataset="nq_open",
            split="validation",
            retriever_type="none",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=0,
            top_k_after_rerank=0,
            context_strategy="none",
            answer_mode="single",
            output_dir=str(tmp_path / "nq_llm_only"),
            random_seed=42,
        )

        generator = MockGenerator(default_response="1963")
        runner = PipelineRunner(config=config, generator=generator)
        results = runner.run(examples)

        assert len(results) == 3
        # Verify no retrievals.jsonl (empty, since no retriever)
        retrievals_file = tmp_path / "nq_llm_only" / "retrievals.jsonl"
        if retrievals_file.exists():
            assert retrievals_file.read_text().strip() == ""


class TestAmbigDocsIntegration:
    """End-to-end: AmbigDocs adapter → retrieval → MockGenerator → evaluation."""

    @pytest.fixture
    def ambig_rows(self):
        return [
            {
                "qid": 100,
                "ambiguous_entity": "Michael Jordan",
                "question": "Where was Michael Jordan educated?",
                "documents": {
                    "title": [
                        "Michael Jordan (basketball)",
                        "Michael I. Jordan (scientist)",
                    ],
                    "text": [
                        "Michael Jordan attended Laney High School and University of North Carolina.",
                        "Michael I. Jordan studied at UC San Diego and earned his PhD from MIT.",
                    ],
                    "pid": ["mj_bball", "mj_sci"],
                    "answer": [
                        "University of North Carolina",
                        "UC San Diego and MIT",
                    ],
                },
            },
        ]

    def test_ambigdocs_adapter_to_pipeline(self, tmp_path, ambig_rows):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter
        from rag_baseline.config.schema import RunConfig
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.pipeline.runner import PipelineRunner

        # 1. Adapt
        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(ambig_rows, split="validation")
        corpus = adapter.get_corpus()

        assert len(examples) == 1
        assert len(corpus) == 2
        assert examples[0].task_type == "multi_answer_qa"

        # 2. Configure
        config = RunConfig(
            dataset="ambigdocs",
            split="validation",
            retriever_type="sparse",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="multi_answer",
            top_k_retrieval=2,
            top_k_after_rerank=2,
            context_strategy="full",
            answer_mode="multi",
            output_dir=str(tmp_path / "ambig_integration"),
            random_seed=42,
        )

        # 3. Run with multi-answer mock response
        generator = MockGenerator(
            default_response="University of North Carolina\nUC San Diego and MIT"
        )
        runner = PipelineRunner(config=config, generator=generator)
        runner.index_corpus(corpus)
        results = runner.run(examples)

        # 4. Verify
        assert len(results) == 1
        run_dir = tmp_path / "ambig_integration"
        assert (run_dir / "evaluations.jsonl").exists()

    def test_ambigdocs_per_example_docs(self, ambig_rows):
        """Adapter tracks which documents belong to which example."""
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(ambig_rows, split="validation")

        docs = adapter.get_example_documents(examples[0].example_id)
        assert docs is not None
        assert len(docs) == 2
        assert docs[0]["source"] == "Michael Jordan (basketball)"
        assert docs[1]["source"] == "Michael I. Jordan (scientist)"


class TestAdapterFactoryIntegration:
    """Factory + pipeline integration test."""

    def test_factory_to_pipeline_nq(self, tmp_path):
        from rag_baseline.adapters import create_adapter
        from rag_baseline.config.schema import RunConfig
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.pipeline.runner import PipelineRunner

        adapter = create_adapter("nq_open")
        examples = adapter.load_from_dicts(
            [{"question": "Test question?", "answer": ["test"]}],
            split="validation",
        )

        config = RunConfig(
            dataset="nq_open",
            split="validation",
            retriever_type="none",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=0,
            top_k_after_rerank=0,
            context_strategy="none",
            answer_mode="single",
            output_dir=str(tmp_path / "factory_test"),
            random_seed=42,
        )

        generator = MockGenerator(default_response="test")
        runner = PipelineRunner(config=config, generator=generator)
        results = runner.run(examples)

        assert len(results) == 1
        assert results[0].metrics.normalized_match is True
