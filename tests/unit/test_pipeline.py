"""RED tests for the pipeline runner (PRD 1 §8).

The pipeline must execute the full stage structure:
load → normalize → retrieve → rerank → assemble → prompt → generate →
parse → evaluate → log
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

class _MockReranker:
    """Deterministic mock reranker that reverses passage order.

    This is intentionally simple: it reverses the input list so tests can
    assert the reranked order differs from the retrieval order, without
    needing a real cross-encoder model.
    """

    def rerank(self, query: str, passages, example_id: str = ""):
        from rag_baseline.schemas.rerank import RerankOutput, RerankedPassage

        reversed_passages = list(reversed(passages))
        reranked = [
            RerankedPassage(
                passage_id=p.passage_id,
                text=p.text,
                source=p.source,
                retrieval_score=p.retrieval_score,
                rerank_score=float(len(reversed_passages) - rank),
                rank_after_rerank=rank + 1,
            )
            for rank, p in enumerate(reversed_passages)
        ]
        return RerankOutput(example_id=example_id, reranked_passages=reranked)


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


# ---------------------------------------------------------------------------
# Reranker wiring + top_k_after_rerank enforcement tests
# ---------------------------------------------------------------------------

class TestPipelineReranker:
    """Tests that the reranker is actually invoked and top_k_after_rerank is
    enforced regardless of context_strategy.

    These tests expose two bugs in the original runner:
      1. The reranker was never instantiated or called (Step 3 was a stub comment).
      2. top_k_after_rerank was only applied for context_strategy='reduced';
         context_strategy='full' always passed all top_k_retrieval passages to the
         generator, silently ignoring the configured pruning.
    """

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
        ]

    @pytest.fixture
    def large_corpus(self):
        """Corpus with 10 passages so we can verify pruning from 10→5."""
        return [
            {"passage_id": f"p{i}", "text": f"Passage {i} about basketball.", "source": f"doc_{i}"}
            for i in range(1, 11)
        ]

    @pytest.fixture
    def stub_retriever(self):
        """Dense-like stub retriever that avoids the rank_bm25 dependency."""
        class _StubRetriever:
            def index(self, corpus):
                self._corpus = corpus

            def retrieve(self, query, top_k):
                from rag_baseline.schemas.retrieval import RetrievedPassage, RetrievalOutput
                passages = [
                    RetrievedPassage(
                        passage_id=p["passage_id"], text=p["text"],
                        source=p.get("source", ""), retrieval_score=float(top_k - i), rank=i + 1,
                    )
                    for i, p in enumerate(self._corpus[:top_k])
                ]
                return RetrievalOutput(example_id="", retrieved_passages=passages)

        return _StubRetriever()

    def test_reranker_is_invoked_when_enabled(self, tmp_path, sample_examples, large_corpus, stub_retriever):
        """Bug 1: runner must call the reranker when reranker_enabled=True.

        We inject a mock reranker that records calls.  If the runner never
        calls it, the assertion fails.
        """
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        call_log = []

        class _CountingReranker(_MockReranker):
            def rerank(self, query, passages, example_id=""):
                call_log.append(example_id)
                return super().rerank(query, passages, example_id)

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="dense",
            reranker_enabled=True,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "run_rerank"),
            random_seed=42,
        )

        runner = PipelineRunner(
            config=config,
            generator=MockGenerator(default_response="1963"),
            retriever=stub_retriever,
            reranker=_CountingReranker(),
        )
        runner.index_corpus(large_corpus)
        runner.run(sample_examples)

        assert len(call_log) == len(sample_examples), (
            "Reranker was not called for every example. "
            "PipelineRunner must invoke the reranker when reranker_enabled=True."
        )

    def test_top_k_after_rerank_enforced_with_full_strategy(
        self, tmp_path, sample_examples, large_corpus, stub_retriever
    ):
        """Bug 2: top_k_after_rerank=5 must cap the prompt at 5 passages even
        when context_strategy='full'.

        Previously, context_strategy='full' passed max_passages=None to
        assemble_context, so all top_k_retrieval (10) passages ended up in the
        prompt regardless of top_k_after_rerank.
        """
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="dense",
            reranker_enabled=True,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "run_full_prune"),
            random_seed=42,
        )

        runner = PipelineRunner(
            config=config,
            generator=MockGenerator(default_response="1963"),
            retriever=stub_retriever,
            reranker=_MockReranker(),
        )
        runner.index_corpus(large_corpus)
        runner.run(sample_examples)

        prompts_file = tmp_path / "run_full_prune" / "prompts.jsonl"
        for line in prompts_file.read_text().strip().splitlines():
            record = json.loads(line)
            n_passages = len(record["used_passage_ids"])
            assert n_passages <= config.top_k_after_rerank, (
                f"Prompt used {n_passages} passages but top_k_after_rerank="
                f"{config.top_k_after_rerank}. The runner must prune after "
                "reranking even when context_strategy='full'."
            )

    def test_no_reranker_full_strategy_uses_all_retrieved(
        self, tmp_path, sample_examples, large_corpus, stub_retriever
    ):
        """Without a reranker, context_strategy='full' correctly uses all
        top_k_retrieval passages (no spurious pruning introduced by the fix).
        """
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="dense",
            reranker_enabled=False,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "run_no_rerank"),
            random_seed=42,
        )

        runner = PipelineRunner(
            config=config,
            generator=MockGenerator(default_response="1963"),
            retriever=stub_retriever,
        )
        runner.index_corpus(large_corpus)
        runner.run(sample_examples)

        prompts_file = tmp_path / "run_no_rerank" / "prompts.jsonl"
        for line in prompts_file.read_text().strip().splitlines():
            record = json.loads(line)
            # Without reranker, all retrieved passages should be in the prompt
            assert len(record["used_passage_ids"]) == config.top_k_retrieval

    def test_reranker_artifact_logged(self, tmp_path, sample_examples, large_corpus, stub_retriever):
        """When reranking is performed, a reranks.jsonl artifact must be saved."""
        from rag_baseline.pipeline.runner import PipelineRunner
        from rag_baseline.generation.vllm_generator import MockGenerator
        from rag_baseline.config.schema import RunConfig

        config = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="dense",
            reranker_enabled=True,
            generator_model="mock",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir=str(tmp_path / "run_rerank_artifact"),
            random_seed=42,
        )

        runner = PipelineRunner(
            config=config,
            generator=MockGenerator(default_response="1963"),
            retriever=stub_retriever,
            reranker=_MockReranker(),
        )
        runner.index_corpus(large_corpus)
        runner.run(sample_examples)

        reranks_file = tmp_path / "run_rerank_artifact" / "reranks.jsonl"
        assert reranks_file.exists(), "reranks.jsonl must be written when reranking is used"
        lines = reranks_file.read_text().strip().splitlines()
        assert len(lines) == len(sample_examples)

