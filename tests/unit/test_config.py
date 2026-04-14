"""RED tests for the run configuration schema (PRD 1 §15).

The config must drive all major pipeline behavior without code edits.
"""

import pytest
from pydantic import ValidationError


class TestRunConfig:
    """Tests for the config-driven run schema."""

    def test_valid_config_creates(self):
        from rag_baseline.config.schema import RunConfig

        cfg = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="hybrid",
            reranker_enabled=True,
            generator_model="Qwen/Qwen2.5-32B-Instruct",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir="/tmp/runs/run_001",
            random_seed=42,
        )
        assert cfg.dataset == "nq_open"
        assert cfg.retriever_type == "hybrid"
        assert cfg.reranker_enabled is True

    def test_retriever_type_must_be_valid(self):
        from rag_baseline.config.schema import RunConfig

        with pytest.raises(ValidationError):
            RunConfig(
                dataset="nq_open",
                split="dev",
                retriever_type="quantum_retriever",
                reranker_enabled=False,
                generator_model="m",
                prompt_family="single_answer",
                top_k_retrieval=10,
                top_k_after_rerank=5,
                context_strategy="full",
                answer_mode="single",
                output_dir="/tmp/runs/run_001",
                random_seed=42,
            )

    def test_context_strategy_must_be_valid(self):
        from rag_baseline.config.schema import RunConfig

        with pytest.raises(ValidationError):
            RunConfig(
                dataset="nq_open",
                split="dev",
                retriever_type="dense",
                reranker_enabled=False,
                generator_model="m",
                prompt_family="single_answer",
                top_k_retrieval=10,
                top_k_after_rerank=5,
                context_strategy="magic",
                answer_mode="single",
                output_dir="/tmp/runs/run_001",
                random_seed=42,
            )

    def test_answer_mode_must_be_valid(self):
        from rag_baseline.config.schema import RunConfig

        with pytest.raises(ValidationError):
            RunConfig(
                dataset="nq_open",
                split="dev",
                retriever_type="dense",
                reranker_enabled=False,
                generator_model="m",
                prompt_family="single_answer",
                top_k_retrieval=10,
                top_k_after_rerank=5,
                context_strategy="full",
                answer_mode="invalid",
                output_dir="/tmp/runs/run_001",
                random_seed=42,
            )

    def test_prompt_family_must_be_valid(self):
        from rag_baseline.config.schema import RunConfig

        with pytest.raises(ValidationError):
            RunConfig(
                dataset="nq_open",
                split="dev",
                retriever_type="dense",
                reranker_enabled=False,
                generator_model="m",
                prompt_family="secret_trick",
                top_k_retrieval=10,
                top_k_after_rerank=5,
                context_strategy="full",
                answer_mode="single",
                output_dir="/tmp/runs/run_001",
                random_seed=42,
            )

    def test_top_k_must_be_positive(self):
        from rag_baseline.config.schema import RunConfig

        with pytest.raises(ValidationError):
            RunConfig(
                dataset="nq_open",
                split="dev",
                retriever_type="dense",
                reranker_enabled=False,
                generator_model="m",
                prompt_family="single_answer",
                top_k_retrieval=0,
                top_k_after_rerank=5,
                context_strategy="full",
                answer_mode="single",
                output_dir="/tmp/runs/run_001",
                random_seed=42,
            )

    def test_load_from_yaml(self, tmp_path):
        from rag_baseline.config.schema import RunConfig

        yaml_content = """
dataset: nq_open
split: dev
retriever_type: hybrid
reranker_enabled: true
generator_model: Qwen/Qwen2.5-32B-Instruct
prompt_family: single_answer
top_k_retrieval: 10
top_k_after_rerank: 5
context_strategy: full
answer_mode: single
output_dir: /tmp/runs/run_001
random_seed: 42
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        cfg = RunConfig.from_yaml(cfg_file)
        assert cfg.dataset == "nq_open"
        assert cfg.random_seed == 42

    def test_config_round_trip_json(self):
        from rag_baseline.config.schema import RunConfig

        cfg = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="hybrid",
            reranker_enabled=True,
            generator_model="m",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir="/tmp/runs/run_001",
            random_seed=42,
        )
        json_str = cfg.model_dump_json()
        cfg2 = RunConfig.model_validate_json(json_str)
        assert cfg == cfg2

    def test_baseline_name_derived(self):
        """Config should derive a human-readable baseline_name."""
        from rag_baseline.config.schema import RunConfig

        cfg = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="hybrid",
            reranker_enabled=True,
            generator_model="m",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir="/tmp/runs/run_001",
            random_seed=42,
        )
        assert cfg.baseline_name == "nq_open_hybrid_rerank_full"

    def test_baseline_name_no_rerank(self):
        from rag_baseline.config.schema import RunConfig

        cfg = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="dense",
            reranker_enabled=False,
            generator_model="m",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=5,
            context_strategy="full",
            answer_mode="single",
            output_dir="/tmp/runs/run_001",
            random_seed=42,
        )
        assert cfg.baseline_name == "nq_open_dense_norerank_full"

    def test_baseline_name_reduced_context(self):
        from rag_baseline.config.schema import RunConfig

        cfg = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="hybrid",
            reranker_enabled=True,
            generator_model="m",
            prompt_family="single_answer",
            top_k_retrieval=10,
            top_k_after_rerank=2,
            context_strategy="reduced",
            answer_mode="single",
            output_dir="/tmp/runs/run_001",
            random_seed=42,
        )
        assert cfg.baseline_name == "nq_open_hybrid_rerank_reduced"

    def test_llm_only_config(self):
        """Baseline 0 — LLM-only control (no retrieval)."""
        from rag_baseline.config.schema import RunConfig

        cfg = RunConfig(
            dataset="nq_open",
            split="dev",
            retriever_type="none",
            reranker_enabled=False,
            generator_model="m",
            prompt_family="single_answer",
            top_k_retrieval=0,
            top_k_after_rerank=0,
            context_strategy="none",
            answer_mode="single",
            output_dir="/tmp/runs/run_001",
            random_seed=42,
        )
        assert cfg.baseline_name == "nq_open_llm_only"
