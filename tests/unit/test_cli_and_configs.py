"""RED tests for baseline config YAML files and CLI entrypoint (PRD 1 §15, §6).

PRD 1 requires:
- Multiple baseline configs runnable without code edits
- Config-driven run script
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "baselines"


class TestBaselineConfigs:
    """Baseline config YAML files must exist and be loadable."""

    def test_vanilla_rag_config_exists(self):
        assert (CONFIGS_DIR / "vanilla_rag.yaml").exists()

    def test_hybrid_rag_config_exists(self):
        assert (CONFIGS_DIR / "hybrid_rag.yaml").exists()

    def test_hybrid_rerank_config_exists(self):
        assert (CONFIGS_DIR / "hybrid_rerank.yaml").exists()

    def test_reduced_context_config_exists(self):
        assert (CONFIGS_DIR / "reduced_context.yaml").exists()

    def test_llm_only_config_exists(self):
        assert (CONFIGS_DIR / "llm_only.yaml").exists()

    def test_all_configs_are_valid_run_configs(self):
        from rag_baseline.config.schema import RunConfig

        for yaml_file in CONFIGS_DIR.glob("*.yaml"):
            config = RunConfig.from_yaml(yaml_file)
            assert config.dataset is not None
            assert config.output_dir is not None

    def test_baseline_names_are_distinct(self):
        from rag_baseline.config.schema import RunConfig

        names = set()
        for yaml_file in CONFIGS_DIR.glob("*.yaml"):
            config = RunConfig.from_yaml(yaml_file)
            names.add(config.baseline_name)
        # All config files should produce unique baseline names
        yaml_count = len(list(CONFIGS_DIR.glob("*.yaml")))
        assert len(names) == yaml_count


class TestCLIEntrypoint:
    """CLI entrypoint must exist and be importable."""

    def test_cli_module_importable(self):
        from rag_baseline.cli import main  # noqa: F401

    def test_cli_dry_run_flag(self, tmp_path):
        """CLI with --dry-run should validate config and exit without running."""
        from rag_baseline.cli import main

        # Create a minimal config for dry-run
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            """
dataset: nq_open
split: validation
retriever_type: none
reranker_enabled: false
generator_model: mock
prompt_family: single_answer
top_k_retrieval: 0
top_k_after_rerank: 0
context_strategy: none
answer_mode: single
output_dir: {output_dir}
random_seed: 42
""".format(output_dir=str(tmp_path / "output"))
        )

        # Dry run should succeed without errors
        result = main(["--config", str(config_path), "--dry-run"])
        assert result == 0
