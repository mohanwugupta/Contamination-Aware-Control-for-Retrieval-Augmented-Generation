"""CLI entrypoint for running baseline RAG systems (PRD 1 §15).

Usage:
    # External vLLM server (default — used by SLURM scripts)
    python -m rag_baseline.cli --config configs/baselines/hybrid_rerank.yaml

    # In-process vLLM (mirrors God's Reach Qwen72BProvider)
    python -m rag_baseline.cli --config configs/baselines/hybrid_rerank.yaml \
        --generator-mode in-process --model-path /path/to/model --tp 2

    # Dry run (validate config only)
    python -m rag_baseline.cli --config path/to/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Run a baseline RAG system from a config file.",
        prog="rag_baseline",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML run config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and print plan without executing.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples to process (for quick testing).",
    )

    # Generator mode (server vs in-process)
    parser.add_argument(
        "--generator-mode",
        choices=["server", "in-process"],
        default="server",
        help=(
            "How to run the LLM. "
            "'server' (default) connects to a running vLLM OpenAI server. "
            "'in-process' loads the model directly via vllm.LLM (like God's Reach)."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path for in-process mode (overrides config generator_model).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size for in-process mode (default: 2).",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=None,
        dest="generator_model",
        help=(
            "Override the generator_model field in the config. "
            "In server mode this must match the --served-model-name passed "
            "to the vLLM server. Derived from MODEL_PATH by the SLURM scripts "
            "so configs never need hardcoded model names."
        ),
    )

    args = parser.parse_args(argv)

    # Load and validate config
    from rag_baseline.config.schema import RunConfig

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = RunConfig.from_yaml(config_path)
    except Exception as e:
        print(f"Error: Invalid config: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        effective_model = args.generator_model or config.generator_model
        print("=" * 60)
        print("DRY RUN — Config validated successfully")
        print("=" * 60)
        print(f"  Baseline name:    {config.baseline_name}")
        print(f"  Dataset:          {config.dataset}")
        print(f"  Split:            {config.split}")
        print(f"  Retriever:        {config.retriever_type}")
        print(f"  Reranker:         {'enabled' if config.reranker_enabled else 'disabled'}")
        print(f"  Context strategy: {config.context_strategy}")
        print(f"  Answer mode:      {config.answer_mode}")
        print(f"  Generator:        {effective_model}")
        print(f"  Generator mode:   {args.generator_mode}")
        if args.generator_mode == "in-process":
            print(f"  Model path:       {args.model_path or '(from config/env)'}")
            print(f"  Tensor parallel:  {args.tp or 2}")
        else:
            print(f"  vLLM base URL:    {config.vllm_base_url}")
        print(f"  Output dir:       {config.output_dir}")
        print(f"  Random seed:      {config.random_seed}")
        print("=" * 60)
        return 0

    # Full run
    return _execute_run(
        config,
        max_examples=args.max_examples,
        generator_mode=args.generator_mode,
        model_path=args.model_path,
        tensor_parallel=args.tp,
        generator_model_override=args.generator_model,
    )


def _execute_run(
    config,
    max_examples: int | None = None,
    generator_mode: str = "server",
    model_path: str | None = None,
    tensor_parallel: int | None = None,
    generator_model_override: str | None = None,
) -> int:
    """Execute a full pipeline run.

    Args:
        config: Validated RunConfig.
        max_examples: Optional limit on examples to process.
        generator_mode: "server" or "in-process".
        model_path: Local model path (in-process mode).
        tensor_parallel: Tensor parallel size (in-process mode).
        generator_model_override: If set, overrides ``config.generator_model``.
            The SLURM scripts derive this from ``MODEL_PATH`` so it always
            matches ``--served-model-name`` and never calls an external API.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from rag_baseline.adapters import create_adapter
    from rag_baseline.generation.vllm_generator import (
        InProcessVLLMGenerator,
        VLLMGenerator,
    )
    from rag_baseline.pipeline.runner import PipelineRunner

    print(f"Starting run: {config.baseline_name}")
    print(f"  Dataset: {config.dataset} ({config.split})")

    # 1. Load dataset
    adapter = create_adapter(config.dataset)
    examples = adapter.load(split=config.split)

    if max_examples is not None:
        examples = examples[:max_examples]
        print(f"  Limiting to {max_examples} examples")

    print(f"  Loaded {len(examples)} examples")

    # 2. Create generator (server mode or in-process mode)
    effective_model = generator_model_override or config.generator_model
    if generator_mode == "in-process":
        print(f"  Generator mode: in-process (TP={tensor_parallel or 2})")
        generator = InProcessVLLMGenerator(
            model_path=model_path or effective_model,
            tensor_parallel_size=tensor_parallel or 2,
            temperature=config.generator_temperature,
            max_tokens=config.generator_max_tokens or 512,
        )
    else:
        print(f"  Generator mode: server ({config.vllm_base_url})")
        generator = VLLMGenerator(
            model_name=effective_model,
            base_url=config.vllm_base_url or "http://localhost:8000/v1",
            temperature=config.generator_temperature,
            max_tokens=config.generator_max_tokens or 512,
        )
        # Health check — fail fast if server is not running
        if not generator.health_check():
            print(
                f"  WARNING: vLLM server not reachable at {config.vllm_base_url}",
                file=sys.stderr,
            )
            print(
                "  Waiting for server (up to 10 min)...",
                file=sys.stderr,
            )
            try:
                generator.wait_until_ready(timeout=600.0)
            except ConnectionError as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                return 1

    # 3. Create pipeline
    runner = PipelineRunner(config=config, generator=generator)

    # 4. Index corpus (if applicable)
    corpus = adapter.get_corpus()
    if corpus is not None:
        print(f"  Indexing {len(corpus)} corpus passages...")
        runner.index_corpus(corpus)

    # 5. Run pipeline
    print("  Running pipeline...")
    results = runner.run(examples)

    print(f"  Completed. {len(results)} examples evaluated.")
    print(f"  Artifacts saved to: {config.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
