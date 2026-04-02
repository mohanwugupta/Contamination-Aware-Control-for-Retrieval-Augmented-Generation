"""Pipeline runner (PRD 1 §8).

Orchestrates the full baseline RAG pipeline:
load → retrieve → rerank → assemble → prompt → generate → parse → evaluate → log

All baseline variants differ only in controlled config-driven places:
retriever type, reranking on/off, number of passages, answer mode.
"""

from __future__ import annotations

from rag_baseline.config.schema import RunConfig
from rag_baseline.context.assembly import assemble_context
from rag_baseline.evaluation.base import evaluate_example
from rag_baseline.generation.vllm_generator import BaseGenerator
from rag_baseline.logging.artifact_logger import ArtifactLogger
from rag_baseline.parsing.output_parser import parse_output
from rag_baseline.prompts.templates import render_prompt
from rag_baseline.retrieval import create_retriever
from rag_baseline.retrieval.base import BaseRetriever
from rag_baseline.schemas.evaluation import EvaluationOutput
from rag_baseline.schemas.generation import GenerationOutput
from rag_baseline.schemas.input import InputExample
from rag_baseline.schemas.prompt import PromptMetadata, PromptRecord


class PipelineRunner:
    """Config-driven pipeline runner for baseline RAG systems."""

    def __init__(
        self,
        config: RunConfig,
        generator: BaseGenerator,
        retriever: BaseRetriever | None = None,
    ) -> None:
        self.config = config
        self.generator = generator
        self.logger = ArtifactLogger(output_dir=config.output_dir)
        self._corpus_indexed: bool = False

        # Create retriever from config if not provided
        if retriever is not None:
            self.retriever = retriever
        elif config.retriever_type != "none":
            self.retriever = create_retriever(
                retriever_type=config.retriever_type,
                dense_model=config.dense_model,
            )
        else:
            self.retriever = None

        # Save config
        self.logger.save_run_config(config)

    def index_corpus(self, corpus: list[dict]) -> None:
        """Index a corpus for retrieval."""
        if self.retriever is not None:
            self.retriever.index(corpus)
        self._corpus_indexed = True

    def run(self, examples: list[InputExample]) -> list[EvaluationOutput]:
        """Run the full pipeline on a list of examples.

        Args:
            examples: List of normalized input examples.

        Returns:
            List of evaluation outputs.
        """
        results: list[EvaluationOutput] = []
        eval_counts = {"total": 0, "normalized_match": 0, "exact_match": 0}

        for example in examples:
            eval_output = self._process_example(example)
            results.append(eval_output)

            # Track metrics
            eval_counts["total"] += 1
            if eval_output.metrics.normalized_match:
                eval_counts["normalized_match"] += 1
            if eval_output.metrics.exact_match:
                eval_counts["exact_match"] += 1

        # Flush all logs
        self.logger.flush()

        # Save summary metrics
        total = eval_counts["total"]
        summary = {
            "total_examples": total,
            "normalized_match_rate": eval_counts["normalized_match"] / total if total > 0 else 0,
            "exact_match_rate": eval_counts["exact_match"] / total if total > 0 else 0,
            "baseline_name": self.config.baseline_name,
            "dataset": self.config.dataset,
            "split": self.config.split,
        }
        self.logger.save_summary_metrics(summary)

        return results

    def _process_example(self, example: InputExample) -> EvaluationOutput:
        """Process a single example through the full pipeline."""
        # Step 1: Log input
        self.logger.log_input(example)

        # Step 2: Retrieve (if applicable)
        retrieved_passages = []
        if self.retriever is not None:
            if not self._corpus_indexed:
                raise RuntimeError(
                    f"Retriever of type '{self.config.retriever_type}' was created but no corpus "
                    "has been indexed. Call index_corpus() before run(), or set "
                    "retriever_type: none in the config for datasets without a bundled corpus "
                    "(e.g. nq_open)."
                )
            retrieval_output = self.retriever.retrieve(
                query=example.question,
                top_k=self.config.top_k_retrieval,
            )
            # Set example_id
            retrieval_output.example_id = example.example_id
            self.logger.log_retrieval(retrieval_output)
            retrieved_passages = retrieval_output.retrieved_passages

        # Step 3: Rerank (if enabled) — skipped for now, hook for later
        # The reranker would be plugged in here; for baseline testing
        # we use retrieved passages as-is when reranker is not enabled.
        context_passages = retrieved_passages

        # Step 4: Assemble context
        assembled = assemble_context(
            passages=context_passages,
            strategy=self.config.context_strategy,
            max_passages=self.config.top_k_after_rerank
            if self.config.context_strategy == "reduced" else None,
        )

        # Step 5: Build prompt
        prompt_text = render_prompt(
            question=example.question,
            context_text=assembled.formatted_text,
            answer_mode=self.config.answer_mode,
        )

        prompt_record = PromptRecord(
            example_id=example.example_id,
            baseline_name=self.config.baseline_name,
            answer_mode=self.config.answer_mode,
            used_passage_ids=assembled.passage_ids,
            prompt_text=prompt_text,
            prompt_metadata=PromptMetadata(
                model_name=self.config.generator_model,
                temperature=self.config.generator_temperature,
                max_context_passages=len(assembled.passages),
            ),
        )
        self.logger.log_prompt(prompt_record)

        # Step 6: Generate
        gen_result = self.generator.generate(prompt_text)

        # Step 7: Parse
        parsed = parse_output(gen_result.text, answer_mode=self.config.answer_mode)

        gen_output = GenerationOutput(
            example_id=example.example_id,
            raw_model_output=gen_result.text,
            parsed_output=parsed,
        )
        self.logger.log_prediction(gen_output)

        # Step 8: Evaluate
        eval_output = evaluate_example(
            dataset=self.config.dataset,
            parsed_output=parsed,
            gold=example.gold,
            example_id=example.example_id,
            baseline_name=self.config.baseline_name,
        )
        self.logger.log_evaluation(eval_output)

        return eval_output
