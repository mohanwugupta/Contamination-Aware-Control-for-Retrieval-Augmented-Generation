"""Pipeline runner (PRD 1 §8).

Orchestrates the full baseline RAG pipeline:
load → retrieve → rerank → assemble → prompt → generate → parse → evaluate → log

All baseline variants differ only in controlled config-driven places:
retriever type, reranking on/off, number of passages, answer mode.

Throughput design
-----------------
The pipeline uses a **three-pass** design to maximise GPU utilisation:

  Pass 1 (CPU)  — retrieve + rerank + assemble + render prompt for every example.
  Pass 2 (GPU)  — send ALL prompts to vLLM concurrently via a thread pool so the
                  server's continuous-batching scheduler can keep the GPUs busy
                  instead of waiting for one sequential request at a time.
  Pass 3 (CPU)  — parse + evaluate + log every result in original order.

Set ``num_workers`` (default 32) to control how many in-flight HTTP requests are
sent to the vLLM server simultaneously.  vLLM queues excess requests internally,
so setting this to the full dataset size is safe and optimal.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_baseline.config.schema import RunConfig
from rag_baseline.context.assembly import assemble_context
from rag_baseline.evaluation.base import evaluate_example
from rag_baseline.generation.vllm_generator import BaseGenerator, GenerationResult
from rag_baseline.logging.artifact_logger import ArtifactLogger
from rag_baseline.parsing.output_parser import parse_output
from rag_baseline.prompts.templates import render_prompt
from rag_baseline.retrieval import create_retriever
from rag_baseline.retrieval.base import BaseRetriever
from rag_baseline.schemas.evaluation import EvaluationOutput
from rag_baseline.schemas.generation import GenerationOutput
from rag_baseline.schemas.input import InputExample
from rag_baseline.schemas.prompt import PromptMetadata, PromptRecord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class _WorkItem:
    """Intermediate state produced by Pass 1 and consumed by Pass 3."""
    example: InputExample
    prompt_record: PromptRecord
    prompt_text: str


class PipelineRunner:
    """Config-driven pipeline runner for baseline RAG systems."""

    def __init__(
        self,
        config: RunConfig,
        generator: BaseGenerator,
        retriever: BaseRetriever | None = None,
        num_workers: int = 32,
    ) -> None:
        self.config = config
        self.generator = generator
        self.num_workers = num_workers
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

        Uses a three-pass approach to keep the GPU busy:
          1. CPU pass  — retrieve + assemble + prompt for all examples.
          2. GPU pass  — generate all prompts concurrently (thread pool →
                         vLLM continuous batching).
          3. CPU pass  — parse + evaluate + log all results.

        Args:
            examples: List of normalized input examples.

        Returns:
            List of evaluation outputs in the same order as *examples*.
        """
        if not examples:
            self.logger.flush()
            self._save_summary({}, 0)
            return []

        # ------------------------------------------------------------------
        # Pass 1 — CPU: retrieve → rerank → assemble → render prompt
        # ------------------------------------------------------------------
        logger.info("Pass 1/3 — preparing %d examples (retrieval + prompts)", len(examples))
        work_items: list[_WorkItem] = [
            self._prepare_example(example) for example in examples
        ]

        # ------------------------------------------------------------------
        # Pass 2 — GPU: generate all prompts concurrently
        # ------------------------------------------------------------------
        prompts = [item.prompt_text for item in work_items]
        effective_workers = min(self.num_workers, len(prompts))
        logger.info(
            "Pass 2/3 — generating %d prompts with %d concurrent workers",
            len(prompts),
            effective_workers,
        )
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            gen_results: list[GenerationResult] = list(
                pool.map(self.generator.generate, prompts)
            )

        # ------------------------------------------------------------------
        # Pass 3 — CPU: parse → evaluate → log (preserves input order)
        # ------------------------------------------------------------------
        logger.info("Pass 3/3 — parsing and evaluating %d results", len(gen_results))
        results: list[EvaluationOutput] = []
        eval_counts: dict[str, int] = {"total": 0, "normalized_match": 0, "exact_match": 0}

        for item, gen_result in zip(work_items, gen_results):
            eval_output = self._finalize_example(item, gen_result)
            results.append(eval_output)

            eval_counts["total"] += 1
            if eval_output.metrics.normalized_match:
                eval_counts["normalized_match"] += 1
            if eval_output.metrics.exact_match:
                eval_counts["exact_match"] += 1

        # Flush all logs
        self.logger.flush()

        # Save summary metrics
        self._save_summary(eval_counts, eval_counts["total"])

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_example(self, example: InputExample) -> _WorkItem:
        """Pass 1 worker: retrieve → rerank → assemble → render prompt.

        Performs all CPU-bound work for one example and returns a
        :class:`_WorkItem` ready to be handed to the generation thread pool.
        """
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
            retrieval_output.example_id = example.example_id
            self.logger.log_retrieval(retrieval_output)
            retrieved_passages = retrieval_output.retrieved_passages

        # Step 3: Rerank (hook — plug reranker here when implemented)
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

        return _WorkItem(
            example=example,
            prompt_record=prompt_record,
            prompt_text=prompt_text,
        )

    def _finalize_example(
        self,
        item: _WorkItem,
        gen_result: GenerationResult,
    ) -> EvaluationOutput:
        """Pass 3 worker: parse → evaluate → log one completed generation."""
        # Step 6: (generation already done in Pass 2)

        # Step 7: Parse
        parsed = parse_output(gen_result.text, answer_mode=self.config.answer_mode)

        gen_output = GenerationOutput(
            example_id=item.example.example_id,
            raw_model_output=gen_result.text,
            parsed_output=parsed,
        )
        self.logger.log_prediction(gen_output)

        # Step 8: Evaluate
        eval_output = evaluate_example(
            dataset=self.config.dataset,
            parsed_output=parsed,
            gold=item.example.gold,
            example_id=item.example.example_id,
            baseline_name=self.config.baseline_name,
        )
        self.logger.log_evaluation(eval_output)

        return eval_output

    def _save_summary(self, eval_counts: dict[str, int], total: int) -> None:
        """Persist summary metrics to disk."""
        summary = {
            "total_examples": total,
            "normalized_match_rate": eval_counts.get("normalized_match", 0) / total if total > 0 else 0,
            "exact_match_rate": eval_counts.get("exact_match", 0) / total if total > 0 else 0,
            "baseline_name": self.config.baseline_name,
            "dataset": self.config.dataset,
            "split": self.config.split,
        }
        self.logger.save_summary_metrics(summary)

    # ------------------------------------------------------------------
    # Legacy single-example entry point (kept for unit-test compatibility)
    # ------------------------------------------------------------------

    def _process_example(self, example: InputExample) -> EvaluationOutput:
        """Process a single example (sequential — used by unit tests only).

        For production throughput use :meth:`run`, which batches all
        generation calls so vLLM can keep the GPUs saturated.
        """
        item = self._prepare_example(example)
        gen_result = self.generator.generate(item.prompt_text)
        return self._finalize_example(item, gen_result)
