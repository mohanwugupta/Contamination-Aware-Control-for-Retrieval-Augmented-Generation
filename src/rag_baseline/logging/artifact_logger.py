"""Structured artifact logger (PRD 1 §14).

Saves all intermediate artifacts in JSONL format to the run output directory.
Required files: inputs.jsonl, retrievals.jsonl, reranks.jsonl,
prompts.jsonl, predictions.jsonl, evaluations.jsonl, run_config.yaml,
summary_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from rag_baseline.schemas.evaluation import EvaluationOutput
from rag_baseline.schemas.generation import GenerationOutput
from rag_baseline.schemas.input import InputExample
from rag_baseline.schemas.prompt import PromptRecord
from rag_baseline.schemas.rerank import RerankOutput
from rag_baseline.schemas.retrieval import RetrievalOutput


class ArtifactLogger:
    """Logger that writes structured JSONL artifacts per run."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffer for batched writing
        self._buffers: dict[str, list[str]] = {
            "inputs": [],
            "retrievals": [],
            "reranks": [],
            "prompts": [],
            "predictions": [],
            "evaluations": [],
        }

    def _append(self, key: str, record: BaseModel) -> None:
        """Append a record to the buffer."""
        self._buffers[key].append(record.model_dump_json())

    def log_input(self, example: InputExample) -> None:
        self._append("inputs", example)

    def log_retrieval(self, output: RetrievalOutput) -> None:
        self._append("retrievals", output)

    def log_rerank(self, output: RerankOutput) -> None:
        self._append("reranks", output)

    def log_prompt(self, record: PromptRecord) -> None:
        self._append("prompts", record)

    def log_prediction(self, output: GenerationOutput) -> None:
        self._append("predictions", output)

    def log_evaluation(self, output: EvaluationOutput) -> None:
        self._append("evaluations", output)

    def flush(self) -> None:
        """Write all buffered records to JSONL files."""
        for key, records in self._buffers.items():
            if records:
                filepath = self.output_dir / f"{key}.jsonl"
                with open(filepath, "a") as f:
                    for record in records:
                        f.write(record + "\n")
                self._buffers[key] = []

    def save_run_config(self, config: object) -> None:
        """Save run config as YAML."""
        from rag_baseline.config.schema import RunConfig
        if isinstance(config, RunConfig):
            config.to_yaml(self.output_dir / "run_config.yaml")

    def save_summary_metrics(self, metrics: dict) -> None:
        """Save summary metrics as JSON."""
        filepath = self.output_dir / "summary_metrics.json"
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
