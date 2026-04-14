"""Run configuration schema (PRD 1 §15).

All major pipeline behavior is config-driven.
Two runs with the same config and seed must be reproducible
without code edits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator


RetrieverType = Literal["dense", "sparse", "hybrid", "none"]
ContextStrategy = Literal["full", "reduced", "none"]
AnswerMode = Literal["single", "multi", "unknown_or_abstain"]
PromptFamily = Literal["single_answer", "multi_answer", "unknown_compatible"]


class RunConfig(BaseModel):
    """Top-level configuration for a single baseline run."""

    dataset: str
    split: str
    retriever_type: RetrieverType
    reranker_enabled: bool
    generator_model: str
    prompt_family: PromptFamily
    top_k_retrieval: int
    top_k_after_rerank: int
    context_strategy: ContextStrategy
    answer_mode: AnswerMode
    output_dir: str
    random_seed: int

    # Optional fields with defaults
    dense_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    generator_temperature: float = 0.0
    generator_max_tokens: int = 512
    vllm_base_url: str = "http://localhost:8000/v1"

    # --- validators ---

    @field_validator("top_k_retrieval")
    @classmethod
    def top_k_retrieval_positive(cls, v: int) -> int:
        # Allow 0 only for LLM-only mode — validated at model level
        if v < 0:
            raise ValueError("top_k_retrieval must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_top_k_and_strategy(self) -> "RunConfig":
        """Cross-validate top_k and context_strategy consistency."""
        if self.retriever_type == "none":
            # LLM-only mode: top_k must be 0 and context_strategy must be none
            if self.context_strategy != "none":
                raise ValueError(
                    "context_strategy must be 'none' when retriever_type is 'none'"
                )
        else:
            if self.top_k_retrieval < 1:
                raise ValueError(
                    "top_k_retrieval must be >= 1 when retriever_type is not 'none'"
                )
        return self

    # --- derived properties ---

    @property
    def baseline_name(self) -> str:
        """Derive a human-readable baseline name from config fields.

        Always includes the dataset so names are globally unique across the
        configs/baselines/ directory — even when the same pipeline variant is
        applied to multiple datasets (e.g. ``ramdocs_hybrid_rerank_full`` vs
        ``ambigdocs_hybrid_rerank_full``).
        """
        if self.retriever_type == "none":
            pipeline = "llm_only"
        else:
            rerank_part = "rerank" if self.reranker_enabled else "norerank"
            pipeline = f"{self.retriever_type}_{rerank_part}_{self.context_strategy}"

        return f"{self.dataset}_{pipeline}"

    # --- I/O ---

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Load a RunConfig from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save a RunConfig to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
