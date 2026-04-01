"""FaithEval dataset adapter (PRD 1 §4 Tier 2).

Normalizes FaithEval (Salesforce) into InputExample.
FaithEval evaluates contextual faithfulness across three tasks:
    - unanswerable: model should say "unknown"
    - inconsistent: model should say "conflict"
    - counterfactual: context presents wrong info, model should be faithful to context

Three separate HuggingFace datasets:
    Salesforce/FaithEval-unanswerable-v1.0  (2,492 rows, test only)
    Salesforce/FaithEval-inconsistent-v1.0  (1,500 rows, test only)
    Salesforce/FaithEval-counterfactual-v1.0 (1,000 rows, test only)

Fields (all subtasks): id, question, answer, answerKey, choices, context
"""

from __future__ import annotations

from rag_baseline.adapters.base import BaseAdapter, _load_hf_split
from rag_baseline.schemas.input import ExampleMetadata, GoldAnswer, InputExample

VALID_SUBTASKS = ("unanswerable", "inconsistent", "counterfactual", "all")

# Subtask abbreviations used in example_id generation
_SUBTASK_ABBREV = {
    "unanswerable": "un",
    "inconsistent": "ic",
    "counterfactual": "cf",
}

# HuggingFace dataset IDs per subtask
_HF_DATASET_IDS = {
    "unanswerable": "Salesforce/FaithEval-unanswerable-v1.0",
    "inconsistent": "Salesforce/FaithEval-inconsistent-v1.0",
    "counterfactual": "Salesforce/FaithEval-counterfactual-v1.0",
}


class FaithEvalAdapter(BaseAdapter):
    """Adapter for the FaithEval benchmark (Salesforce, ICLR 2025)."""

    DATASET_NAME = "faitheval"

    def __init__(self, subtask: str = "all") -> None:
        if subtask not in VALID_SUBTASKS:
            raise ValueError(
                f"Invalid subtask '{subtask}'. Must be one of {VALID_SUBTASKS}"
            )
        self._subtask = subtask
        self._corpus: list[dict] | None = None
        self._example_contexts: dict[str, str] = {}

    @property
    def subtask(self) -> str:
        return self._subtask

    def load(self, split: str = "test", **kwargs: object) -> list[InputExample]:
        """Load FaithEval from HuggingFace and normalize.

        FaithEval only has a 'test' split. If subtask is 'all', loads
        all three subtasks and combines them.

        Args:
            split: Dataset split (FaithEval only has 'test').

        Returns:
            List of normalized InputExample instances.
        """
        if self._subtask == "all":
            unanswerable = list(
                _load_hf_split(
                    _HF_DATASET_IDS["unanswerable"],
                    split,
                    f"faitheval_unanswerable_{split}",
                )
            )
            inconsistent = list(
                _load_hf_split(
                    _HF_DATASET_IDS["inconsistent"],
                    split,
                    f"faitheval_inconsistent_{split}",
                )
            )
            counterfactual = list(
                _load_hf_split(
                    _HF_DATASET_IDS["counterfactual"],
                    split,
                    f"faitheval_counterfactual_{split}",
                )
            )
            return self.load_all_from_dicts(
                unanswerable=unanswerable,
                inconsistent=inconsistent,
                counterfactual=counterfactual,
                split=split,
            )
        else:
            ds = _load_hf_split(
                _HF_DATASET_IDS[self._subtask],
                split,
                f"faitheval_{self._subtask}_{split}",
            )
            return self.load_from_dicts(list(ds), split=split)

    def load_from_dicts(
        self, rows: list[dict], split: str = "test"
    ) -> list[InputExample]:
        """Normalize raw FaithEval dicts into InputExamples.

        Uses the current subtask setting to determine gold answer semantics.

        Args:
            rows: List of dicts matching HuggingFace FaithEval schema.
            split: Split name for metadata.

        Returns:
            List of InputExample instances.
        """
        subtask = self._subtask if self._subtask != "all" else "unanswerable"
        return self._normalize_rows(rows, subtask=subtask, split=split)

    def load_all_from_dicts(
        self,
        unanswerable: list[dict],
        inconsistent: list[dict],
        counterfactual: list[dict],
        split: str = "test",
    ) -> list[InputExample]:
        """Load and combine all three FaithEval subtasks from dicts.

        Args:
            unanswerable: Raw dicts for the unanswerable subtask.
            inconsistent: Raw dicts for the inconsistent subtask.
            counterfactual: Raw dicts for the counterfactual subtask.
            split: Split name for metadata.

        Returns:
            Combined list of InputExample instances across all subtasks.
        """
        examples: list[InputExample] = []
        examples.extend(
            self._normalize_rows(unanswerable, subtask="unanswerable", split=split)
        )
        examples.extend(
            self._normalize_rows(inconsistent, subtask="inconsistent", split=split)
        )
        examples.extend(
            self._normalize_rows(counterfactual, subtask="counterfactual", split=split)
        )
        return examples

    def get_corpus(self) -> list[dict] | None:
        """Return bundled contexts as a corpus for retrieval.

        Each FaithEval example's context is treated as a single passage.

        Returns:
            List of passage dicts with keys (passage_id, text, source),
            or None if no data has been loaded yet.
        """
        return self._corpus

    def get_example_context(self, example_id: str) -> str | None:
        """Get the context associated with a specific example.

        Args:
            example_id: The example ID (e.g. "faitheval_un_un_001").

        Returns:
            The context string, or None if not found.
        """
        return self._example_contexts.get(example_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_rows(
        self, rows: list[dict], *, subtask: str, split: str
    ) -> list[InputExample]:
        """Normalize a list of rows for a specific subtask."""
        abbrev = _SUBTASK_ABBREV[subtask]
        examples: list[InputExample] = []

        if self._corpus is None:
            self._corpus = []

        for row in rows:
            raw_id = row["id"]
            example_id = f"faitheval_{abbrev}_{raw_id}"

            # Determine gold answer based on subtask
            if subtask == "unanswerable":
                gold = GoldAnswer(
                    single_answer="unknown",
                    multi_answers=None,
                    unknown_allowed=True,
                )
            elif subtask == "inconsistent":
                gold = GoldAnswer(
                    single_answer="conflict",
                    multi_answers=None,
                    unknown_allowed=True,
                )
            else:  # counterfactual
                gold = GoldAnswer(
                    single_answer=row["answer"],
                    multi_answers=None,
                    unknown_allowed=False,
                )

            # Build context corpus entry
            context_text = row["context"]
            corpus_entry = {
                "passage_id": f"faitheval_ctx_{example_id}",
                "text": context_text,
                "source": f"faitheval_{subtask}",
            }
            self._corpus.append(corpus_entry)
            self._example_contexts[example_id] = context_text

            example = InputExample(
                example_id=example_id,
                question=row["question"],
                task_type="single_answer_qa",
                gold=gold,
                metadata=ExampleMetadata(
                    dataset=self.DATASET_NAME,
                    split=split,
                    extra={
                        "subtask": subtask,
                        "answer_key": row.get("answerKey"),
                        "choices": row.get("choices"),
                    },
                ),
            )
            examples.append(example)

        return examples
