"""NQ-Open dataset adapter (PRD 1 §4 Tier 0, §18 Task Group A).

Normalizes NQ-Open (google-research-datasets/nq_open) into InputExample.
NQ-Open is open-domain single-answer factual QA with no bundled corpus.

HuggingFace schema:
    question: str
    answer: list[str]   # list of acceptable answer strings

Splits: train (87,925), validation (3,610)
"""

from __future__ import annotations

from rag_baseline.adapters.base import BaseAdapter
from rag_baseline.schemas.input import ExampleMetadata, GoldAnswer, InputExample  # noqa: F401


class NQOpenAdapter(BaseAdapter):
    """Adapter for the NQ-Open dataset."""

    DATASET_NAME = "nq_open"
    HF_DATASET_ID = "google-research-datasets/nq_open"

    def load(self, split: str = "validation", **kwargs: object) -> list[InputExample]:
        """Load NQ-Open from HuggingFace and normalize.

        Args:
            split: Dataset split (train or validation).

        Returns:
            List of normalized InputExample instances.
        """
        from datasets import load_dataset

        ds = load_dataset(self.HF_DATASET_ID, split=split)
        return self.load_from_dicts(list(ds), split=split)

    def load_from_dicts(
        self, rows: list[dict], split: str = "validation"
    ) -> list[InputExample]:
        """Normalize raw NQ-Open dicts into InputExamples.

        Args:
            rows: List of dicts with 'question' and 'answer' keys.
            split: Split name for metadata.

        Returns:
            List of InputExample instances.
        """
        examples: list[InputExample] = []

        for idx, row in enumerate(rows):
            answers = row["answer"]
            single_answer = answers[0] if answers else ""

            # For NQ, all entries in `answer` are acceptable alternative forms
            # of the same answer. Store as single_answer with alternatives
            # available via multi_answers only when there are multiple.
            multi_answers: list[str] | None = None
            if len(answers) > 1:
                multi_answers = list(answers)

            example = InputExample(
                example_id=f"nq_{idx:05d}",
                question=row["question"],
                task_type="single_answer_qa",
                gold=GoldAnswer(
                    single_answer=single_answer,
                    multi_answers=multi_answers,
                    unknown_allowed=False,
                ),
                metadata=ExampleMetadata(
                    dataset=self.DATASET_NAME,
                    split=split,
                ),
            )
            examples.append(example)

        return examples

    def get_corpus(self) -> list[dict] | None:
        """NQ-Open is open-domain; no bundled corpus.

        Returns:
            None — corpus must be provided externally for retrieval.
        """
        return None
