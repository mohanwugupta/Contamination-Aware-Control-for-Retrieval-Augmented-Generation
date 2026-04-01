"""RAMDocs dataset adapter (PRD 1 §4 Tier 3).

Normalizes RAMDocs (HanNight/RAMDocs) into InputExample.
RAMDocs simulates mixed conflict: ambiguity, misinformation, noise.

HuggingFace schema:
    question: str
    documents: list[{text: str, type: str (correct|misinfo|noise), answer: str}]
    disambig_entity: list[str]
    gold_answers: list[str]
    wrong_answers: list[str]

Splits: test (500 rows only)
"""

from __future__ import annotations

from rag_baseline.adapters.base import BaseAdapter, _load_hf_split
from rag_baseline.schemas.input import ExampleMetadata, GoldAnswer, InputExample


class RAMDocsAdapter(BaseAdapter):
    """Adapter for the RAMDocs dataset (mixed-conflict QA)."""

    DATASET_NAME = "ramdocs"
    HF_DATASET_ID = "HanNight/RAMDocs"

    def __init__(self) -> None:
        self._corpus: list[dict] | None = None
        self._example_docs: dict[str, list[dict]] = {}

    def load(self, split: str = "test", **kwargs: object) -> list[InputExample]:
        """Load RAMDocs from HuggingFace and normalize.

        RAMDocs only has a 'test' split.

        Args:
            split: Dataset split (RAMDocs only has 'test').

        Returns:
            List of normalized InputExample instances.
        """
        ds = _load_hf_split(self.HF_DATASET_ID, split, f"ramdocs_{split}")
        return self.load_from_dicts(list(ds), split=split)

    def load_from_dicts(
        self, rows: list[dict], split: str = "test"
    ) -> list[InputExample]:
        """Normalize raw RAMDocs dicts into InputExamples.

        Args:
            rows: List of dicts matching HuggingFace RAMDocs schema.
            split: Split name for metadata.

        Returns:
            List of InputExample instances.
        """
        examples: list[InputExample] = []
        corpus_entries: list[dict] = []

        for idx, row in enumerate(rows):
            example_id = f"ramdocs_{idx}"
            documents = row["documents"]

            # Build per-example document list and global corpus
            example_doc_list: list[dict] = []
            for doc_idx, doc in enumerate(documents):
                passage_id = f"{example_id}_doc_{doc_idx}"
                doc_entry = {
                    "passage_id": passage_id,
                    "text": doc["text"],
                    "source": f"ramdocs_{doc['type']}",
                    "doc_type": doc["type"],
                }
                example_doc_list.append(doc_entry)
                corpus_entries.append(doc_entry)

            self._example_docs[example_id] = example_doc_list

            example = InputExample(
                example_id=example_id,
                question=row["question"],
                task_type="multi_answer_qa",
                gold=GoldAnswer(
                    single_answer=None,
                    multi_answers=row["gold_answers"],
                    unknown_allowed=True,
                ),
                metadata=ExampleMetadata(
                    dataset=self.DATASET_NAME,
                    split=split,
                    extra={
                        "disambig_entity": row.get("disambig_entity", []),
                        "wrong_answers": row.get("wrong_answers", []),
                    },
                ),
            )
            examples.append(example)

        self._corpus = corpus_entries
        return examples

    def get_corpus(self) -> list[dict] | None:
        """Return the full corpus extracted from all loaded examples.

        Returns:
            List of passage dicts with keys (passage_id, text, source, doc_type),
            or None if no data has been loaded yet.
        """
        return self._corpus

    def get_example_documents(self, example_id: str) -> list[dict] | None:
        """Get the documents associated with a specific example.

        Args:
            example_id: The example ID (e.g. "ramdocs_0").

        Returns:
            List of document dicts for that example, or None if not found.
        """
        return self._example_docs.get(example_id)
