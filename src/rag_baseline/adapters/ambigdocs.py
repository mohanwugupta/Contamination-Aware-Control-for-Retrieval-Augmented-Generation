"""AmbigDocs dataset adapter (PRD 1 §4 Tier 1, §18 Task Group A).

Normalizes AmbigDocs (yoonsanglee/AmbigDocs) into InputExample.
AmbigDocs bundles documents per example, so the adapter extracts
both examples AND a retrieval corpus.

HuggingFace schema:
    qid: int
    ambiguous_entity: str
    question: str
    documents: {
        title: list[str],
        text: list[str],
        pid: list[str],
        answer: list[str]
    }

Splits: train (25,268), validation (3,610), test (7,220)
"""

from __future__ import annotations

from rag_baseline.adapters.base import BaseAdapter, _load_hf_split
from rag_baseline.schemas.input import ExampleMetadata, GoldAnswer, InputExample


class AmbigDocsAdapter(BaseAdapter):
    """Adapter for the AmbigDocs dataset."""

    DATASET_NAME = "ambigdocs"
    HF_DATASET_ID = "yoonsanglee/AmbigDocs"

    def __init__(self) -> None:
        self._corpus: list[dict] | None = None
        self._example_docs: dict[str, list[dict]] = {}

    def load(self, split: str = "validation", **kwargs: object) -> list[InputExample]:
        """Load AmbigDocs from HuggingFace and normalize.

        Args:
            split: Dataset split (train, validation, or test).

        Returns:
            List of normalized InputExample instances.
        """
        ds = _load_hf_split(self.HF_DATASET_ID, split, f"ambigdocs_{split}")
        return self.load_from_dicts(list(ds), split=split)

    def load_from_dicts(
        self, rows: list[dict], split: str = "validation"
    ) -> list[InputExample]:
        """Normalize raw AmbigDocs dicts into InputExamples.

        Args:
            rows: List of dicts matching HuggingFace AmbigDocs schema.
            split: Split name for metadata.

        Returns:
            List of InputExample instances.
        """
        examples: list[InputExample] = []
        corpus_entries: list[dict] = []
        seen_pids: set[str] = set()

        for row in rows:
            qid = row["qid"]
            example_id = f"ambig_{qid}"
            docs = row["documents"]

            # Normalise two possible layouts:
            #   • HF schema (load_dataset):  struct-of-arrays
            #       docs["title"][i], docs["pid"][i], ...
            #   • Raw JSON (load_from_disk):  list-of-structs
            #       docs[i]["title"], docs[i]["pid"], ...
            if isinstance(docs, dict):
                n = len(docs["title"])
                docs_list: list[dict] = [
                    {
                        "pid": docs["pid"][i],
                        "title": docs["title"][i],
                        "text": docs["text"][i],
                        "answer": docs["answer"][i],
                    }
                    for i in range(n)
                ]
            else:
                # list of dicts — already the right shape
                docs_list = list(docs)

            # Extract answers from all documents
            multi_answers = [d["answer"] for d in docs_list]

            # Build per-example document list and global corpus
            example_doc_list: list[dict] = []
            for doc in docs_list:
                pid = doc["pid"]
                doc_entry = {
                    "passage_id": pid,
                    "text": doc["text"],
                    "source": doc["title"],
                }
                example_doc_list.append(doc_entry)

                if pid not in seen_pids:
                    corpus_entries.append(doc_entry)
                    seen_pids.add(pid)

            self._example_docs[example_id] = example_doc_list

            example = InputExample(
                example_id=example_id,
                question=row["question"],
                task_type="multi_answer_qa",
                gold=GoldAnswer(
                    single_answer=None,
                    multi_answers=multi_answers,
                    unknown_allowed=False,
                ),
                metadata=ExampleMetadata(
                    dataset=self.DATASET_NAME,
                    split=split,
                    extra={
                        "ambiguous_entity": row["ambiguous_entity"],
                        "qid": qid,
                    },
                ),
            )
            examples.append(example)

        self._corpus = corpus_entries
        return examples

    def get_corpus(self) -> list[dict] | None:
        """Return the full corpus extracted from all loaded examples.

        Returns:
            List of passage dicts with keys (passage_id, text, source),
            or None if no data has been loaded yet.
        """
        return self._corpus

    def get_example_documents(self, example_id: str) -> list[dict] | None:
        """Get the documents associated with a specific example.

        Args:
            example_id: The example ID (e.g. "ambig_6430").

        Returns:
            List of document dicts for that example, or None if not found.
        """
        return self._example_docs.get(example_id)
