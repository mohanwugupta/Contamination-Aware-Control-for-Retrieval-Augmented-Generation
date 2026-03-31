"""RED tests for RAMDocs adapter (PRD 1 §4 Tier 3).

RAMDocs (HanNight/RAMDocs) — 500 rows, test split only.
Simulates mixed conflict: ambiguity, misinformation, noise.

Fields:
    question: str
    documents: list[{text: str, type: str (correct|misinfo|noise), answer: str}]
    disambig_entity: list[str]
    gold_answers: list[str]
    wrong_answers: list[str]
"""

import pytest


# ---------------------------------------------------------------------------
# Sample data matching HuggingFace RAMDocs schema
# ---------------------------------------------------------------------------

RAMDOCS_ROWS = [
    {
        "question": "When was the University of Georgia founded?",
        "documents": [
            {
                "text": "The University of Georgia was chartered in 1785.",
                "type": "correct",
                "answer": "1785",
            },
            {
                "text": "The University of Georgia was established in 1801.",
                "type": "correct",
                "answer": "1801",
            },
            {
                "text": "The University of Georgia was founded in 1950 as a modern institution.",
                "type": "misinfo",
                "answer": "1950",
            },
            {
                "text": "Georgia is known for its peach orchards and warm climate.",
                "type": "noise",
                "answer": "unknown",
            },
        ],
        "disambig_entity": [
            "University of Georgia (chartered)",
            "University of Georgia (opened)",
        ],
        "gold_answers": ["1785", "1801"],
        "wrong_answers": ["1950"],
    },
    {
        "question": "Who is the president of Springfield?",
        "documents": [
            {
                "text": "Mayor Joe Quimby is the current mayor of Springfield.",
                "type": "correct",
                "answer": "Joe Quimby",
            },
            {
                "text": "Springfield's president is Bob Smith according to recent polls.",
                "type": "misinfo",
                "answer": "Bob Smith",
            },
        ],
        "disambig_entity": ["Springfield (The Simpsons)"],
        "gold_answers": ["Joe Quimby"],
        "wrong_answers": ["Bob Smith"],
    },
    {
        "question": "What is the population of Greenville?",
        "documents": [
            {
                "text": "Greenville, South Carolina has a population of about 72,000.",
                "type": "correct",
                "answer": "72,000",
            },
            {
                "text": "Greenville is a lovely city with many parks.",
                "type": "noise",
                "answer": "unknown",
            },
        ],
        "disambig_entity": ["Greenville, South Carolina"],
        "gold_answers": ["72,000"],
        "wrong_answers": [],
    },
]


# ===================================================================
# RAMDocsAdapter: construction
# ===================================================================


class TestRAMDocsAdapterConstruction:
    """Test adapter initialization."""

    def test_dataset_name(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        assert adapter.DATASET_NAME == "ramdocs"

    def test_is_base_adapter(self):
        from rag_baseline.adapters.base import BaseAdapter
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        assert isinstance(adapter, BaseAdapter)


# ===================================================================
# RAMDocsAdapter: load_from_dicts normalization
# ===================================================================


class TestRAMDocsLoadFromDicts:
    """Test normalization of RAMDocs examples."""

    def test_load_from_dicts_returns_correct_count(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert len(examples) == 3

    def test_example_id_format(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].example_id == "ramdocs_0"
        assert examples[1].example_id == "ramdocs_1"

    def test_question_preserved(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].question == "When was the University of Georgia founded?"

    def test_task_type_is_multi_answer(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].task_type == "multi_answer_qa"

    def test_gold_multi_answers(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].gold.multi_answers == ["1785", "1801"]

    def test_gold_unknown_allowed(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        # RAMDocs examples may have unusable docs → unknown should be allowed
        assert examples[0].gold.unknown_allowed is True

    def test_metadata_dataset(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].metadata.dataset == "ramdocs"

    def test_metadata_split(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].metadata.split == "test"

    def test_metadata_extra_disambig_entity(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].metadata.extra["disambig_entity"] == [
            "University of Georgia (chartered)",
            "University of Georgia (opened)",
        ]

    def test_metadata_extra_wrong_answers(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        assert examples[0].metadata.extra["wrong_answers"] == ["1950"]

    def test_single_gold_answer_example(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        examples = adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        # Third example has only one gold answer
        assert examples[2].gold.multi_answers == ["72,000"]


# ===================================================================
# RAMDocsAdapter: corpus
# ===================================================================


class TestRAMDocsCorpus:
    """Test corpus extraction from RAMDocs documents."""

    def test_corpus_is_none_before_load(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        assert adapter.get_corpus() is None

    def test_corpus_after_load(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        corpus = adapter.get_corpus()
        assert corpus is not None
        # 4 + 2 + 2 = 8 total documents
        assert len(corpus) == 8

    def test_corpus_entries_have_required_keys(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        corpus = adapter.get_corpus()
        entry = corpus[0]
        assert "passage_id" in entry
        assert "text" in entry
        assert "source" in entry

    def test_corpus_entries_have_doc_type(self):
        """Each corpus entry should include the document type for analysis."""
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        corpus = adapter.get_corpus()
        entry = corpus[0]
        assert "doc_type" in entry
        assert entry["doc_type"] in ("correct", "misinfo", "noise")

    def test_get_example_documents(self):
        """Adapter should expose per-example document retrieval."""
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        docs = adapter.get_example_documents("ramdocs_0")
        assert docs is not None
        assert len(docs) == 4  # first example has 4 documents

    def test_get_example_documents_unknown_id(self):
        from rag_baseline.adapters.ramdocs import RAMDocsAdapter

        adapter = RAMDocsAdapter()
        adapter.load_from_dicts(RAMDOCS_ROWS, split="test")
        docs = adapter.get_example_documents("nonexistent")
        assert docs is None


# ===================================================================
# Factory registration
# ===================================================================


class TestRAMDocsFactory:
    """Test that the adapter factory recognizes RAMDocs."""

    def test_create_adapter_ramdocs(self):
        from rag_baseline.adapters import create_adapter

        adapter = create_adapter("ramdocs")
        assert type(adapter).__name__ == "RAMDocsAdapter"

    def test_create_adapter_ram_docs_alias(self):
        from rag_baseline.adapters import create_adapter

        adapter = create_adapter("ram_docs")
        assert type(adapter).__name__ == "RAMDocsAdapter"
