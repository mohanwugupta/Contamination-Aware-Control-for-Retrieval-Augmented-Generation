"""RED tests for dataset adapters (PRD 1 §9.1, §18 Task Group A).

Tests the base adapter interface and concrete adapters for:
- NQ-Open (Tier 0, single-answer factual QA)
- AmbigDocs (Tier 1, multi-answer ambiguity QA)
"""

import pytest


# ---------------------------------------------------------------------------
# Base adapter interface
# ---------------------------------------------------------------------------

class TestBaseAdapter:
    """Base adapter must be abstract with load() and get_corpus() methods."""

    def test_base_adapter_is_abstract(self):
        from rag_baseline.adapters.base import BaseAdapter
        with pytest.raises(TypeError):
            BaseAdapter()

    def test_base_adapter_has_load_method(self):
        from rag_baseline.adapters.base import BaseAdapter
        assert hasattr(BaseAdapter, "load")

    def test_base_adapter_has_get_corpus_method(self):
        from rag_baseline.adapters.base import BaseAdapter
        assert hasattr(BaseAdapter, "get_corpus")


# ---------------------------------------------------------------------------
# NQ-Open adapter
# ---------------------------------------------------------------------------

class TestNQOpenAdapter:
    """NQ-Open adapter normalizes HuggingFace NQ-Open into InputExample."""

    @pytest.fixture
    def raw_nq_examples(self):
        """Simulates raw HuggingFace NQ-Open rows."""
        return [
            {
                "question": "When was Michael Jordan born?",
                "answer": ["1963"],
            },
            {
                "question": "Who starred in the movie deep water horizon?",
                "answer": [
                    "Kurt Russell",
                    "Dylan O'Brien",
                    "John Malkovich",
                    "Kate Hudson",
                    "Mark Wahlberg",
                ],
            },
            {
                "question": "What is the capital of France?",
                "answer": ["Paris"],
            },
        ]

    def test_nq_adapter_creates(self):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        assert adapter is not None

    def test_nq_adapter_load_from_dicts(self, raw_nq_examples):
        from rag_baseline.adapters.nq_open import NQOpenAdapter
        from rag_baseline.schemas.input import InputExample

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(raw_nq_examples, split="validation")

        assert len(examples) == 3
        assert all(isinstance(e, InputExample) for e in examples)

    def test_nq_adapter_sets_task_type(self, raw_nq_examples):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(raw_nq_examples, split="validation")

        for ex in examples:
            assert ex.task_type == "single_answer_qa"

    def test_nq_adapter_sets_gold_single_answer(self, raw_nq_examples):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(raw_nq_examples, split="validation")

        # First answer should be set as single_answer
        assert examples[0].gold.single_answer == "1963"
        assert examples[2].gold.single_answer == "Paris"

    def test_nq_adapter_stores_all_acceptable_answers(self, raw_nq_examples):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(raw_nq_examples, split="validation")

        # Multi-answer NQ examples store all acceptable answers
        ex_multi = examples[1]
        assert ex_multi.gold.single_answer == "Kurt Russell"
        assert ex_multi.gold.multi_answers is not None
        assert "Mark Wahlberg" in ex_multi.gold.multi_answers

    def test_nq_adapter_sets_metadata(self, raw_nq_examples):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(raw_nq_examples, split="validation")

        for ex in examples:
            assert ex.metadata.dataset == "nq_open"
            assert ex.metadata.split == "validation"

    def test_nq_adapter_assigns_example_ids(self, raw_nq_examples):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(raw_nq_examples, split="validation")

        ids = [ex.example_id for ex in examples]
        # All IDs should be unique
        assert len(set(ids)) == len(ids)
        # IDs should start with nq_ prefix
        assert all(eid.startswith("nq_") for eid in ids)

    def test_nq_adapter_get_corpus_returns_none(self):
        """NQ-Open is open-domain; no bundled corpus."""
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        assert adapter.get_corpus() is None

    def test_nq_adapter_unknown_not_allowed(self, raw_nq_examples):
        from rag_baseline.adapters.nq_open import NQOpenAdapter

        adapter = NQOpenAdapter()
        examples = adapter.load_from_dicts(raw_nq_examples, split="validation")

        for ex in examples:
            assert ex.gold.unknown_allowed is False


# ---------------------------------------------------------------------------
# AmbigDocs adapter
# ---------------------------------------------------------------------------

class TestAmbigDocsAdapter:
    """AmbigDocs adapter normalizes HuggingFace AmbigDocs into InputExample."""

    @pytest.fixture
    def raw_ambig_examples(self):
        """Simulates raw HuggingFace AmbigDocs rows."""
        return [
            {
                "qid": 6430,
                "ambiguous_entity": "Wang Heun",
                "question": "Who was the parent of Wang Heun?",
                "documents": {
                    "title": [
                        "Chungmok of Goryeo",
                        "Myeongjong of Goryeo",
                    ],
                    "text": [
                        "Chungmok of Goryeo King Chungmok was the 29th king.",
                        "Myeongjong of Goryeo was the 19th king.",
                    ],
                    "pid": ["doc_001", "doc_002"],
                    "answer": ["King Chunghye", "King Uijong"],
                },
            },
            {
                "qid": 9601,
                "ambiguous_entity": "Port Washington",
                "question": "What is the population density of Port Washington?",
                "documents": {
                    "title": [
                        "Port Washington, Ohio",
                        "Port Washington, New York",
                        "Port Washington, Wisconsin",
                    ],
                    "text": [
                        "Port Washington Ohio has density 1091.",
                        "Port Washington NY has density 5002.",
                        "Port Washington WI has density 1540.",
                    ],
                    "pid": ["doc_003", "doc_004", "doc_005"],
                    "answer": ["1,091.2", "5,002.1", "1,540.3"],
                },
            },
        ]

    def test_ambigdocs_adapter_creates(self):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        assert adapter is not None

    def test_ambigdocs_adapter_load_from_dicts(self, raw_ambig_examples):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter
        from rag_baseline.schemas.input import InputExample

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        assert len(examples) == 2
        assert all(isinstance(e, InputExample) for e in examples)

    def test_ambigdocs_adapter_sets_task_type(self, raw_ambig_examples):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        for ex in examples:
            assert ex.task_type == "multi_answer_qa"

    def test_ambigdocs_adapter_sets_multi_answers(self, raw_ambig_examples):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        # First example should have 2 answers
        assert examples[0].gold.multi_answers is not None
        assert len(examples[0].gold.multi_answers) == 2
        assert "King Chunghye" in examples[0].gold.multi_answers
        assert "King Uijong" in examples[0].gold.multi_answers

        # Second example should have 3 answers
        assert examples[1].gold.multi_answers is not None
        assert len(examples[1].gold.multi_answers) == 3

    def test_ambigdocs_adapter_sets_no_single_answer(self, raw_ambig_examples):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        for ex in examples:
            assert ex.gold.single_answer is None

    def test_ambigdocs_adapter_sets_metadata(self, raw_ambig_examples):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        for ex in examples:
            assert ex.metadata.dataset == "ambigdocs"
            assert ex.metadata.split == "validation"
            assert ex.metadata.extra is not None
            assert "ambiguous_entity" in ex.metadata.extra

        assert examples[0].metadata.extra["ambiguous_entity"] == "Wang Heun"

    def test_ambigdocs_adapter_assigns_example_ids(self, raw_ambig_examples):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        ids = [ex.example_id for ex in examples]
        assert len(set(ids)) == len(ids)
        assert all(eid.startswith("ambig_") for eid in ids)

    def test_ambigdocs_adapter_extracts_corpus(self, raw_ambig_examples):
        """AmbigDocs bundles documents per example → adapter extracts corpus."""
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        adapter.load_from_dicts(raw_ambig_examples, split="validation")
        corpus = adapter.get_corpus()

        assert corpus is not None
        assert len(corpus) == 5  # 2 + 3 documents total

        # Each corpus entry should have passage_id, text, source
        for doc in corpus:
            assert "passage_id" in doc
            assert "text" in doc
            assert "source" in doc

    def test_ambigdocs_adapter_corpus_has_unique_ids(self, raw_ambig_examples):
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        adapter.load_from_dicts(raw_ambig_examples, split="validation")
        corpus = adapter.get_corpus()

        ids = [d["passage_id"] for d in corpus]
        assert len(set(ids)) == len(ids)

    def test_ambigdocs_adapter_get_example_documents(self, raw_ambig_examples):
        """Should be able to get documents for a specific example."""
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        docs = adapter.get_example_documents(examples[0].example_id)
        assert docs is not None
        assert len(docs) == 2

    def test_ambigdocs_adapter_unknown_allowed(self, raw_ambig_examples):
        """AmbigDocs does not have unknown/unanswerable examples."""
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter

        adapter = AmbigDocsAdapter()
        examples = adapter.load_from_dicts(raw_ambig_examples, split="validation")

        for ex in examples:
            assert ex.gold.unknown_allowed is False


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

class TestAdapterFactory:
    """Factory creates the right adapter by dataset name."""

    def test_factory_creates_nq(self):
        from rag_baseline.adapters import create_adapter

        adapter = create_adapter("nq_open")
        from rag_baseline.adapters.nq_open import NQOpenAdapter
        assert isinstance(adapter, NQOpenAdapter)

    def test_factory_creates_ambigdocs(self):
        from rag_baseline.adapters import create_adapter

        adapter = create_adapter("ambigdocs")
        from rag_baseline.adapters.ambigdocs import AmbigDocsAdapter
        assert isinstance(adapter, AmbigDocsAdapter)

    def test_factory_rejects_unknown(self):
        from rag_baseline.adapters import create_adapter

        with pytest.raises(ValueError):
            create_adapter("unknown_dataset")
