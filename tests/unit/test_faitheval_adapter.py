"""RED tests for FaithEval adapter (PRD 1 §4 Tier 2).

FaithEval (Salesforce) evaluates contextual faithfulness across three tasks:
- unanswerable: model should say "unknown"
- inconsistent: model should say "conflict"
- counterfactual: context presents wrong info, model should be faithful to context

Three separate HuggingFace datasets:
- Salesforce/FaithEval-unanswerable-v1.0  (2,492 rows, test only)
- Salesforce/FaithEval-inconsistent-v1.0   (1,500 rows, test only)
- Salesforce/FaithEval-counterfactual-v1.0 (1,000 rows, test only)

Fields: id, question, answer, answerKey, choices ({"label": [...], "text": [...]}), context
"""

import pytest


# ---------------------------------------------------------------------------
# Sample data matching HuggingFace FaithEval schema
# ---------------------------------------------------------------------------

UNANSWERABLE_ROWS = [
    {
        "id": "un_001",
        "question": "What is the capital of Mars?",
        "answer": "unknown",
        "answerKey": "A",
        "choices": {"label": ["A", "B"], "text": ["unknown", "Olympus City"]},
        "context": "Mars is the fourth planet from the Sun. No capital city has been established.",
    },
    {
        "id": "un_002",
        "question": "Who wrote the poem about invisible ink?",
        "answer": "unknown",
        "answerKey": "B",
        "choices": {"label": ["A", "B"], "text": ["Shakespeare", "unknown"]},
        "context": "Invisible ink has been used throughout history for secret communications.",
    },
]

INCONSISTENT_ROWS = [
    {
        "id": "ic_001",
        "question": "What year was the Great Library destroyed?",
        "answer": "conflict",
        "answerKey": "C",
        "choices": {
            "label": ["A", "B", "C"],
            "text": ["48 BC", "642 AD", "conflict"],
        },
        "context": (
            "Source 1: The Great Library was destroyed in 48 BC by Caesar's fire. "
            "Source 2: The Great Library was destroyed in 642 AD during the Muslim conquest."
        ),
    },
]

COUNTERFACTUAL_ROWS = [
    {
        "id": "cf_001",
        "question": "At which temperature does water freeze?",
        "answer": "100 degrees Celsius",
        "answerKey": "C",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": [
                "0 degrees Celsius",
                "32 degrees Celsius",
                "100 degrees Celsius",
                "212 degrees Celsius",
            ],
        },
        "context": (
            "Recent studies show water freezes at 100 degrees Celsius under "
            "specific high-pressure conditions."
        ),
    },
    {
        "id": "cf_002",
        "question": "Which planet is closest to the Sun?",
        "answer": "Venus",
        "answerKey": "B",
        "choices": {
            "label": ["A", "B", "C"],
            "text": ["Mercury", "Venus", "Earth"],
        },
        "context": "Venus is the closest planet to the Sun due to its orbital resonance.",
    },
]


# ===================================================================
# FaithEvalAdapter: construction and subtask selection
# ===================================================================


class TestFaithEvalAdapterConstruction:
    """Test adapter initialization with different subtask modes."""

    def test_default_subtask_is_all(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter()
        assert adapter.subtask == "all"

    def test_subtask_unanswerable(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        assert adapter.subtask == "unanswerable"

    def test_subtask_inconsistent(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="inconsistent")
        assert adapter.subtask == "inconsistent"

    def test_subtask_counterfactual(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        assert adapter.subtask == "counterfactual"

    def test_invalid_subtask_raises(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        with pytest.raises(ValueError, match="subtask"):
            FaithEvalAdapter(subtask="garbage")

    def test_dataset_name(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter()
        assert adapter.DATASET_NAME == "faitheval"

    def test_is_base_adapter(self):
        from rag_baseline.adapters.base import BaseAdapter
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter()
        assert isinstance(adapter, BaseAdapter)


# ===================================================================
# FaithEvalAdapter: unanswerable task normalization
# ===================================================================


class TestFaithEvalUnanswerable:
    """Test normalization of unanswerable-context examples."""

    def test_load_from_dicts_returns_examples(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert len(examples) == 2

    def test_example_id_includes_subtask(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert examples[0].example_id == "faitheval_un_un_001"

    def test_task_type_is_single_answer(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert examples[0].task_type == "single_answer_qa"

    def test_gold_answer_is_unknown(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert examples[0].gold.single_answer == "unknown"

    def test_unknown_allowed_true(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert examples[0].gold.unknown_allowed is True

    def test_metadata_dataset(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert examples[0].metadata.dataset == "faitheval"

    def test_metadata_split(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert examples[0].metadata.split == "test"

    def test_metadata_extra_contains_subtask(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert examples[0].metadata.extra["subtask"] == "unanswerable"

    def test_metadata_extra_contains_choices(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="unanswerable")
        examples = adapter.load_from_dicts(UNANSWERABLE_ROWS, split="test")
        assert "choices" in examples[0].metadata.extra


# ===================================================================
# FaithEvalAdapter: inconsistent task normalization
# ===================================================================


class TestFaithEvalInconsistent:
    """Test normalization of inconsistent-context examples."""

    def test_load_from_dicts_returns_examples(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="inconsistent")
        examples = adapter.load_from_dicts(INCONSISTENT_ROWS, split="test")
        assert len(examples) == 1

    def test_gold_answer_is_conflict(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="inconsistent")
        examples = adapter.load_from_dicts(INCONSISTENT_ROWS, split="test")
        assert examples[0].gold.single_answer == "conflict"

    def test_unknown_allowed_true_for_inconsistent(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="inconsistent")
        examples = adapter.load_from_dicts(INCONSISTENT_ROWS, split="test")
        assert examples[0].gold.unknown_allowed is True

    def test_example_id_includes_ic_prefix(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="inconsistent")
        examples = adapter.load_from_dicts(INCONSISTENT_ROWS, split="test")
        assert examples[0].example_id == "faitheval_ic_ic_001"


# ===================================================================
# FaithEvalAdapter: counterfactual task normalization
# ===================================================================


class TestFaithEvalCounterfactual:
    """Test normalization of counterfactual-context examples."""

    def test_load_from_dicts_returns_examples(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        examples = adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        assert len(examples) == 2

    def test_gold_is_counterfactual_answer(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        examples = adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        # The gold is the (wrong) answer faithful to the counterfactual context
        assert examples[0].gold.single_answer == "100 degrees Celsius"

    def test_unknown_allowed_false_for_counterfactual(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        examples = adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        assert examples[0].gold.unknown_allowed is False

    def test_example_id_includes_cf_prefix(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        examples = adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        assert examples[0].example_id == "faitheval_cf_cf_001"


# ===================================================================
# FaithEvalAdapter: corpus (bundled context as passages)
# ===================================================================


class TestFaithEvalCorpus:
    """Test that context is exposed as a bundled corpus."""

    def test_corpus_is_none_before_load(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        assert adapter.get_corpus() is None

    def test_corpus_after_load(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        corpus = adapter.get_corpus()
        assert corpus is not None
        assert len(corpus) == 2

    def test_corpus_entries_have_required_keys(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        corpus = adapter.get_corpus()
        entry = corpus[0]
        assert "passage_id" in entry
        assert "text" in entry
        assert "source" in entry

    def test_corpus_text_matches_context(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        corpus = adapter.get_corpus()
        assert COUNTERFACTUAL_ROWS[0]["context"] in corpus[0]["text"]

    def test_get_example_context(self):
        """Adapter should expose per-example context retrieval."""
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="counterfactual")
        examples = adapter.load_from_dicts(COUNTERFACTUAL_ROWS, split="test")
        ctx = adapter.get_example_context(examples[0].example_id)
        assert ctx is not None
        assert COUNTERFACTUAL_ROWS[0]["context"] in ctx


# ===================================================================
# FaithEvalAdapter: "all" mode combines all subtasks
# ===================================================================


class TestFaithEvalAllMode:
    """Test that subtask='all' combines multiple subtask dicts."""

    def test_load_all_from_dicts(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="all")
        examples = adapter.load_all_from_dicts(
            unanswerable=UNANSWERABLE_ROWS,
            inconsistent=INCONSISTENT_ROWS,
            counterfactual=COUNTERFACTUAL_ROWS,
            split="test",
        )
        # 2 + 1 + 2 = 5
        assert len(examples) == 5

    def test_all_mode_preserves_subtask_in_metadata(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="all")
        examples = adapter.load_all_from_dicts(
            unanswerable=UNANSWERABLE_ROWS,
            inconsistent=INCONSISTENT_ROWS,
            counterfactual=COUNTERFACTUAL_ROWS,
            split="test",
        )
        subtasks = {ex.metadata.extra["subtask"] for ex in examples}
        assert subtasks == {"unanswerable", "inconsistent", "counterfactual"}

    def test_all_mode_corpus_spans_all_subtasks(self):
        from rag_baseline.adapters.faitheval import FaithEvalAdapter

        adapter = FaithEvalAdapter(subtask="all")
        adapter.load_all_from_dicts(
            unanswerable=UNANSWERABLE_ROWS,
            inconsistent=INCONSISTENT_ROWS,
            counterfactual=COUNTERFACTUAL_ROWS,
            split="test",
        )
        corpus = adapter.get_corpus()
        assert corpus is not None
        # 2 + 1 + 2 = 5 passages
        assert len(corpus) == 5


# ===================================================================
# Factory registration
# ===================================================================


class TestFaithEvalFactory:
    """Test that the adapter factory recognizes FaithEval."""

    def test_create_adapter_faitheval(self):
        from rag_baseline.adapters import create_adapter

        adapter = create_adapter("faitheval")
        assert type(adapter).__name__ == "FaithEvalAdapter"

    def test_create_adapter_faith_eval_alias(self):
        from rag_baseline.adapters import create_adapter

        adapter = create_adapter("faith_eval")
        assert type(adapter).__name__ == "FaithEvalAdapter"
