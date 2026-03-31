"""RED tests for context assembly layer (PRD 1 §9.4).

Context assembly must support:
- full retrieved set
- reranked full set
- reduced-context top-1
- reduced-context top-2
Deterministic formatting is required.
"""

import pytest


class TestContextAssembly:
    """Tests for the context assembly module."""

    @pytest.fixture
    def sample_passages(self):
        from rag_baseline.schemas.retrieval import RetrievedPassage
        return [
            RetrievedPassage(
                passage_id=f"p{i}", text=f"Passage {i} content here.",
                source=f"doc_{i}", retrieval_score=10.0 - i, rank=i,
            )
            for i in range(1, 6)
        ]

    @pytest.fixture
    def sample_reranked_passages(self):
        from rag_baseline.schemas.rerank import RerankedPassage
        return [
            RerankedPassage(
                passage_id="p3", text="Passage 3 content here.", source="doc_3",
                retrieval_score=8.0, rerank_score=0.95, rank_after_rerank=1,
            ),
            RerankedPassage(
                passage_id="p1", text="Passage 1 content here.", source="doc_1",
                retrieval_score=10.0, rerank_score=0.80, rank_after_rerank=2,
            ),
            RerankedPassage(
                passage_id="p5", text="Passage 5 content here.", source="doc_5",
                retrieval_score=6.0, rerank_score=0.60, rank_after_rerank=3,
            ),
        ]

    def test_assemble_full_context(self, sample_passages):
        from rag_baseline.context.assembly import assemble_context

        result = assemble_context(
            passages=sample_passages,
            strategy="full",
            max_passages=None,
        )
        assert len(result.passages) == 5
        assert result.passage_ids == ["p1", "p2", "p3", "p4", "p5"]

    def test_assemble_reduced_top_1(self, sample_passages):
        from rag_baseline.context.assembly import assemble_context

        result = assemble_context(
            passages=sample_passages,
            strategy="reduced",
            max_passages=1,
        )
        assert len(result.passages) == 1
        assert result.passage_ids == ["p1"]

    def test_assemble_reduced_top_2(self, sample_passages):
        from rag_baseline.context.assembly import assemble_context

        result = assemble_context(
            passages=sample_passages,
            strategy="reduced",
            max_passages=2,
        )
        assert len(result.passages) == 2
        assert result.passage_ids == ["p1", "p2"]

    def test_assemble_from_reranked(self, sample_reranked_passages):
        from rag_baseline.context.assembly import assemble_context

        result = assemble_context(
            passages=sample_reranked_passages,
            strategy="full",
            max_passages=None,
        )
        # Should use rerank order
        assert result.passage_ids == ["p3", "p1", "p5"]

    def test_assemble_reduced_from_reranked(self, sample_reranked_passages):
        from rag_baseline.context.assembly import assemble_context

        result = assemble_context(
            passages=sample_reranked_passages,
            strategy="reduced",
            max_passages=2,
        )
        assert result.passage_ids == ["p3", "p1"]

    def test_deterministic_formatting(self, sample_passages):
        """Same input must always produce the same formatted text."""
        from rag_baseline.context.assembly import assemble_context

        r1 = assemble_context(passages=sample_passages, strategy="full", max_passages=None)
        r2 = assemble_context(passages=sample_passages, strategy="full", max_passages=None)
        assert r1.formatted_text == r2.formatted_text

    def test_formatted_text_contains_all_passages(self, sample_passages):
        from rag_baseline.context.assembly import assemble_context

        result = assemble_context(passages=sample_passages, strategy="full", max_passages=None)
        for p in sample_passages:
            assert p.text in result.formatted_text

    def test_empty_context_for_none_strategy(self):
        from rag_baseline.context.assembly import assemble_context

        result = assemble_context(passages=[], strategy="none", max_passages=None)
        assert len(result.passages) == 0
        assert result.formatted_text == ""
