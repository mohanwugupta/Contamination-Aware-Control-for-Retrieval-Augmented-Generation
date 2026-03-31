"""RED tests for the reranker layer (PRD 1 §9.3).

Reranker must be easy to toggle on/off and produce RerankedPassage outputs.
"""

import pytest


class TestRerankerInterface:
    """Base reranker contract tests."""

    def test_base_reranker_is_abstract(self):
        from rag_baseline.reranking.base import BaseReranker
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_base_reranker_has_rerank_method(self):
        from rag_baseline.reranking.base import BaseReranker
        assert hasattr(BaseReranker, "rerank")


class TestCrossEncoderReranker:
    """Tests for cross-encoder reranker implementation."""

    @pytest.fixture
    def sample_passages(self):
        from rag_baseline.schemas.retrieval import RetrievedPassage
        return [
            RetrievedPassage(
                passage_id=f"p{i}", text=text, source=f"doc_{i}",
                retrieval_score=10.0 - i, rank=i,
            )
            for i, text in enumerate([
                "Michael Jordan was born in 1963 in Brooklyn.",
                "The capital of France is Paris.",
                "Michael Jordan is a retired basketball player.",
            ], start=1)
        ]

    @pytest.mark.slow
    def test_cross_encoder_reranker_creates(self):
        from rag_baseline.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
        assert reranker is not None

    @pytest.mark.slow
    def test_cross_encoder_reranker_produces_rerank_output(self, sample_passages):
        from rag_baseline.reranking.cross_encoder import CrossEncoderReranker
        from rag_baseline.schemas.rerank import RerankOutput

        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
        result = reranker.rerank(
            query="When was Michael Jordan born?",
            passages=sample_passages,
            example_id="ex_001",
        )
        assert isinstance(result, RerankOutput)
        assert len(result.reranked_passages) == 3
        # Must have rerank scores
        for p in result.reranked_passages:
            assert p.rerank_score is not None
            assert p.rank_after_rerank >= 1

    @pytest.mark.slow
    def test_reranked_passages_sorted_by_rerank_score(self, sample_passages):
        from rag_baseline.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
        result = reranker.rerank(
            query="When was Michael Jordan born?",
            passages=sample_passages,
            example_id="ex_001",
        )
        scores = [p.rerank_score for p in result.reranked_passages]
        assert scores == sorted(scores, reverse=True)


class TestRerankerFactory:
    """Tests for reranker factory."""

    @pytest.mark.slow
    def test_factory_creates_cross_encoder(self):
        from rag_baseline.reranking import create_reranker
        from rag_baseline.reranking.cross_encoder import CrossEncoderReranker

        reranker = create_reranker(reranker_model="BAAI/bge-reranker-v2-m3")
        assert isinstance(reranker, CrossEncoderReranker)

    def test_factory_returns_none_when_disabled(self):
        from rag_baseline.reranking import create_reranker

        reranker = create_reranker(reranker_model=None)
        assert reranker is None
