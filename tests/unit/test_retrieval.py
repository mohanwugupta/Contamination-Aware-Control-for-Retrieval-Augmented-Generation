"""RED tests for the modular retrieval layer (PRD 1 §9.2).

Tests the abstract base retriever and concrete implementations:
- DenseRetriever (FAISS-based)
- SparseRetriever (BM25-based)
- HybridRetriever (dense + sparse fusion)

All retrievers must conform to the same interface and produce
RetrievalOutput schema-compliant results.
"""

import pytest


# ---------------------------------------------------------------------------
# Base retriever interface contract
# ---------------------------------------------------------------------------

class TestRetrieverInterface:
    """All retriever implementations must satisfy this contract."""

    def test_base_retriever_is_abstract(self):
        from rag_baseline.retrieval.base import BaseRetriever

        with pytest.raises(TypeError):
            BaseRetriever()  # type: ignore[abstract]

    def test_base_retriever_has_retrieve_method(self):
        from rag_baseline.retrieval.base import BaseRetriever
        assert hasattr(BaseRetriever, "retrieve")

    def test_base_retriever_has_index_method(self):
        from rag_baseline.retrieval.base import BaseRetriever
        assert hasattr(BaseRetriever, "index")


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class TestDenseRetriever:
    """Tests for FAISS-based dense retrieval."""

    @pytest.fixture
    def sample_corpus(self):
        return [
            {"passage_id": "p1", "text": "Michael Jordan was born in 1963 in Brooklyn.", "source": "doc_1"},
            {"passage_id": "p2", "text": "Michael Jordan is a retired basketball player.", "source": "doc_2"},
            {"passage_id": "p3", "text": "The capital of France is Paris.", "source": "doc_3"},
            {"passage_id": "p4", "text": "Michael B. Jordan is an American actor.", "source": "doc_4"},
            {"passage_id": "p5", "text": "Basketball was invented by James Naismith.", "source": "doc_5"},
        ]

    @pytest.mark.slow
    def test_dense_retriever_creates(self):
        from rag_baseline.retrieval.dense import DenseRetriever

        retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
        assert retriever is not None

    @pytest.mark.slow
    def test_dense_retriever_index_and_retrieve(self, sample_corpus):
        from rag_baseline.retrieval.dense import DenseRetriever
        from rag_baseline.schemas.retrieval import RetrievalOutput

        retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
        retriever.index(sample_corpus)

        result = retriever.retrieve("When was Michael Jordan born?", top_k=3)

        assert isinstance(result, RetrievalOutput)
        assert len(result.retrieved_passages) == 3
        assert result.retrieved_passages[0].rank == 1
        # The top result should be about MJ's birth
        assert "1963" in result.retrieved_passages[0].text or "Jordan" in result.retrieved_passages[0].text

    @pytest.mark.slow
    def test_dense_retriever_scores_are_sorted(self, sample_corpus):
        from rag_baseline.retrieval.dense import DenseRetriever

        retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
        retriever.index(sample_corpus)
        result = retriever.retrieve("basketball player", top_k=3)

        scores = [p.retrieval_score for p in result.retrieved_passages]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.slow
    def test_dense_retriever_respects_top_k(self, sample_corpus):
        from rag_baseline.retrieval.dense import DenseRetriever

        retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
        retriever.index(sample_corpus)
        result = retriever.retrieve("Jordan", top_k=2)
        assert len(result.retrieved_passages) == 2


# ---------------------------------------------------------------------------
# Sparse (BM25) retriever
# ---------------------------------------------------------------------------

class TestSparseRetriever:
    """Tests for BM25-based sparse retrieval."""

    @pytest.fixture
    def sample_corpus(self):
        return [
            {"passage_id": "p1", "text": "Michael Jordan was born in 1963 in Brooklyn.", "source": "doc_1"},
            {"passage_id": "p2", "text": "Michael Jordan is a retired basketball player.", "source": "doc_2"},
            {"passage_id": "p3", "text": "The capital of France is Paris.", "source": "doc_3"},
            {"passage_id": "p4", "text": "Michael B. Jordan is an American actor.", "source": "doc_4"},
            {"passage_id": "p5", "text": "Basketball was invented by James Naismith.", "source": "doc_5"},
        ]

    def test_sparse_retriever_creates(self):
        from rag_baseline.retrieval.sparse import SparseRetriever

        retriever = SparseRetriever()
        assert retriever is not None

    def test_sparse_retriever_index_and_retrieve(self, sample_corpus):
        from rag_baseline.retrieval.sparse import SparseRetriever
        from rag_baseline.schemas.retrieval import RetrievalOutput

        retriever = SparseRetriever()
        retriever.index(sample_corpus)

        result = retriever.retrieve("When was Michael Jordan born?", top_k=3)

        assert isinstance(result, RetrievalOutput)
        assert len(result.retrieved_passages) == 3
        assert result.retrieved_passages[0].rank == 1

    def test_sparse_retriever_scores_sorted(self, sample_corpus):
        from rag_baseline.retrieval.sparse import SparseRetriever

        retriever = SparseRetriever()
        retriever.index(sample_corpus)
        result = retriever.retrieve("basketball player", top_k=3)

        scores = [p.retrieval_score for p in result.retrieved_passages]
        assert scores == sorted(scores, reverse=True)

    def test_sparse_retriever_respects_top_k(self, sample_corpus):
        from rag_baseline.retrieval.sparse import SparseRetriever

        retriever = SparseRetriever()
        retriever.index(sample_corpus)
        result = retriever.retrieve("Jordan", top_k=2)
        assert len(result.retrieved_passages) == 2


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    """Tests for hybrid (dense + sparse) retrieval with score fusion."""

    @pytest.fixture
    def sample_corpus(self):
        return [
            {"passage_id": "p1", "text": "Michael Jordan was born in 1963 in Brooklyn.", "source": "doc_1"},
            {"passage_id": "p2", "text": "Michael Jordan is a retired basketball player.", "source": "doc_2"},
            {"passage_id": "p3", "text": "The capital of France is Paris.", "source": "doc_3"},
            {"passage_id": "p4", "text": "Michael B. Jordan is an American actor.", "source": "doc_4"},
            {"passage_id": "p5", "text": "Basketball was invented by James Naismith.", "source": "doc_5"},
        ]

    @pytest.mark.slow
    def test_hybrid_retriever_creates(self):
        from rag_baseline.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(dense_model_name="BAAI/bge-base-en-v1.5")
        assert retriever is not None

    @pytest.mark.slow
    def test_hybrid_retriever_index_and_retrieve(self, sample_corpus):
        from rag_baseline.retrieval.hybrid import HybridRetriever
        from rag_baseline.schemas.retrieval import RetrievalOutput

        retriever = HybridRetriever(dense_model_name="BAAI/bge-base-en-v1.5")
        retriever.index(sample_corpus)

        result = retriever.retrieve("When was Michael Jordan born?", top_k=3)

        assert isinstance(result, RetrievalOutput)
        assert len(result.retrieved_passages) == 3
        assert result.retrieved_passages[0].rank == 1

    @pytest.mark.slow
    def test_hybrid_retriever_combines_both_sources(self, sample_corpus):
        """Hybrid should surface results that benefit from both dense and sparse signals."""
        from rag_baseline.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(dense_model_name="BAAI/bge-base-en-v1.5")
        retriever.index(sample_corpus)

        result = retriever.retrieve("Michael Jordan born year", top_k=3)
        passage_ids = {p.passage_id for p in result.retrieved_passages}
        # p1 has both lexical match and semantic match — should be in top 3
        assert "p1" in passage_ids

    @pytest.mark.slow
    def test_hybrid_retriever_scores_sorted(self, sample_corpus):
        from rag_baseline.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(dense_model_name="BAAI/bge-base-en-v1.5")
        retriever.index(sample_corpus)
        result = retriever.retrieve("basketball", top_k=3)

        scores = [p.retrieval_score for p in result.retrieved_passages]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

class TestRetrieverFactory:
    """Tests for the retriever factory that instantiates by config."""

    @pytest.mark.slow
    def test_factory_creates_dense(self):
        from rag_baseline.retrieval import create_retriever

        retriever = create_retriever(retriever_type="dense", dense_model="BAAI/bge-base-en-v1.5")
        from rag_baseline.retrieval.dense import DenseRetriever
        assert isinstance(retriever, DenseRetriever)

    def test_factory_creates_sparse(self):
        from rag_baseline.retrieval import create_retriever

        retriever = create_retriever(retriever_type="sparse")
        from rag_baseline.retrieval.sparse import SparseRetriever
        assert isinstance(retriever, SparseRetriever)

    @pytest.mark.slow
    def test_factory_creates_hybrid(self):
        from rag_baseline.retrieval import create_retriever

        retriever = create_retriever(retriever_type="hybrid", dense_model="BAAI/bge-base-en-v1.5")
        from rag_baseline.retrieval.hybrid import HybridRetriever
        assert isinstance(retriever, HybridRetriever)

    def test_factory_creates_none(self):
        from rag_baseline.retrieval import create_retriever

        retriever = create_retriever(retriever_type="none")
        assert retriever is None

    def test_factory_rejects_unknown(self):
        from rag_baseline.retrieval import create_retriever

        with pytest.raises(ValueError):
            create_retriever(retriever_type="quantum")
