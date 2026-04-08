"""Dense retriever using FAISS and sentence-transformers (PRD 1 §9.2).

Uses a dense embedding model to encode passages and queries,
then performs approximate nearest-neighbor search via FAISS.
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_baseline.retrieval.base import BaseRetriever
from rag_baseline.schemas.retrieval import RetrievalOutput, RetrievedPassage


class DenseRetriever(BaseRetriever):
    """FAISS-based dense retriever."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index = None
        self._corpus: list[dict] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            # precache_models.sh downloads models with cache_dir=HF_HOME
            # (not HF_HOME/hub), so files land at {HF_HOME}/models--BAAI--...
            # We must pass the same root here so offline lookup succeeds.
            hf_home = os.environ.get("HF_HOME")
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=hf_home,
            )
        return self._model

    def index(self, corpus: list[dict]) -> None:
        """Index corpus using FAISS."""
        import faiss

        self._corpus = corpus
        texts = [doc["text"] for doc in corpus]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalOutput:
        """Retrieve top-k passages via dense similarity search."""
        if self._index is None:
            raise RuntimeError("Must call index() before retrieve()")

        top_k = min(top_k, len(self._corpus))
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self._index.search(query_embedding, top_k)

        passages = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            doc = self._corpus[idx]
            passages.append(
                RetrievedPassage(
                    passage_id=doc["passage_id"],
                    text=doc["text"],
                    source=doc["source"],
                    retrieval_score=float(score),
                    rank=rank,
                )
            )

        return RetrievalOutput(
            example_id="",  # Set by caller
            retrieved_passages=passages,
        )
