import os
import faiss
import numpy as np
from config import settings


class FAISSIndex:
    """IVF-Flat FAISS index for scalable approximate nearest-neighbour search."""

    def __init__(self, dimension: int = settings.embedding_dim, n_lists: int = 256):
        self.dimension = dimension
        self.n_lists = n_lists
        self.documents: list[dict] = []

        quantizer = faiss.IndexFlatIP(dimension)  # inner-product (cosine after L2-norm)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, n_lists, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = 32  # probe 32 lists for recall/speed balance

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-10)

    def train(self, embedded_chunks: list[dict]) -> None:
        vectors = np.array(
            [c["embedding"] for c in embedded_chunks], dtype="float32"
        )
        vectors = self._normalize(vectors)
        if not self.index.is_trained:
            self.index.train(vectors)

    def add(self, embedded_chunks: list[dict]) -> None:
        if not self.index.is_trained:
            self.train(embedded_chunks)
        vectors = np.array(
            [c["embedding"] for c in embedded_chunks], dtype="float32"
        )
        vectors = self._normalize(vectors)
        self.index.add(vectors)
        self.documents.extend(embedded_chunks)

    def search(self, query_vector: list[float], top_k: int = settings.top_k) -> list[dict]:
        query = np.array([query_vector], dtype="float32")
        query = self._normalize(query)
        scores, indices = self.index.search(query, top_k)
        return [
            {**self.documents[i], "score": float(scores[0][rank])}
            for rank, i in enumerate(indices[0])
            if i != -1
        ]

    def save(self, path: str = settings.faiss_index_path) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str = settings.faiss_index_path, documents: list[dict] | None = None) -> "FAISSIndex":
        instance = cls.__new__(cls)
        instance.index = faiss.read_index(path)
        instance.documents = documents or []
        instance.dimension = instance.index.d
        return instance
