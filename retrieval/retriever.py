import asyncio
from embeddings.embed_pipeline import EmbeddingPipeline
from retrieval.faiss_index import FAISSIndex
from retrieval.pgvector_store import PGVectorStore
from retrieval.redis_cache import RedisSemanticCache
from config import settings


class HybridRetriever:
    """
    Async retriever: Redis semantic cache → FAISS ANN → pgvector fallback.
    Target: sub-350ms avg retrieval latency.
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        pg_store: PGVectorStore,
        cache: RedisSemanticCache,
        embedder: EmbeddingPipeline,
    ):
        self.faiss_index = faiss_index
        self.pg_store = pg_store
        self.cache = cache
        self.embedder = embedder

    async def search(self, query: str, top_k: int = settings.top_k) -> dict:
        query_embedding = await self.embedder.embed_query(query)

        cached = await self.cache.get(query_embedding)
        if cached is not None:
            return {"results": cached, "source": "redis_cache"}

        faiss_results, pg_results = await asyncio.gather(
            asyncio.to_thread(self.faiss_index.search, query_embedding, top_k),
            self.pg_store.similarity_search(query_embedding, top_k),
        )

        merged = _merge_results(faiss_results, pg_results, top_k)

        await self.cache.set(query_embedding, merged)

        return {"results": merged, "source": "hybrid"}


def _merge_results(
    faiss_results: list[dict],
    pg_results: list[dict],
    top_k: int,
) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []

    for result in faiss_results + pg_results:
        cid = result.get("chunk_id", "")
        if cid not in seen:
            seen.add(cid)
            merged.append(result)
        if len(merged) >= top_k:
            break

    return merged
