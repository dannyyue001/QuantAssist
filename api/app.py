"""
QuantAssist FastAPI application.
Endpoints: /ingest, /query, /benchmark, /health
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import settings
from embeddings.chunking import DocumentChunker
from embeddings.embed_pipeline import EmbeddingPipeline
from retrieval.faiss_index import FAISSIndex
from retrieval.pgvector_store import PGVectorStore
from retrieval.redis_cache import RedisSemanticCache
from retrieval.retriever import HybridRetriever
from agents.graph import run_pipeline
from evaluation.metrics import evaluate_retrieval, hallucination_score
from evaluation.langsmith_tracer import run_regression_benchmark


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

chunker = DocumentChunker()
embedder = EmbeddingPipeline()
faiss_index = FAISSIndex()
pg_store = PGVectorStore()
cache = RedisSemanticCache()
retriever: Optional[HybridRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    try:
        await pg_store.connect()
        await cache.connect()
    except Exception as e:
        print(f"[warn] Could not connect to Postgres/Redis: {e}. Running in FAISS-only mode.")

    retriever = HybridRetriever(
        faiss_index=faiss_index,
        pg_store=pg_store,
        cache=cache,
        embedder=embedder,
    )
    yield
    await pg_store.close()
    await cache.close()


app = FastAPI(title="QuantAssist", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    documents: list[dict]  # [{"text": str, "metadata": {"doc_id": str, ...}}]


class QueryRequest(BaseModel):
    query: str
    top_k: int = settings.top_k
    use_agents: bool = False


class EvalRequest(BaseModel):
    queries: list[dict]  # [{"query": str, "relevant_ids": list[str]}]
    k_values: list[int] = [1, 3, 5]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def healthcheck():
    return {"status": "running", "version": "1.0.0"}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    chunks = chunker.chunk_batch(req.documents)
    chunks = await embedder.embed_chunks(chunks)
    faiss_index.add(chunks)

    try:
        await pg_store.upsert(chunks)
    except Exception:
        pass  # pgvector optional

    return {"ingested_chunks": len(chunks)}


@app.post("/query")
async def query(req: QueryRequest):
    if retriever is None:
        raise HTTPException(503, "Retriever not initialized")

    start = time.perf_counter()
    retrieval_result = await retriever.search(req.query, top_k=req.top_k)
    retrieval_ms = (time.perf_counter() - start) * 1000

    response: dict = {
        "query": req.query,
        "retrieval_ms": round(retrieval_ms, 2),
        "source": retrieval_result["source"],
        "results": retrieval_result["results"],
    }

    if req.use_agents:
        agent_start = time.perf_counter()
        agent_output = await run_pipeline(req.query, retrieval_result["results"])
        response["agent_ms"] = round((time.perf_counter() - agent_start) * 1000, 2)
        response["research"] = agent_output["research"]
        response["summary"] = agent_output["summary"]
        response["hallucination"] = hallucination_score(
            agent_output["research"], retrieval_result["results"]
        )

    return response


@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    if retriever is None:
        raise HTTPException(503, "Retriever not initialized")

    report = await run_regression_benchmark(retriever, req.queries, req.k_values)
    return report


@app.get("/stats")
def stats():
    return {
        "faiss_index_size": faiss_index.index.ntotal,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "cache_threshold": settings.cache_similarity_threshold,
    }
