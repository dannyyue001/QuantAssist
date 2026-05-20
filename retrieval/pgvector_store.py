import asyncpg
import numpy as np
from config import settings


CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    doc_id      TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({dim}),
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 256);
""".format(dim=settings.embedding_dim)


class PGVectorStore:
    """Async PostgreSQL pgvector store for persistent vector storage."""

    def __init__(self):
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=4,
            max_size=16,
        )
        await self._init_schema()

    async def _init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            for stmt in CREATE_TABLE_SQL.split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(stmt)

    async def upsert(self, chunks: list[dict]) -> None:
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO documents (chunk_id, doc_id, text, embedding, metadata)
                VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE
                    SET text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """,
                [
                    (
                        c["chunk_id"],
                        c["metadata"]["doc_id"],
                        c["text"],
                        str(c["embedding"]),
                        str(c["metadata"]).replace("'", '"'),
                    )
                    for c in chunks
                ],
            )

    async def similarity_search(
        self, query_embedding: list[float], top_k: int = settings.top_k
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunk_id, doc_id, text, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM documents
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                str(query_embedding),
                top_k,
            )
        return [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "score": float(r["score"]),
            }
            for r in rows
        ]

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
