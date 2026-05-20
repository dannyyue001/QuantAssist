import asyncio
from openai import AsyncOpenAI
from config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


class EmbeddingPipeline:
    """Async batched embedding pipeline with parallel execution."""

    def __init__(
        self,
        model: str = settings.embedding_model,
        batch_size: int = settings.embedding_batch_size,
        max_concurrent: int = settings.max_concurrent_batches,
    ):
        self.model = model
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self.semaphore:
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]

    async def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        texts = [c["text"] for c in chunks]
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        tasks = [self._embed_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        embeddings = [emb for batch in batch_results for emb in batch]
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks

    async def embed_query(self, query: str) -> list[float]:
        response = await client.embeddings.create(
            model=self.model,
            input=query,
        )
        return response.data[0].embedding
