import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


FINANCIAL_SEPARATORS = [
    "\n## ", "\n# ",         # markdown headings (10-K/10-Q sections)
    "\nItem ", "\nPART ",    # SEC filing section markers
    "\n\n\n", "\n\n",
    "\n", " ", "",
]


class DocumentChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 120):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=FINANCIAL_SEPARATORS,
            length_function=len,
        )

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def chunk_document(self, text: str, metadata: dict) -> list[dict]:
        text = self._clean(text)
        chunks = self.splitter.split_text(text)
        return [
            {
                "text": chunk,
                "metadata": {**metadata, "chunk_index": i, "total_chunks": len(chunks)},
                "chunk_id": f"{metadata['doc_id']}_{i}",
            }
            for i, chunk in enumerate(chunks)
        ]

    def chunk_batch(self, documents: list[dict]) -> list[dict]:
        results = []
        for doc in documents:
            results.extend(self.chunk_document(doc["text"], doc["metadata"]))
        return results
