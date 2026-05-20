"""
Retrieval evaluation: Precision@K, Recall@K, NDCG@K.
Hallucination scoring via token-overlap between answer and retrieved context.
"""
import math
import re
from collections import Counter


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        (1.0 / math.log2(rank + 2))
        for rank, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate_retrieval(
    queries: list[dict],  # [{"query": str, "relevant_ids": list[str], "retrieved_ids": list[str]}]
    k_values: list[int] = [1, 3, 5],
) -> dict[str, float]:
    results: dict[str, list[float]] = {f"p@{k}": [] for k in k_values}
    results.update({f"r@{k}": [] for k in k_values})
    results.update({f"ndcg@{k}": [] for k in k_values})

    for item in queries:
        rel = set(item["relevant_ids"])
        ret = item["retrieved_ids"]
        for k in k_values:
            results[f"p@{k}"].append(precision_at_k(ret, rel, k))
            results[f"r@{k}"].append(recall_at_k(ret, rel, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(ret, rel, k))

    return {metric: sum(vals) / len(vals) for metric, vals in results.items() if vals}


# ---------------------------------------------------------------------------
# Hallucination scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Counter:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return Counter(tokens)


def hallucination_score(answer: str, context_chunks: list[dict]) -> dict:
    """
    Token-overlap faithfulness score (0-1, higher = more grounded).
    Also returns unsupported_ratio as a proxy hallucination rate.
    """
    context_text = " ".join(c["text"] for c in context_chunks)
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(context_text)

    total_answer_tokens = sum(answer_tokens.values())
    if not total_answer_tokens:
        return {"faithfulness": 1.0, "unsupported_ratio": 0.0}

    supported = sum(
        min(count, context_tokens[token]) for token, count in answer_tokens.items()
    )
    faithfulness = supported / total_answer_tokens
    return {
        "faithfulness": round(faithfulness, 4),
        "unsupported_ratio": round(1.0 - faithfulness, 4),
    }
