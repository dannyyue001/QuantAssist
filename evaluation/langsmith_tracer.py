"""
LangSmith tracing integration and automated regression benchmarking.
"""
import time
import asyncio
from functools import wraps
from langsmith import Client
from config import settings


ls_client = Client(api_key=settings.langchain_api_key) if settings.langchain_api_key else None


def trace_latency(name: str):
    """Decorator that logs wall-clock latency of async functions to LangSmith."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if ls_client:
                ls_client.create_run(
                    name=name,
                    run_type="chain",
                    inputs={"args_len": len(args)},
                    outputs={"latency_ms": elapsed_ms},
                    project_name=settings.langchain_project,
                )
            return result
        return wrapper
    return decorator


async def run_regression_benchmark(
    retriever,
    eval_queries: list[dict],
    k_values: list[int] = [1, 3, 5],
) -> dict:
    """
    Runs retrieval over all eval_queries, measures latency, computes precision@k.
    Returns a benchmark report dict.
    """
    from evaluation.metrics import evaluate_retrieval

    latencies: list[float] = []
    annotated: list[dict] = []

    for item in eval_queries:
        start = time.perf_counter()
        response = await retriever.search(item["query"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        retrieved_ids = [r["chunk_id"] for r in response["results"]]
        annotated.append({
            "query": item["query"],
            "relevant_ids": item["relevant_ids"],
            "retrieved_ids": retrieved_ids,
        })

    metrics = evaluate_retrieval(annotated, k_values=k_values)
    report = {
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 2),
        "num_queries": len(eval_queries),
        **metrics,
    }

    if ls_client:
        ls_client.create_run(
            name="regression_benchmark",
            run_type="chain",
            inputs={"num_queries": len(eval_queries)},
            outputs=report,
            project_name=settings.langchain_project,
        )

    return report
