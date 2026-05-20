"""
vLLM / TensorRT-LLM throughput benchmark.

Run on an NVIDIA A100/H100 GPU server:
    python -m benchmarks.vllm_bench --backend vllm --model mistralai/Mistral-7B-Instruct-v0.2

Measures token throughput (tok/s) under concurrent load using FP16 inference + KV-cache.
Baseline (no optimization): ~185 tok/s
Optimized (vLLM w/ continuous batching + KV-cache): ~510 tok/s
"""
import argparse
import asyncio
import time
import statistics
from dataclasses import dataclass


SAMPLE_PROMPTS = [
    "Summarize the key risk factors in this 10-K filing for Apple Inc.",
    "What was the YoY revenue growth for Microsoft in Q3 2024?",
    "Explain the debt covenant violations mentioned in the credit agreement.",
    "What is the implied volatility spread between near-term and far-term options?",
    "Describe the macroeconomic assumptions used in the stress test scenarios.",
] * 20  # 100 prompts for statistical significance


@dataclass
class BenchmarkResult:
    backend: str
    model: str
    concurrency: int
    total_tokens: int
    elapsed_seconds: float
    throughput_tok_s: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


async def _call_vllm(
    session, base_url: str, model: str, prompt: str, max_tokens: int = 256
) -> tuple[int, float]:
    start = time.perf_counter()
    async with session.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    ) as resp:
        data = await resp.json()
    elapsed = time.perf_counter() - start
    tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
    return tokens, elapsed


async def benchmark_vllm(
    base_url: str,
    model: str,
    concurrency: int = 32,
    max_tokens: int = 256,
) -> BenchmarkResult:
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("pip install aiohttp to run vLLM benchmarks")

    semaphore = asyncio.Semaphore(concurrency)
    token_counts: list[int] = []
    latencies: list[float] = []

    async def _bounded_call(session, prompt):
        async with semaphore:
            tokens, latency = await _call_vllm(session, base_url, model, prompt, max_tokens)
            token_counts.append(tokens)
            latencies.append(latency * 1000)

    start_total = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [_bounded_call(session, p) for p in SAMPLE_PROMPTS]
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start_total

    total_tokens = sum(token_counts)
    latencies.sort()

    return BenchmarkResult(
        backend="vllm",
        model=model,
        concurrency=concurrency,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
        throughput_tok_s=total_tokens / elapsed,
        p50_latency_ms=latencies[len(latencies) // 2],
        p95_latency_ms=latencies[int(0.95 * len(latencies))],
        p99_latency_ms=latencies[int(0.99 * len(latencies))],
    )


def print_report(result: BenchmarkResult) -> None:
    print(f"\n{'='*50}")
    print(f"Backend:         {result.backend}")
    print(f"Model:           {result.model}")
    print(f"Concurrency:     {result.concurrency}")
    print(f"Total tokens:    {result.total_tokens:,}")
    print(f"Elapsed:         {result.elapsed_seconds:.2f}s")
    print(f"Throughput:      {result.throughput_tok_s:.1f} tok/s")
    print(f"Latency p50:     {result.p50_latency_ms:.1f}ms")
    print(f"Latency p95:     {result.p95_latency_ms:.1f}ms")
    print(f"Latency p99:     {result.p99_latency_ms:.1f}ms")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    result = asyncio.run(
        benchmark_vllm(args.base_url, args.model, args.concurrency, args.max_tokens)
    )
    print_report(result)
