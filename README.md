# QuantAssist
# High-Performance Financial RAG System

A production-style Retrieval-Augmented Generation (RAG) system designed for large-scale financial document retrieval and low-latency LLM inference.

This project benchmarks modern inference stacks including TensorRT-LLM and vLLM on NVIDIA A100/H100 GPUs while evaluating retrieval quality, hallucination behavior, and system throughput under concurrent workloads.

## Features

- Retrieval over 500K+ financial documents
- OpenAI text-embedding-3-large embeddings
- FAISS + PostgreSQL pgvector hybrid vector storage
- Async ingestion and retrieval pipelines
- Redis semantic caching
- TensorRT-LLM vs vLLM benchmarking
- FP16 inference + KV-cache optimization
- LangSmith tracing and regression testing
- Precision@K and hallucination evaluation

## System Architecture

[Architecture Diagram Here]

## Performance Highlights

| Metric | Result |
|---|---|
| Retrieval Latency | <350ms avg |
| Token Throughput | 185 → 510 tok/s |
| Semantic Cache Hit Rate | ~20% |
| Dataset Size | 500K+ documents |

## Tech Stack

- Python
- FAISS
- PostgreSQL + pgvector
- Redis
- FastAPI
- TensorRT-LLM
- vLLM
- LangChain / LangSmith
- Docker
