# Qdrant

## Overview
Qdrant is an open-source vector database written in Rust. It exposes REST and gRPC APIs and can run as a single Docker container, as a distributed cluster, or as the managed Qdrant Cloud service.

## Indexing and search
Qdrant uses an HNSW graph index. Payload filtering is applied during graph traversal, so filtered searches keep good recall instead of filtering results after the search. It supports dense vectors, sparse vectors for keyword-style retrieval, multiple named vectors per point, and hybrid queries that fuse several result lists.

## Compression
Qdrant offers scalar quantization (float32 to int8, about 4x smaller), product quantization, and binary quantization (about 32x smaller). Original vectors can be kept on disk for rescoring, which lets the quantized copy stay in RAM.

## Best fit
Self-hosted or cloud deployments that need rich metadata filtering, hybrid search, and fine control over memory use.
