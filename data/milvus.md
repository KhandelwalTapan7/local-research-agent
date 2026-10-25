# Milvus

## Overview
Milvus is an open-source, cloud-native vector database designed for large scale. Its architecture separates storage and compute and runs as a distributed system on Kubernetes. Zilliz Cloud is the managed service, and a lightweight standalone mode exists for development.

## Indexing
Milvus supports many index types, including FLAT, IVF variants, HNSW, and disk-based and GPU-accelerated indexes. It supports dense and sparse vectors, scalar filtering, and hybrid search across multiple vector fields.

## Trade-offs
It targets collections from hundreds of millions to billions of vectors. The distributed deployment has more moving parts than single-binary systems and demands more operational expertise.

## Best fit
Very large collections, high ingestion and query throughput, and teams able to operate a distributed system or buy the managed version.
