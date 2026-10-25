# Weaviate

## Overview
Weaviate is an open-source vector database written in Go. It offers REST, gRPC, and GraphQL APIs and is available self-hosted or as Weaviate Cloud.

## Features
Weaviate stores objects with properties and vectors. Vectorizer and generative modules can call embedding or language models so that data is embedded on import. It supports hybrid search that combines BM25 keyword scoring with vector similarity, filtering, and multi-tenancy with per-tenant isolation.

## Indexing
The default vector index is HNSW. Flat and dynamic index options exist, and compression such as product, binary, and scalar quantization is available to reduce memory.

## Best fit
Applications that want built-in hybrid search, schema-based objects, and module integrations with embedding providers.
