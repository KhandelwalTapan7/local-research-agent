# Choosing a Vector Database

## Decision factors
Scale, operational capacity, filtering needs, hybrid search needs, existing infrastructure, and budget drive the choice.

## Heuristics
- Prototype or notebook work: Chroma, or FAISS for pure in-memory experiments.
- Already running PostgreSQL with a modest number of vectors: pgvector keeps everything in one transactional store.
- Heavy metadata filtering and fine memory control, self-hosted: Qdrant.
- Built-in hybrid search and embedding-provider modules: Weaviate.
- Hundreds of millions to billions of vectors, distributed operation: Milvus.
- No operations team and a willingness to pay for a managed service: Pinecone.
- Maximum raw speed, GPU acceleration, and custom serving: FAISS as a library.

## Common mistakes
Choosing for peak scale that will never arrive, ignoring the cost of filtered search recall, and forgetting that embedding dimensionality and quantization often change memory needs more than the database choice does.
