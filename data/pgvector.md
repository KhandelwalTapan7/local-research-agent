# pgvector

## Overview
pgvector is a PostgreSQL extension that adds a vector column type and similarity operators for L2 distance, inner product, and cosine distance.

## Indexing
It supports exact search and two approximate indexes: HNSW and IVFFlat. HNSW generally gives a better speed and recall trade-off but takes longer to build and uses more memory. IVFFlat builds faster and needs training data present when the index is created. The vector type can store up to 16,000 dimensions, and indexes support up to 2,000 dimensions for it.

## Strengths
Vectors live next to relational data, so one SQL query can join, filter, and rank. Transactions, backups, replication, and access control come from Postgres itself.

## Limits
Very large collections and high-throughput workloads can strain a single Postgres instance, and specialised vector databases offer more compression options.

## Best fit
Applications already on Postgres with up to a few million vectors that value transactional consistency and a single data store.
