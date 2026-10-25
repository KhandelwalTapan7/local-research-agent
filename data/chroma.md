# Chroma

## Overview
Chroma is an open-source embedding database with a Python-first developer experience. It runs embedded inside a Python process, or as a client-server deployment, and a hosted Chroma Cloud offering exists.

## Indexing and search
Chroma builds an HNSW index and stores metadata and documents alongside the vectors. A collection holds embeddings, documents, and metadata, and queries can filter on metadata and document text.

## Operations
Persistent local mode writes to a directory on disk, which makes setup trivial for notebooks, prototypes, and small applications. It is not designed for billion-scale workloads or complex horizontal scaling in its local form.

## Best fit
Prototypes, local development, and small to medium applications where simplicity matters more than scale.
