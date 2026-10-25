# FAISS

## Overview
FAISS is a C++ library with Python bindings from Meta for similarity search over dense vectors. It is a library, not a database: it has no server, no built-in metadata filtering, and no replication.

## Index types
FAISS provides flat exact indexes, IVF indexes that cluster vectors into cells, product quantization for compression, and HNSW graphs. These can be combined, for example IVF with PQ. GPU implementations are available for several index types and give very high throughput.

## Operations
Indexes are saved to and loaded from files. Applications must manage ids, metadata, updates, and deletions themselves, and some index types do not support removal of vectors.

## Best fit
Research, batch pipelines, and custom systems that need maximum search speed or GPU acceleration and are willing to build the surrounding infrastructure.
