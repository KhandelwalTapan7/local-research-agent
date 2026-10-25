# Vector Index Algorithms

## HNSW
Hierarchical Navigable Small World builds a multi-layer proximity graph. The parameter M sets how many links each node keeps, efConstruction controls build quality, and efSearch controls the recall and latency trade-off at query time. HNSW gives high recall with low latency but the graph must sit in memory and builds are slow. At the base layer each vector has up to 2*M links, each stored as a 4-byte id, so with M=16 the graph adds roughly 128 bytes per vector.

## IVF
An inverted file index clusters vectors with k-means into nlist cells. A query probes the nprobe nearest cells instead of scanning everything. It builds quickly and uses little extra memory, but recall depends on nprobe and the data distribution.

## Product quantization
PQ splits each vector into m sub-vectors and encodes each with an 8-bit code, so a vector takes m bytes. With 768 dimensions and m=96 a vector shrinks from 3072 bytes to 96 bytes, a 32x reduction, at some cost in recall. It is often paired with IVF and with re-ranking over full-precision vectors.

## Scalar and binary quantization
Scalar quantization stores each dimension as int8, a 4x reduction from float32. Binary quantization stores one bit per dimension, a 32x reduction, and works best with high-dimensional embeddings plus rescoring.
