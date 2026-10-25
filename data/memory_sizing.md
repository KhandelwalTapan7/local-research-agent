# Memory Sizing for Vector Search

## Raw storage
A float32 value takes 4 bytes. The raw size of a collection is vectors * dimensions * 4 bytes. For example, 1 million vectors of 768 dimensions need 1,000,000 * 768 * 4 = 3.072 GB before any index overhead. Use 1 GB = 10^9 bytes for these estimates.

## Index overhead
HNSW adds graph links on top of the raw vectors. A common estimate is vectors * 2 * M * 4 bytes for the base layer. Metadata, payload indexes, and the operating system add further overhead, so plan for 1.5x to 2x of the raw size when everything is kept in RAM.

## Reducing memory
Quantization trades some recall for memory. Scalar int8 cuts vector memory 4x and binary quantization cuts it 32x. Keeping full-precision vectors on disk and only the compressed vectors in RAM is a common pattern, with the top candidates rescored from disk.

## Dimensions matter
Memory grows linearly with embedding dimensionality. Truncating Matryoshka-style embeddings or choosing a smaller model, such as a 384-dimension model instead of a 1536-dimension one, reduces memory by the same factor.
