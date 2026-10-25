# Pinecone

## Overview
Pinecone is a fully managed, proprietary vector database. It cannot be self-hosted. It is offered in serverless and pod-based forms.

## Features
Data is organised in indexes and namespaces, and namespaces provide lightweight isolation, for example one per tenant. Vectors carry metadata and can be filtered at query time. Pinecone supports dense and sparse vectors for hybrid retrieval and offers integrated embedding and reranking models.

## Trade-offs
There is no infrastructure to operate and scaling is handled by the service, but cost grows with stored data and query volume, and data must live in the vendor's cloud regions.

## Best fit
Teams that want zero operational burden and accept vendor lock-in and usage-based pricing.
