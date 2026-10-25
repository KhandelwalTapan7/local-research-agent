import json
import re
import threading

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from rag.ingest import EMBED_MODEL, INDEX_DIR, ingest

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(INDEX_DIR / "emb.npy")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = TextEmbedding(EMBED_MODEL)
        self.reranker = TextCrossEncoder(model_name=RERANK_MODEL)

    def search(self, query, k=4, pool=15, rerank_pool=10):
        q = np.array(list(self.embedder.query_embed(query))[0], dtype="float32")
        q /= np.linalg.norm(q)
        fused = {}
        for scores in (self.embeddings @ q, self.bm25.get_scores(tokenize(query))):
            ranked = [int(i) for i in np.argsort(-scores)[:pool] if scores[i] > 0]
            for rank, idx in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:rerank_pool]
        if not candidates:
            return []
        scores = list(self.reranker.rerank(query, [self.chunks[i]["text"] for i in candidates]))
        best = sorted(zip(candidates, scores), key=lambda x: -x[1])[:k]
        return [{**self.chunks[i], "score": float(s)} for i, s in best]


_instance = None
_lock = threading.Lock()


def get_retriever():
    global _instance
    with _lock:
        if _instance is None:
            if not (INDEX_DIR / "chunks.json").exists():
                ingest()
            _instance = Retriever()
    return _instance
