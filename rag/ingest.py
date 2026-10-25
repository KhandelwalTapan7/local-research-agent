import json
import os
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "runtime" / "index"))
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def windows(text, size=900, overlap=150):
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text) - overlap, size - overlap)]


def split_markdown(path):
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()
    chunks = []
    for section in re.split(r"\n(?=## )", text):
        lines = section.strip().splitlines()
        heading = "Overview"
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        if lines and lines[0].startswith("## "):
            heading = lines[0][3:].strip()
            lines = lines[1:]
        body = "\n".join(lines).strip()
        if not body:
            continue
        for part in windows(body):
            chunks.append({"title": title, "path": path.name, "heading": heading, "text": f"{title} - {heading}\n{part}"})
    return chunks


def ingest():
    chunks = [c for p in sorted(DATA_DIR.glob("*.md")) for c in split_markdown(p)]
    if not chunks:
        raise RuntimeError(f"No markdown documents found in {DATA_DIR}")
    embeddings = np.array(list(TextEmbedding(EMBED_MODEL).embed([c["text"] for c in chunks])), dtype="float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    np.save(INDEX_DIR / "emb.npy", embeddings)
    return len(chunks)


if __name__ == "__main__":
    print(f"Indexed {ingest()} chunks into {INDEX_DIR}")
