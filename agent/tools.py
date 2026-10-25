import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

from rag.retriever import get_retriever

RERANK_MIN = float(os.getenv("RERANK_MIN", "0"))

SPECS = """rag_search {"query": str} - search the local vector-database knowledge base; returns passages with source ids
web_search {"query": str} - live web search; use when the knowledge base lacks the answer
summarise {"source_id": "S1"} - compress one retrieved source into key points
code_executor {"code": str} - run plain Python (stdlib only) to compute numbers and print the result; other tools such as memory_store do not exist inside the code
structured_output {"instruction": str, "source_ids": ["S1"], "format": "table" | "json"} - build a table or JSON block from retrieved sources
memory_store {"key": str, "value": str} - persist a fact for this session; only when the user asks you to remember something
memory_recall {"query": str} - retrieve facts stored in this session; "*" returns all"""


@dataclass
class Context:
    session_id: str
    memory: object
    llm: object
    sources: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    structured: list = field(default_factory=list)


def add_source(ctx, title, location, text):
    for sid, s in ctx.sources.items():
        if s["location"] == location and s["text"] == text:
            return sid
    sid = f"S{len(ctx.sources) + 1}"
    ctx.sources[sid] = {"id": sid, "title": title, "location": location, "text": text}
    return sid


def rag_search(ctx, query):
    hits = get_retriever().search(query, k=4)
    if not hits:
        return {"observation": "No results.", "good": False}
    if hits[0]["score"] <= RERANK_MIN:
        titles = ", ".join(sorted({h["title"] for h in hits}))
        return {"observation": f"Low relevance (best rerank score {hits[0]['score']:.2f}). Closest documents: {titles}.", "good": False}
    lines = []
    for h in hits:
        sid = add_source(ctx, h["title"], f"{h['path']} > {h['heading']}", h["text"])
        lines.append(f"[{sid}] {h['title']} / {h['heading']} (score {h['score']:.2f}): {h['text'][:400]}")
    return {"observation": "\n".join(lines), "good": True}


def real_url(href):
    return parse_qs(urlparse(href).query).get("uddg", [href])[0]


def web_search(ctx, query):
    r = httpx.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
        follow_redirects=True,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    lines = []
    for res in soup.select(".result")[:5]:
        a, sn = res.select_one(".result__a"), res.select_one(".result__snippet")
        if not a or not sn:
            continue
        url, title, snippet = real_url(a.get("href", "")), a.get_text(strip=True), sn.get_text(" ", strip=True)
        sid = add_source(ctx, title, url, snippet)
        lines.append(f"[{sid}] {title} ({url}): {snippet}")
        if len(lines) == 4:
            break
    return {"observation": "\n".join(lines) or "No results.", "good": bool(lines)}


def summarise(ctx, source_id=None, text=None):
    if source_id:
        src = ctx.sources.get(source_id)
        if not src:
            return {"observation": f"Unknown source_id {source_id}. Known: {list(ctx.sources)}", "good": False}
        text = src["text"]
    if not text:
        return {"observation": "Provide source_id or text.", "good": False}
    out = ctx.llm(
        [
            {"role": "system", "content": "Summarise the text as 3-5 terse bullet points. Keep numbers and names exact. Plain text only."},
            {"role": "user", "content": text[:6000]},
        ],
        False,
    )
    ctx.notes.append(f"Summary of [{source_id}]: {out}" if source_id else f"Summary: {out}")
    return {"observation": out, "good": bool(out)}


def code_executor(ctx, code):
    with tempfile.TemporaryDirectory() as workdir:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10, cwd=workdir, env={})
        except subprocess.TimeoutExpired:
            return {"observation": "Timed out after 10s.", "good": False}
    out = p.stdout.strip()[:2000]
    if p.returncode != 0:
        return {"observation": f"Error:\n{p.stderr[-1000:]}", "good": False}
    if not out:
        return {"observation": "No output. Use print().", "good": False}
    ctx.notes.append(f"Computed with Python:\n{code}\nOutput: {out}")
    return {"observation": out, "good": True}


def structured_output(ctx, instruction, source_ids=None, format="table"):
    ids = [s for s in (source_ids or list(ctx.sources)) if s in ctx.sources]
    if not ids and not ctx.notes:
        return {"observation": "No sources or notes available yet.", "good": False}
    material = "\n\n".join(f"[{s}] {ctx.sources[s]['text'][:1200]}" for s in ids)
    shape = '{"title": str, "columns": [str], "rows": [[str]]}' if format == "table" else "a JSON object"
    data = ctx.llm(
        [
            {"role": "system", "content": f"Return {shape} that fulfils the instruction using only the material. Put citations like [S1] inside cells or values. JSON only."},
            {"role": "user", "content": f"Instruction: {instruction}\n\nMaterial:\n{material}\n\nComputed notes:\n" + "\n".join(ctx.notes)},
        ],
        True,
    )
    if format == "table" and not (isinstance(data.get("columns"), list) and isinstance(data.get("rows"), list)):
        return {"observation": 'Invalid table; need {"title","columns","rows"}.', "good": False}
    ctx.structured.append(data)
    return {"observation": json.dumps(data)[:1500], "good": True}


def memory_store(ctx, key, value):
    ctx.memory.store(ctx.session_id, str(key), str(value))
    return {"observation": f"Stored {key}.", "good": True}


def memory_recall(ctx, query="*"):
    facts = ctx.memory.recall(ctx.session_id, str(query))
    text = "\n".join(f"{f['key']}: {f['value']}" for f in facts) or "No matching facts."
    return {"observation": text, "good": True}


REGISTRY = {
    "rag_search": rag_search,
    "web_search": web_search,
    "summarise": summarise,
    "code_executor": code_executor,
    "structured_output": structured_output,
    "memory_store": memory_store,
    "memory_recall": memory_recall,
}


def run_tool(ctx, name, args):
    fn = REGISTRY.get(name)
    if not fn:
        return {"observation": f"Unknown tool {name}. Available: {list(REGISTRY)}", "good": False}
    try:
        return fn(ctx, **(args if isinstance(args, dict) else {}))
    except Exception as e:
        return {"observation": f"{type(e).__name__}: {e}", "good": False}