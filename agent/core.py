import json
import os
import re

import httpx

from agent import tools

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
MAX_RETRIES = 2
MAX_PLAN_STEPS = 6
MIN_DISTINCT_TOOLS = 3

PADDING = [
    ("summarise", "Compress the most relevant retrieved source into key points"),
    ("structured_output", "Format the key findings as a comparison table"),
]

DOMAIN = (
    "You are a research agent specialised in vector databases and similarity-search infrastructure "
    "(Qdrant, Chroma, Pinecone, pgvector, FAISS, Weaviate, Milvus, HNSW/IVF/PQ indexing, memory sizing). "
    "The local knowledge base covers these topics; web_search is for anything it lacks."
)

PLAN_PROMPT = f"""{DOMAIN}
Break the user's question into 3-6 ordered steps. Each step uses exactly one tool.
Tools:
{tools.SPECS}
Rules: retrieve before anything else; use at least 3 distinct tools; use code_executor for any arithmetic;
use structured_output when comparing options; use memory_store when the user states a fact or preference to keep;
use memory_recall when the question refers to earlier context.
Respond with JSON: {{"steps": [{{"goal": str, "tool": str}}]}}"""

ACT_PROMPT = f"""{DOMAIN}
You execute one step of a plan. Choose the arguments for the tool call that accomplishes the current step.
Tools:
{tools.SPECS}
Use source ids (S1, S2...) exactly as listed in the state. Write self-contained Python for code_executor.
Respond with JSON: {{"thought": str, "tool": str, "args": object}}"""

REFLECT_PROMPT = f"""{DOMAIN}
The last tool call produced a poor or failed result. Diagnose why and revise the strategy: rephrase the query
with different keywords, switch tool (for example rag_search to web_search), or fix the code using the error.
Tools:
{tools.SPECS}
Respond with JSON: {{"reflection": str, "tool": str, "args": object}}"""

SYNTH_PROMPT = """You write the final answer of a research agent.
Use only the numbered sources and computed notes provided. After every factual claim append citations such as [S1] [S3],
each id in its own brackets and taken from the source list. Use short paragraphs and '- ' bullets.
Report computed values as computed. Tables already attached are shown below the answer; refer to them instead of repeating them.
If the sources do not support part of the question, say so plainly.
Respond with JSON: {"answer": str}"""


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Model returned no JSON: {text[:200]}")
        return json.loads(match.group(0))


def chat(messages, json_mode=True):
    payload = {"model": MODEL, "messages": messages, "stream": False, "options": {"temperature": 0.2, "num_ctx": 8192}}
    if json_mode:
        payload["format"] = "json"
    for attempt in range(2):
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()
        if not json_mode:
            return content
        try:
            data = parse_json(content)
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    raise ValueError("Model failed to return valid JSON")


class Agent:
    def __init__(self, memory):
        self.memory = memory

    def run(self, query, session_id):
        ctx = tools.Context(session_id, self.memory, chat)
        history = self.conversation(session_id)
        try:
            plan = self.plan(query, history)
            yield {"type": "plan", "steps": plan}
            log, used = [], set()
            for i, step in enumerate(plan, 1):
                yield {"type": "step", "index": i, "goal": step["goal"]}
                yield from self.execute(ctx, query, plan, step, log, used)
            yield {"type": "final", **self.synthesise(ctx, query, history)}
        except httpx.ConnectError:
            yield {"type": "error", "message": f"Cannot reach Ollama at {OLLAMA_URL}. Start it and pull {MODEL}."}
        except Exception as e:
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}

    def conversation(self, session_id):
        turns = self.memory.recent_turns(session_id)
        return "\n".join(f"User: {t['query']}\nAgent: {t['answer'][:400]}" for t in turns)

    def plan(self, query, history):
        data = chat([
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": f"Earlier turns:\n{history or 'none'}\n\nQuestion: {query}"},
        ])
        steps = [
            {"goal": str(s["goal"]), "tool": s["tool"]}
            for s in data.get("steps", [])
            if isinstance(s, dict) and s.get("goal") and s.get("tool") in tools.REGISTRY
        ][:MAX_PLAN_STEPS]
        steps = [s for s in steps if s["tool"] != "memory_store" or re.search(r"remember|note that|keep in mind", query, re.I)]
        steps = [s for s in steps if s["tool"] != "memory_recall" or history]
        if not any(s["tool"] in ("rag_search", "web_search") for s in steps):
            steps.insert(0, {"goal": "Retrieve documents relevant to the question", "tool": "rag_search"})
        if re.search(r"comput|calculat|how much|\d", query) and not any(s["tool"] == "code_executor" for s in steps):
            steps.insert(1, {"goal": "Compute the requested numbers with Python", "tool": "code_executor"})
        names = {s["tool"] for s in steps}
        for tool, goal in PADDING:
            if len(names) >= MIN_DISTINCT_TOOLS:
                break
            if tool not in names:
                steps.append({"goal": goal, "tool": tool})
                names.add(tool)
        return steps

    def state(self, ctx, query, plan, step, log):
        plan_text = "\n".join(f"{i}. [{s['tool']}] {s['goal']}" for i, s in enumerate(plan, 1))
        sources = "\n".join(f"{s['id']}: {s['title']}" for s in ctx.sources.values()) or "none yet"
        return (
            f"Question: {query}\n\nPlan:\n{plan_text}\n\nCurrent step: [{step['tool']}] {step['goal']}\n\n"
            f"Sources so far:\n{sources}\n\nProgress:\n{chr(10).join(log) or 'nothing yet'}"
        )

    def execute(self, ctx, query, plan, step, log, used):
        state = self.state(ctx, query, plan, step, log)
        action = chat([{"role": "system", "content": ACT_PROMPT}, {"role": "user", "content": state}])
        tool = step["tool"]
        if action.get("tool") != tool:
            action = chat([
                {"role": "system", "content": ACT_PROMPT},
                {"role": "user", "content": f"{state}\n\nYou must use the tool {tool} for this step."},
            ])
        args = action.get("args") if isinstance(action.get("args"), dict) and action.get("tool") == tool else {}
        yield {"type": "thought", "text": str(action.get("thought", ""))}
        attempt = 0
        tried = {(tool, json.dumps(args, sort_keys=True))}
        while True:
            yield {"type": "tool_call", "tool": tool, "args": args}
            res = tools.run_tool(ctx, tool, args)
            used.add(tool)
            yield {"type": "tool_result", "tool": tool, "observation": res["observation"][:1200], "good": res["good"]}
            if res["good"] or attempt >= MAX_RETRIES:
                break
            attempt += 1
            fix = chat([
                {"role": "system", "content": REFLECT_PROMPT},
                {"role": "user", "content": f"{state}\n\nFailed call: {tool} {json.dumps(args)}\nObservation:\n{res['observation'][:1500]}\n\nDo not repeat these failed calls: {json.dumps(sorted(tried))}"},
            ])
            tool = fix.get("tool") if fix.get("tool") in tools.REGISTRY else tool
            args = fix.get("args") if isinstance(fix.get("args"), dict) else args
            yield {"type": "reflection", "text": str(fix.get("reflection", "")), "tool": tool, "args": args}
            key = (tool, json.dumps(args, sort_keys=True))
            if key in tried:
                break
            tried.add(key)
        status = "OK" if res["good"] else "POOR"
        log.append(f"{step['goal']} -> {tool}({json.dumps(args)[:200]}) [{status}]: {res['observation'][:600]}")

    def synthesise(self, ctx, query, history):
        sources = "\n\n".join(f"[{s['id']}] {s['title']} ({s['location']}): {s['text'][:900]}" for s in ctx.sources.values())
        user = (
            f"Earlier turns:\n{history or 'none'}\n\nQuestion: {query}\n\nSources:\n{sources or 'none'}\n\n"
            f"Computed notes:\n{chr(10).join(ctx.notes) or 'none'}\n\nAttached tables:\n{json.dumps(ctx.structured) if ctx.structured else 'none'}"
        )
        feedback, answer, cited = "", "", []
        for _ in range(2):
            data = chat([{"role": "system", "content": SYNTH_PROMPT}, {"role": "user", "content": user + feedback}])
            answer = re.sub(
                r"\[(S\d+(?:\s*,\s*S\d+)+)\]",
                lambda m: " ".join(f"[{s.strip()}]" for s in m.group(1).split(",")),
                str(data.get("answer", "")),
            )
            answer = re.sub(r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in ctx.sources else "", answer)
            cited = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", answer)))
            if cited or not ctx.sources:
                break
            feedback = "\n\nYour previous answer had no valid [S#] citations. Rewrite it citing the provided sources."
        sources_out = [{**ctx.sources[s], "text": ctx.sources[s]["text"][:500]} for s in cited]
        return {"answer": answer, "structured": ctx.structured, "sources": sources_out}