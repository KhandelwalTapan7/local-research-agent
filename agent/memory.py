import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).resolve().parent.parent / "runtime" / "agent.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, session_id TEXT, query TEXT, answer TEXT, created REAL);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, session_id TEXT, type TEXT, payload TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS facts(session_id TEXT, key TEXT, value TEXT, ts REAL, PRIMARY KEY(session_id, key));
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


class Memory:
    def start_run(self, session_id, query):
        run_id = uuid.uuid4().hex
        with connect() as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (run_id, session_id, query, None, time.time()))
        return run_id

    def log_event(self, run_id, session_id, event):
        with connect() as c:
            c.execute(
                "INSERT INTO events(run_id, session_id, type, payload, ts) VALUES (?,?,?,?,?)",
                (run_id, session_id, event["type"], json.dumps(event), time.time()),
            )

    def finish_run(self, run_id, answer):
        with connect() as c:
            c.execute("UPDATE runs SET answer=? WHERE run_id=?", (json.dumps(answer), run_id))

    def history(self, session_id):
        with connect() as c:
            runs = c.execute("SELECT * FROM runs WHERE session_id=? ORDER BY created", (session_id,)).fetchall()
            out = []
            for r in runs:
                events = c.execute("SELECT payload FROM events WHERE run_id=? ORDER BY id", (r["run_id"],)).fetchall()
                out.append({
                    "run_id": r["run_id"],
                    "query": r["query"],
                    "created": r["created"],
                    "answer": json.loads(r["answer"]) if r["answer"] else None,
                    "events": [json.loads(e["payload"]) for e in events],
                })
        return out

    def recent_turns(self, session_id, n=3):
        with connect() as c:
            rows = c.execute(
                "SELECT query, answer FROM runs WHERE session_id=? AND answer IS NOT NULL ORDER BY created DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        return [{"query": r["query"], "answer": json.loads(r["answer"]).get("answer", "")} for r in reversed(rows)]

    def store(self, session_id, key, value):
        with connect() as c:
            c.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?,?)", (session_id, key, value, time.time()))

    def recall(self, session_id, query="*"):
        with connect() as c:
            rows = c.execute("SELECT key, value FROM facts WHERE session_id=? ORDER BY ts", (session_id,)).fetchall()
        tokens = re.findall(r"\w+", query.lower())
        if not tokens or "*" in query:
            return [dict(r) for r in rows]
        return [dict(r) for r in rows if any(t in f"{r['key']} {r['value']}".lower() for t in tokens)]
