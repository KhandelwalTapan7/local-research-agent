import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.core import Agent
from agent.memory import Memory
from rag.retriever import get_retriever

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
memory = Memory()
agent = Agent(memory)


@asynccontextmanager
async def lifespan(app):
    get_retriever()
    yield


app = FastAPI(title="Vector DB Research Agent", lifespan=lifespan)


class RunRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(min_length=1, max_length=64)


@app.post("/run")
def run(req: RunRequest):
    run_id = memory.start_run(req.session_id, req.query)

    def stream():
        for event in agent.run(req.query, req.session_id):
            memory.log_event(run_id, req.session_id, event)
            if event["type"] == "final":
                memory.finish_run(run_id, event)
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/history/{session_id}")
def history(session_id: str):
    return {"session_id": session_id, "runs": memory.history(session_id)}


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
