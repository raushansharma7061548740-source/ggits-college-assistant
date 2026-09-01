"""
api.py
------
FastAPI wrapper around the GGITS College Assistant LangGraph app.

Run with:
    uv run uvicorn api:app --reload --port 8000

Endpoints:
    GET  /api/programmes   -> list of programmes the student can select
    POST /api/chat         -> send a message, get the assistant's reply
    GET  /api/health       -> simple health check
"""

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_graph import compiled_app, PROGRAMME_MAP

app = FastAPI(title="GGITS College Assistant API")

# Allow the frontend (served from a different origin/port, e.g. file:// or
# a dev server on :5500/:3000) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    programme: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    category: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/programmes")
def programmes():
    """Returns the programme choices, in the same order as the original CLI menu."""
    return [{"code": code, "name": name} for code, name in PROGRAMME_MAP.items()]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = compiled_app.invoke(
            {
                "programme": req.programme,
                "message": [("human", req.message)],
            },
            config=config,
        )
    except Exception as exc:  # surface a clean error to the frontend
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    reply = result["message"][-1].content
    category = result.get("query_type", "general")

    return ChatResponse(session_id=session_id, reply=reply, category=category)
