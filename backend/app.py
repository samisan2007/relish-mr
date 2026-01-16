from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict
import os, uuid, time

APP_VERSION = "0.1.0"
sessions: Dict[str, dict] = {}

class SessionResponse(BaseModel):
    session_id: str
    join_url: str
    created_at: int

app = FastAPI(title="RELiSH MR Hub", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"ok": True, "service": "relish-hub", "version": APP_VERSION, "t_ms": int(time.time() * 1000)}

def compute_base_url(request: Request) -> str:
    # Prefer explicit override for dev tunnels / https hostnames
    override = os.getenv("RElish_BASE_URL") or os.getenv("RELISH_BASE_URL")
    if override:
        return override.rstrip("/")
    return str(request.base_url).rstrip("/")

@app.post("/session", response_model=SessionResponse)
def create_session(request: Request):
    session_id = uuid.uuid4().hex[:10]  # short but safer than 8; OK for prototype
    created_at = int(time.time() * 1000)

    sessions[session_id] = {
        "session_id": session_id,
        "created_at": created_at,
        "state": "CREATED",
        "participants": {},
        "last_activity": created_at,
    }

    base_url = compute_base_url(request)
    join_url = f"{base_url}/join?session_id={session_id}"

    return SessionResponse(session_id=session_id, join_url=join_url, created_at=created_at)

@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.get("/join")
def join_page():
    # Serve a static page; in Section 6 you’ll add WS + WebRTC JS.
    return FileResponse("static/join.html")

@app.delete("/session/{session_id}")
def end_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions.pop(session_id)
    return {"ok": True, "session_id": session_id, "duration_ms": int(time.time() * 1000) - s["created_at"]}
