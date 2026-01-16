# RELISH MR Cooking Mentor (Research Prototype)

**Stack (current):** Unity 6 (URP) + Meta Quest 3 + Meta All-in-One SDK v83 + FastAPI (Python)

**Dev environment:** Windows 11 + VS Code + Anaconda + Insomnia

## What this is

A context-aware Mixed Reality cooking assistant where:

- **Cook** wears Quest 3 and cooks with an MR layer (task board, timers, checks).
- **Mentor** joins via a **simple web link** and has a **natural two-way conversation** with the cook (like a phone call).
- The system transcribes the conversation and produces MR guidance as auto-overlays (high-confidence) plus suggestion cards (tasks/timers/checks).
- The cook stays in control: **accept / dismiss / edit** suggestions.

## What this is not (Phase 1 non-goals)

- No PSTN / telephony bridging.
- Mentor does **not** see MR and does **not** operate any MR UI.
- Vision-based object grounding (RAM/SAM3, passthrough segmentation) is **Phase 2+** only.

## Prototype MVP (Phase 1)

You can run a demo where:

1. Cook starts a session on Quest and gets a join link/code.
2. Mentor opens the link in a browser and joins a voice call.
3. Hub emits transcripts + suggestion objects (tasks/timers/checks).
4. Quest renders MR suggestions and supports accept/dismiss/edit.
5. Hub logs transcripts, suggestions, and cook actions.

## Repo structure (recommended)

- `backend/` — FastAPI hub (session + WS routes + ASR + compiler + logging)
- `unity/` — Unity 6 Quest project
- `docs/` — technical spec + logging + future scope

Docs:

- `docs/SPEC.md` — the single source of truth for Phase 1 architecture, routes, messages, audio format, flows
- `docs/LOGGING.md` — what gets recorded and how
- `docs/FUTURE.md` — Phase 2+ notes (vision grounding, RAM/SAM3 setup work)

---

## Quickstart (minimal)

### 1) Backend (FastAPI hub)

Using Anaconda:

```powershell
conda create -n relish-hub python=3.11 -y
conda activate relish-hub
pip install fastapi uvicorn[standard] websockets pydantic
```

Run:

```powershell
uvicorn app:app--host0.0.0.0--port8000--reload

```

Check:

- `GET http://localhost:8000/health`

### 2) Mentor web link

Mentor uses a browser and opens the session join link served by the hub:

- `/join?session_id=...`

Mentor grants mic permissions and joins the call.

> Browser mic capture requires HTTPS except on localhost. For LAN testing, use a local TLS cert (e.g., mkcert) or a dev tunnel.
> 

### 3) Quest app

- Build/run the Unity Quest app.
- Start a session and share the join link/code with the mentor.

---

## Testing (Insomnia)

Use Insomnia for:

- WebSocket connect to the hub events channel (debug)
- Sending synthetic events (fake suggestions) to validate MR UI rendering
- Observing transcript/suggestion traffic

The exact WS routes and message formats are defined in `docs/SPEC.md`.

---

## Smoke tests (quick checks)

- Hub: `GET /health` returns 200.
- WebSocket: connect to events channel and receive a test event.
- Quest UI: render a fake suggestion list; verify accept/dismiss/edit work.
- Quest forwarding: Quest sends PCM (forwarded remote mentor WebRTC audio) to /ws/audio/{session}/mentor; hub reports bytes received.
- ASR: speaking a known sentence produces `transcript_final`.
- Compiler: transcript “Dice onions and sauté 10 minutes” yields an action + a 600s timer suggestion.
- Fallback: if compiler output is invalid, hub emits transcript-only (no crash).
- WebRTC call: mentor and cook hear each other.
- Full loop: call + transcripts + suggestions work simultaneously for ~2–3 minutes.
- UX sanity: suggestions do not flicker/reorder constantly; they batch per finalized transcript.
- Control: cook can dismiss quickly and start/edit timers quickly.

## Design principles (keep it clean)

- **One hub** in Phase 1 (FastAPI does sessions, WS routing, ASR, compiler, logs).
- **Suggestions not commands** (system proposes; cook decides).
- **MR-first UX** (calm task board + timers; minimal interruptions).
- **Defer on uncertainty** (ask the cook or keep as an ungrounded card).
- **Auto overlays for high-confidence cues** (cook can remove/edit/disable); cards for the rest.

## Decisions

- Mentor is audio-only and non-instrumented by default (no MR view, no UI controls).
- Phase 1 uses one FastAPI hub process (no microservices in the critical path).
- WebRTC is used for the two-way call; the Quest forwards the received mentor audio as PCM-over-WebSocket for ASR (mentor PCM streaming is fallback only).
- The system outputs suggestions, not commands; the cook accepts/edits/dismisses.
- Vision grounding is Phase 2+ and must never block the Phase 1 MR experience.