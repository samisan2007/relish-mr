# AI Copilot Instructions for RELISH MR

## Project Overview

**RELISH MR** is a Mixed Reality cooking mentor prototype where a cook on Meta Quest 3 converses naturally with a remote mentor via web link. The system transcribes conversation, generates MR suggestions (tasks/timers/checks), and lets the cook accept/dismiss/edit them.

**Stack:** Unity 6 (Quest 3) + FastAPI hub (Python) + WebRTC (bidirectional audio) + Web mentee browser

## Architecture & Core Patterns

### System Components & Data Flow

- **Cook client** (Unity/Quest 3): Runs MR UI, captures mic, decodes WebRTC audio, **forwards mentor audio as PCM-over-WebSocket** to hub for ASR
- **Mentor client** (Web browser): Joins session via URL, sends/receives WebRTC audio only (no MR, no vision)
- **Hub** (FastAPI, single-process): Sessions, WebRTC signaling relay, ASR ingest, compiler (transcripts → suggestions), logging

**Audio flow (Phase 1 default):**
Mentor speaks → Browser Web Audio API captures PCM → WebSocket to Hub → VAD + ASR → Transcript → Compiler → Suggestions → Quest renders MR
(WebRTC runs in parallel for the live call so cook hears mentor immediately; ASR runs independently on the browser-streamed PCM)

### Key Architectural Decisions

1. **One hub process** – Phase 1 keeps services simple; Vision (RAM/SAM3) is Phase 2+ in separate envs
2. **Suggestions not commands** – System proposes; cook decides (accept/dismiss/edit); no auto overlays in Phase 1
3. **Browser-direct PCM for ASR** – Mentor browser streams raw PCM to hub via Web Audio API + WebSocket. Not Quest-forwarded (that was the old default; inverted because browser path is simpler, higher quality, and independent of Quest state)
4. **Session lifecycle in-memory** – Sessions not resumable; cleanup on disconnect/timeout/explicit end
5. **Graceful degradation** – If ASR/compiler fails, call continues; hub logs errors, no crash
6. **Suggestion delta protocol** – `instruction_suggestions` adds new items; `suggestion_update` patches in place; `suggestion_remove` removes. Quest never replaces its full suggestion list — prevents flicker/reorder
7. **ASR in thread executor** – Whisper (faster-whisper) is CPU-bound; always run via `run_in_executor` to avoid blocking the asyncio event loop
8. **TURN required for remote testing** – STUN alone fails under NAT; add TURN before any user study where mentor is off-LAN

### Expected Latency Profile

Speech → MR suggestion: **~2.6 seconds** (2.0s ASR buffer + 0.5s ASR + 0.1s network/compile). Cook hears audio immediately; suggestion appears ~2.5s later. Acceptable for non-urgent instructions.

## Repository Structure & Key Files

- **[SPEC.md](SPEC.md)** – Single source of truth for Phase 1 architecture, routes, message contracts, audio format, session lifecycle, latency expectations
- **[FUTURE.md](FUTURE.md)** – Phase 2+ vision grounding (RAM/SAM3 setup notes); not blocking Phase 1
- **[README.md](README.md)** – Quickstart, testing, smoke tests
- **`backend/`** – FastAPI hub — implement WS routes, ASR (faster-whisper), compiler (Claude Haiku), logging here
  - `backend/app.py` – hub skeleton (REST routes done; WS routes and pipeline not yet implemented)
  - `backend/static/join.html` – mentor web UI (to be created: WebRTC + Web Audio API PCM streaming)
- **`unity/`** – Unity 6 Quest 3 project (to be created)
- **`services/ram/recognize-anything/`** – Phase 2+ vision grounding (RAM library cloned; weights not yet downloaded)
- **`services/sam3/`** – Phase 2+ segmentation stub (empty)
- **`shared/`** – Message schema (JSON Schema for hub ↔ client contract; to be populated)

## Critical Developer Workflows

### Backend Setup & Testing

```powershell
# Hub environment (Phase 1)
conda create -n relish-hub python=3.11 -y
conda activate relish-hub
pip install fastapi uvicorn[standard] websockets pydantic

# Run hub
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Check health
GET http://localhost:8000/health
```

### Testing with Insomnia

- WebSocket debug: Connect to `/ws/events/{session_id}/debug` to observe transcripts & suggestions
- Synthetic events: Send fake suggestions to test Quest MR UI rendering
- Message formats defined in **SPEC.md** section 5

### Smoke Tests (Quick Validation)

1. Hub health endpoint returns 200
2. WebSocket connection to events channel receives test event
3. Quest UI renders fake suggestion list; accept/dismiss/edit work
4. Quest PCM forwarding to `/ws/audio/{session}/mentor` shows bytes received
5. Speaking produces `transcript_final` from ASR
6. Transcript → suggestion compilation works (e.g., "Dice onions and sauté 10 min" yields action + 600s timer)
7. Compiler failure doesn't crash session (transcript-only fallback)
8. Full loop stable for 2–3 minutes without flickering/reordering suggestions
9. Cook can dismiss & edit timers quickly

## WebSocket Routes & Message Contract

### Three WS Routes
- **`/ws/signal/{session_id}/{role}`** – WebRTC signaling (offer/answer/ICE) relay only
- **`/ws/events/{session_id}/{client}`** – JSON messages (transcripts, suggestions, session events)
- **`/ws/audio/{session_id}/{speaker}`** – Binary PCM frames; `speaker=mentor` fed by browser Web Audio API; `speaker=cook` optionally fed by Quest

### Message Envelope (All JSON)
```json
{ "type": "...", "session_id": "...", "t_ms": <epoch_ms>, "seq": <optional_int>, "v": "0.1" }
```

**Key events:**
- `transcript_final` – Hub → Cook (emit suggestions only from this, not partials)
- `instruction_suggestions` – Hub → Cook (ADD new suggestion items only; append to pending list)
- `suggestion_update` – Hub → Cook (PATCH existing suggestion in place by ID)
- `suggestion_remove` – Hub → Cook (REMOVE suggestion from UI by ID)
- `participant_joined` / `participant_left` – Session notifications
- `session_end` – Cook → Hub (explicit cleanup trigger)

### Audio Format
- **Preferred:** 16 kHz, mono, PCM int16
- **Accepted:** 48 kHz, mono/stereo, int16 (hub downmixes to 16kHz mono before ASR)
- **Chunking:** VAD-based utterance chunking (not fixed 2–5s buffer); flush on end-of-speech (500ms silence), hard cap 10s
- **ASR:** faster-whisper `base` model (default); always run in `asyncio.run_in_executor` — never block the event loop

## Session Lifecycle & Cleanup

**States:** CREATED → COOK_READY → ACTIVE → ENDING → ENDED

**Cleanup triggers:**
- Explicit: Cook sends `{"type":"session_end"}`
- Timeout: 30 minutes inactivity
- Disconnect: No reconnect within 5 minutes

**Cleanup procedure:** Close WSs, flush logs (JSONL), delete in-memory state. Logs persist; state not resumable.

## Code Patterns & Conventions

1. **Logging to JSONL** – Use epoch milliseconds for timestamps (simple, log-friendly)
2. **Session ID strings** – Use for routing; can be generated client-side or via REST `/session`
3. **Compiler input** – `transcript_final` text only; output is JSON list of suggestions with `type`, `needs_grounding`, `priority`
4. **Vision grounding decision:** 
   - Only invoked when cook **accepts** a suggestion with `needs_grounding=true`
   - Never block UX on vision failure; suggestion remains normal MR card if grounding unavailable
5. **Fallback handling:** ASR/compiler failures log errors but don't crash session; emit transcript-only or skip suggestion
6. **WebRTC:** Relay ICE/offer/answer through hub; Quest and mentor peer directly for media

## Phase 2+ Vision Integration (Not Phase 1)

**Keep isolated:** Vision deps in separate envs (`relish-ram` with PyTorch+CUDA 12.6). Do NOT pollute Phase 1 hub env.

**RAM (Recognize Anything) setup notes in FUTURE.md:**
- Python 3.10 + CUDA 12.6 wheels
- Clone https://github.com/xinyu1205/recognize-anything.git
- Weights stored in `services/ram/weights/`

**SAM3 notes:** Optional; use only if Phase 2 adds per-object segmentation.

## Quick Debugging & Tips

- **Hub won't connect?** Check port 8000; verify `--host 0.0.0.0` for network access
- **WebRTC no audio?** Verify signaling messages (offer/answer/ICE) flow correctly via hub; check TURN config for off-LAN testing
- **Suggestions don't appear?** Check `transcript_final` is firing; verify faster-whisper is running in executor (not blocking event loop); verify compiler returns valid JSON
- **Compiler returns garbage?** Ensure structured output (tool use / JSON mode) is configured — never parse free-form LLM text
- **Suggestion list flickers/reorders?** Verify Quest appends on `instruction_suggestions`, patches on `suggestion_update`, removes on `suggestion_remove` — never replaces full list
- **ASR hub lag?** VAD-based chunking means latency scales with utterance length; typical ~1.5s utterance + 500ms silence = ~2s before ASR starts. This is expected.
- **Session cleanup hangs?** Check for lingering WebSocket connections; verify timeout logic fires
- **Mentor browser mic perms?** HTTPS required (except localhost); use mkcert for LAN testing; check AudioWorklet vs ScriptProcessorNode compatibility

## References

- Read **SPEC.md** for full message contracts and route details
- See **README.md** quickstart for commands
- Check **FUTURE.md** for vision grounding roadmap
- Insomnia collection: Test WebSockets and synthetic payloads without rebuilding UI
