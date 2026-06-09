```markdown
# SPEC — Phase 1 (Audio-First + MR Suggestions)
**Last updated:** 2026-06-09
**Audience:** You (build + iterate fast)
**Goal:** One clean source of truth for Phase 1 system behavior.

---

## 1) Scope and principles

### 1.1 What we’re building
A Quest 3 MR cooking companion that augments a cook during a natural, two-way conversation with a remote mentor by:
- transcribing relevant speech
- compiling it into **suggestions** (tasks/timers/checks/warnings)
- presenting those suggestions as a calm MR layer
- letting the cook accept/dismiss/edit at any time

### 1.2 Hard constraints (Phase 1)
- Mentor is **non-instrumented**: no MR view, no MR UI controls.
- No PSTN/telephony bridging.
- Prototype architecture must stay simple: **one FastAPI hub**.

### 1.3 Non-goals (Phase 1)
- Vision grounding (RAM/SAM3, passthrough segmentation, object highlights anchored to real objects).
- Server-side WebRTC media termination.
- Complex identity/auth; session IDs are sufficient for local testing.

---

## 2) System components

### 2.1 Cook client (Unity 6 / Quest 3)
Responsibilities:
- MR UI: task board, timers, transcript (optional)
- Cook actions: accept/dismiss/edit/mark-done
- WebRTC peer: send cook mic, receive mentor audio
- WebSocket peers: signaling + events
- **Optional (recommended):** Stream cook mic audio as PCM to `/ws/audio/{session}/cook` for full dialogue context in the compiler. Uses the same audio thread → ConcurrentQueue → WebSocket pattern as any PCM forwarding.

### 2.1.1 Quest audio forwarding (Phase 2 optional enhancement)

Re-forwarding the received mentor WebRTC track from Quest to the hub is **not required in Phase 1**. The mentor browser streams PCM directly (see §2.2). This section describes the Phase 2 path if you want Quest-side audio processing (e.g., spatial audio normalization) before ASR.

**Implementation note (if built):** Unity's `OnAudioFilterRead` runs on the audio thread — not the main thread or any coroutine. Never call WebSocket send directly from it. Use a `ConcurrentQueue<byte[]>` that `OnAudioFilterRead` enqueues into; drain the queue from a background `Task` that writes to the WebSocket.

Quest would forward mentor audio when:
- WebRTC peer connection state is `connected`
- Remote audio track is received and playing
- Hub WebSocket (`/ws/audio/...`) is connected

### 2.2 Mentor client (Web link)
Responsibilities:
- Join by URL/session code
- WebRTC peer: send mentor mic, receive cook audio
- **Stream PCM directly to hub for ASR (Phase 1 default):** mentor browser captures mic via Web Audio API (ScriptProcessorNode or AudioWorklet) and streams raw PCM over WebSocket to `/ws/audio/{session}/mentor`. This is the primary ASR path — it is simpler, higher quality (no Opus round-trip), and independent of Quest connection state.
- Minimal UI: Join, call status, transcript echo (see §5.3.1)

### 2.3 Hub (FastAPI, Python)
Single-process hub does:
- sessions (create/join/leave)
- WebRTC signaling relay (offer/answer/ICE)
- PCM ingest → ASR → transcript events
- transcript → compiler → suggestion events
- logging to disk (JSONL)
- PCM ingest source (Phase 1 default): mentor browser streams PCM directly to `/ws/audio/{session}/mentor` via Web Audio API

---

## 3) Real-time channels

### 3.1 WebRTC (Cook ↔ Mentor)
Purpose: real conversational call (two-way audio).
Signaling: via hub WebSocket (below).
Media: audio only.

**ICE server configuration (required for non-LAN testing):**

STUN alone is not sufficient when peers are on different networks behind NAT. TURN is required for user studies where mentor is remote.

Hub must return ICE server config when serving the join page (or via a `/ice-servers` endpoint):

```json
{
  "iceServers": [
    { "urls": "stun:stun.l.google.com:19302" },
    { "urls": "turn:<host>:3478", "username": "<user>", "credential": "<pass>" }
  ]
}
```

**TURN server options:**
- **Dev/prototype:** [Metered TURN](https://www.metered.ca/tools/openrelay/) — free tier, no setup
- **Self-hosted:** coturn (`docker run coturn/coturn`)
- **Production:** Twilio Network Traversal, Cloudflare TURN

For LAN-only testing (Quest and mentor PC on same network/subnet), STUN alone is sufficient. Add TURN before any remote user study.

### 3.2 WebSockets (clients ↔ hub)
We use three WS routes:
- signaling (WebRTC negotiation)
- events (JSON messages)
- audio ingest (binary PCM to ASR)
Phase 1 default: `/ws/audio/{session}/mentor` is fed by the mentor browser streaming PCM directly via Web Audio API. `/ws/audio/{session}/cook` is optionally fed by the Quest streaming cook mic PCM.

---

## 4) Hub endpoints

### 4.1 REST (optional but recommended)
#### `GET /health`
Returns: `{ "ok": true }`

#### `POST /session`
Creates a session.
Returns: `{ "session_id": "...", "join_url": "..." }`

> If you want ultra-minimal, you can generate `session_id` client-side. REST just makes logging + lifecycle cleaner.

### 4.2 WebSockets
#### `WS /ws/signal/{session_id}/{role}`
- `role`: `cook | mentor`
- Payload: WebRTC negotiation messages only (`offer`, `answer`, `ice`)
- Hub behavior: relay messages between peers

#### `WS /ws/events/{session_id}/{client}`
- `client`: `cook | mentor | debug`
- Hub → Cook: transcripts, suggestions, clarification prompts, participant join/leave
- Cook → Hub: accept/dismiss/edit responses, mark done

#### `WS /ws/audio/{session_id}/{speaker}`
- `speaker`: `mentor | cook`
- Payload: binary PCM frames (see section 6)
- Hub behavior: VAD → buffer utterance → ASR → emit transcript events

Phase 1:
- `speaker=mentor`: mentor browser streams PCM directly via Web Audio API (primary ASR path).
- `speaker=cook`: Quest streams cook mic PCM (optional; gives compiler full dialogue context).
## 4.3 Session lifecycle and cleanup

### States:
- `CREATED`: POST /session returned session_id, no clients connected
- `COOK_READY`: cook connected to signaling + events, awaiting mentor
- `ACTIVE`: both peers connected, WebRTC established, audio flowing
- `ENDING`: one peer disconnected or explicit end requested
- `ENDED`: session terminated, resources cleaned up

### Cleanup triggers:
- Explicit: Cook sends `{"type":"session_end"}` → immediate cleanup
- Timeout: No activity for 30 minutes → automatic cleanup
- Disconnect: If either peer disconnects and doesn't reconnect within 5 minutes → cleanup

### Cleanup procedure:
1. Close all WebSocket connections for this session
2. Flush transcript and suggestion logs to disk (JSONL)
3. Delete in-memory session state (audio buffers, WebRTC signaling)
4. Emit `session_ended` event to any remaining connected clients

### Persistence:
- Logs persist on disk (for research analysis)
- Session state is in-memory only (sessions are not resumable after cleanup)

## 4.5 End-to-end sequences (short)
- Session: cook creates a session → hub returns `session_id` + join URL/code → cook shares link/QR.
- Join: mentor opens link → connects to signaling WS and (optionally) events WS → hub notifies cook that mentor joined.
- Call setup: cook and mentor exchange WebRTC `offer/answer/ice` via hub signaling WS → two-way audio established.
- ASR ingest (mentor, default): mentor browser streams PCM directly to `/ws/audio/{session}/mentor` → hub applies VAD, buffers utterance → ASR emits `transcript_final`.
- ASR ingest (cook, optional): Quest streams cook mic PCM to `/ws/audio/{session}/cook` → hub transcribes and labels speaker `cook` → compiler has full dialogue context.

- Compilation: on each `transcript_final`, hub runs compiler (LLM + rules) → emits `instruction_suggestions` to cook.
- MR loop: cook accepts/dismisses/edits suggestions → hub logs actions; accepted timers start or prompt for duration (per UX choice).
- Fallback: if ASR or compiler fails, the call continues; hub emits transcript-only and logs errors (no session crash).

### 4.5.1 Expected latency profile (Phase 1, browser-direct PCM)

Typical timeline from mentor speaks → cook sees suggestion:

T=0ms:    Mentor speaks into mic
T=20ms:   Browser Web Audio API captures PCM frame (~20ms frame interval)
T=20ms:   Browser sends PCM directly over WebSocket to hub (parallel to WebRTC)
T=30ms:   Hub receives PCM (LAN: ~10ms)
T=30ms:   Browser also encodes WebRTC audio (Opus) → Quest
T=75ms:   Cook HEARS audio (Quest WebRTC decode + audio output)
           [Hub has been accumulating PCM since T=30ms]
T=1530ms: VAD detects end-of-utterance (typical ~1.5s utterance + 500ms silence)
T=2030ms: ASR completes (faster-whisper base: ~500ms)
T=2045ms: Compiler runs (~15ms LLM API call excluded; add ~300ms for API)
T=2360ms: Hub emits instruction_suggestions
T=2370ms: Cook sees MR suggestion card

Total: ~2.4 seconds from speech to suggestion (utterance-length dependent)
Breakdown: ~1.5s utterance + 0.5s VAD silence + 0.5s ASR + 0.3s compiler + 0.1s network

Key differences from Quest-forwarding path:
- No Opus encode/decode round-trip → cleaner audio for ASR
- Hub gets audio at T=30ms (not T=85ms via Quest) → marginally earlier buffering
- No dependency on Quest connection state for ASR to function
- Latency scales with utterance length, not a fixed buffer window

Cook heard the audio at T=75ms, saw suggestion at T=2370ms
→ Suggestion appears ~2.3 seconds after cook heard it (for a typical 1.5s utterance)

For user studies: This delay is expected and acceptable for non-urgent instructions.
For urgent/safety items: Mentor should repeat or confirm immediately.
---

## 5) Message contract (events)

### 5.1 Common fields (JSON envelope)
All JSON messages MUST include:
- `type` (string)
- `session_id` (string)
- `t_ms` (int)

Optional (recommended):
- `seq` (int): per-connection incrementing sequence number
- `v` (string): `"0.1"`

**Timestamp convention**
- Use epoch milliseconds for `t_ms` (simple, log-friendly).

### 5.2 Signaling messages (relay-only)
```json
{ "type":"offer",  "session_id":"abc", "t_ms":0, "sdp":"..." }
{ "type":"answer", "session_id":"abc", "t_ms":0, "sdp":"..." }
{ "type":"ice",    "session_id":"abc", "t_ms":0, "candidate":{ "...":"..." } }

```

### 5.3 Session events (Hub → clients)

```json
{ "type":"session_created", "session_id":"abc", "t_ms":0, "join_url":"https://.../join?session_id=abc" }
{ "type":"participant_joined", "session_id":"abc", "t_ms":10, "role":"mentor" }
{ "type":"participant_left",   "session_id":"abc", "t_ms":20, "role":"mentor" }

```

### 5.3.1 Mentor feedback echo (Phase 1)

Hub emits to mentor's events WebSocket after each compilation. This is a single extra `send` call per compiled suggestion — implement it in Phase 1.

```json
{
  "type":"instruction_suggestions_echo",
  "session_id":"abc",
  "t_ms":300,
  "items":[...],
  "note":"These suggestions were sent to the cook"
}
```

Mentor UI displays:
- Last transcript: "You said: 'Dice the onions'"
- Compiled: "→ Action card: 'Dice the onions'"
- Cook status: "Accepted" | "Dismissed" | "Pending"

Rationale: Allows mentor to verify the system understood them correctly and adjust language if needed. Directly supports research question RQ3 (does mentor adapt communication style in response to system feedback?). Cost is negligible — one extra WebSocket send per compiled suggestion.


### 5.4 Transcript events (Hub → Cook; optional to mentor/debug)

```json
{ "type":"transcript_partial", "session_id":"abc", "t_ms":100, "speaker":"mentor", "text":"dice the..." }
{ "type":"transcript_final",   "session_id":"abc", "t_ms":200, "speaker":"mentor", "text":"Dice the onions." }

```

---

## 6) Audio format (PCM ingest)

### 6.1 PCM spec (Phase 1 default)

Transport payload is PCM int16 little-endian, chunked in ~20–50 ms frames.

Preferred (ASR-friendly):

- 16 kHz, mono, PCM int16

Accepted (common from WebRTC/Unity):

- 48 kHz, mono or stereo, PCM int16

If input is not 16 kHz mono, the hub downmixes/downsamples to 16 kHz mono before ASR.

### 6.1.1 Mentor browser PCM capture (Phase 1 implementation note)

The mentor browser uses the Web Audio API to capture raw PCM and send it to the hub:

- `AudioContext.createScriptProcessorNode(bufferSize=4096)` or an `AudioWorkletProcessor` (preferred for modern browsers — avoids main-thread blocking)
- At 48kHz: buffer of 4096 samples = ~85ms of audio; downsample to 16kHz before sending, or send as-is and let the hub resample
- Send each buffer as a single binary WebSocket message (no batching — minimize latency)
- Typical send rate with 4096-sample buffers at 48kHz: ~12 messages/second (with hub resampling) or ~50/s at 16kHz with smaller buffers

**HTTPS requirement:** Browser mic access requires a secure context. On localhost this is automatic. For LAN or remote testing, use `mkcert` for a self-signed cert or a dev tunnel (ngrok, Cloudflare Tunnel).

### 6.1.2 Quest cook audio streaming (optional Phase 1)

If streaming cook audio for full dialogue context, Unity's `OnAudioFilterRead` provides mic buffers at ~20ms intervals:

- At 48kHz mono: ~960 samples/buffer = ~3840 bytes (int16)
- **Thread safety:** `OnAudioFilterRead` runs on Unity's audio thread. Do NOT call WebSocket send from it. Use a `ConcurrentQueue<byte[]>` that `OnAudioFilterRead` enqueues into; a background `Task` drains the queue to the WebSocket.
- Drop oldest frames on network congestion; log dropped frame count.

### 6.2 ASR implementation

**Library:** `faster-whisper` (CTranslate2-optimized Whisper; pip: `faster-whisper`). Significantly faster than `openai-whisper` on both CPU and GPU.

**Model choice:**
- Default: `base` (~74M params) — fast on CPU, good English accuracy
- Better accuracy: `small` (~244M params) — use if inference time is acceptable
- GPU available: `large-v3` — best accuracy, ~500ms on A10/A100

**Language:** Always set `language="en"` — skips language detection, saves ~100ms.

**Event loop safety:** Whisper inference is CPU-bound and will block the asyncio event loop if called directly. Always run in a thread pool executor:

```python
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(asr_executor, transcribe_sync, audio_bytes)
```

Use a dedicated `ThreadPoolExecutor(max_workers=1)` for the ASR model instance (it is not thread-safe for concurrent calls).

**Kitchen noise:** Whisper handles moderate background noise well, but loud continuous noise (running water, exhaust fans) will degrade accuracy. VAD (§6.3) prevents transcribing silence and filters the worst cases. If accuracy is poor in a real kitchen, try `small` or `large-v3`.

### 6.3 VAD and utterance-level chunking

Do not buffer audio by fixed duration. Instead, use Voice Activity Detection to chunk at natural speech boundaries:

**Library:** `silero-vad` (pip: `silero-vad`) — ~1ms per 30ms audio chunk on CPU.

**Chunking rules:**
- Start buffering on voice onset (VAD transitions to speech).
- Flush buffer to ASR when VAD detects end-of-speech (500ms of continuous silence).
- Hard cap: flush at 10s regardless of VAD state (prevents unbounded accumulation on continuous speech).
- Floor: discard chunks shorter than 200ms (noise bursts, not speech).

**Why:** Fixed 2–5s buffers transcribe silence (Whisper hallucinates on silence — known issue), split sentences mid-thought, and add unnecessary latency for short utterances. Utterance-level chunking gives complete sentences at the natural pause boundary.

**Rule:** compile suggestions from `transcript_final` only (never from partials, to avoid UI thrash).

---

## 7) Suggestion model (instruction compiler output)

### 7.0 Interaction policy

**Phase 1 (this spec):** All compiler output appears as **suggestion cards** on the task board. There are no auto-rendered overlays in Phase 1. The cook explicitly accepts, edits, or dismisses every suggestion.

- **Suggestions (cook chooses):** All items appear as cards (Accept/Edit/Dismiss). High-confidence items are visually distinguished (e.g., bolder style) but still require explicit action.
- **Confirm-first:** Safety-critical items (`kind: warning`) display with a confirmation prompt before logging the accept.

**Phase 2:** Auto overlays will be introduced once the UI pattern is validated and world anchoring (vision grounding) is available. See §13 and FUTURE.md.

### 7.1 Philosophy

The system outputs **suggestions**, not commands. The cook retains full agency at every step. This is a deliberate research and safety choice — the system's role is to reduce cognitive load, not to act autonomously.

### 7.2 Suggestion object (v0.1)

`kind` is one of:

- `action` (do something)
- `timer` (time-bound process)
- `check` (verify a condition)
- `warning` (risk/safety)
- `info` (contextual note)
- `question` (prompt cook for info)
- `sequence` (multi-step chunk)

Common fields:

- `id` (string)
- `kind` (string)
- `text` (string)
- `confidence` (0–1)
- `needs_grounding` (bool; usually false in Phase 1)

Optional:

- `duration_sec` (timer)
- `entities`: `{ "ingredient": [...], "tool": [...] }`
- `temporal`: `now | after | while | until`
- `depends_on`: `["task_id"]`

### 7.3 Suggestion delta protocol

The hub uses three distinct event types for suggestion lifecycle. The Quest client processes each event type independently; it does not replace its full suggestion list.

**`instruction_suggestions` (Hub → Cook) — ADD new items**

Emitted after each `transcript_final`. Contains only newly extracted suggestions. The Quest appends these to its pending list.

```json
{
  "type":"instruction_suggestions",
  "session_id":"abc",
  "t_ms":300,
  "items":[
    { "id":"sug_001", "kind":"action", "text":"Dice the onions", "confidence":0.82, "needs_grounding":false,
      "entities":{"ingredient":["onion"],"tool":[]}, "temporal":"now" },
    { "id":"sug_002", "kind":"timer", "text":"Sauté for 10 minutes", "duration_sec":600,
      "confidence":0.77, "needs_grounding":false }
  ]
}
```

**`suggestion_update` (Hub → Cook) — PATCH an existing suggestion**

Used when a follow-up transcript revises a previously emitted suggestion (e.g., cook says "actually 8 minutes"). Hub identifies the affected suggestion by ID and patches it.

```json
{
  "type":"suggestion_update",
  "session_id":"abc",
  "t_ms":500,
  "id":"sug_002",
  "patch":{"duration_sec":480,"text":"Sauté for 8 minutes"}
}
```

**`suggestion_remove` (Hub → Cook) — REMOVE a suggestion from the UI**

Used when the cook dismisses a suggestion, or when the hub determines a suggestion is superseded and should be cleared.

```json
{"type":"suggestion_remove","session_id":"abc","t_ms":520,"id":"sug_001"}
```

**Quest client rule:** Never re-render or reorder the pending list on receipt of `instruction_suggestions`. Append only. Apply patches in place. Remove by ID. This is what prevents flicker and reorder storms.

---

## 8) Cook action messages (Cook → Hub)

```json
{ "type":"suggestion_accept",  "session_id":"abc", "t_ms":400, "id":"sug_002" }
{ "type":"suggestion_dismiss", "session_id":"abc", "t_ms":410, "id":"sug_001" }
{ "type":"suggestion_edit",    "session_id":"abc", "t_ms":420, "id":"sug_002",
  "patch":{"duration_sec":480,"text":"Sauté for 8 minutes"} }
{ "type":"mark_done",          "session_id":"abc", "t_ms":800, "id":"task_007" }

```

---

## 9) Clarification flow (uncertainty handling)

### 9.1 Hub → Cook

Use when the transcript implies ambiguity (“that pan”, “add it now”, etc.) and the system cannot safely resolve it.

```json
{
  "type":"clarification_request",
  "session_id":"abc",
  "t_ms":500,
  "question":"Which pan are you using?",
  "options":["Large pan","Small pan","Not sure"],
  "related_suggestion_id":"sug_010"
}

```

### 9.2 Cook → Hub

```json
{
  "type":"clarification_response",
  "session_id":"abc",
  "t_ms":520,
  "related_suggestion_id":"sug_010",
  "choice":"Large pan",
  "free_text":null
}

```

**Policy:** prefer asking the cook or keeping a plain task card over guessing.

## 9.5 Overlay events — Phase 2 only

> **Not implemented in Phase 1.** All Phase 1 compiler output goes to suggestion cards (§7.0). Overlay events are defined here for forward compatibility and to inform Phase 2 Unity work.

Overlays are spatial/UI-anchored visual cues (highlights, arrows, labels) that can be auto-rendered without cook approval when confidence is high.

### overlay_upsert (Hub → Cook) — Phase 2

```json
{
  "type":"overlay_upsert",
  "session_id":"abc",
  "t_ms":600,
  "overlay":{
    "id":"ov_001",
    "overlay_kind":"highlight",
    "text":"Onions",
    "confidence":0.9,
    "ttl_ms":4000,
    "dismissible":true,
    "editable":true,
    "anchor_hint":{ "target_label":"onion" }
  }
}
```

### overlay_remove (Cook → Hub) — Phase 2

```json
{"type":"overlay_remove","session_id":"abc","t_ms":650,"id":"ov_001"}
```

### overlay_edit (Cook → Hub) — Phase 2

```json
{"type":"overlay_edit","session_id":"abc","t_ms":660,"id":"ov_001","patch":{"ttl_ms":8000,"text":"Red onions"}}
```

**Phase 2 anchor progression:**
- Phase 2a: UI anchor — screen-space, attached to task board or floating label; cook can drag to reposition.
- Phase 2b: World anchor — spatially locked to detected object via vision grounding (RAM/SAM3). Falls back to UI anchor if grounding fails.

```json
// Phase 2a: UI anchor
"anchor_hint": { "type": "ui", "target_label": "onion", "position": "task_board" }

// Phase 2b: World anchor (requires vision grounding)
"anchor_hint": { "type": "world", "target_label": "onion", "object_id": "obj_123", "confidence": 0.92 }
```

## 10) Compiler behavior (LLM + rules)

### 10.1 Inputs

- newest `transcript_final` (the trigger)
- **transcript window:** all `transcript_final` entries from the last 5 minutes (both `mentor` and `cook` speakers), in chronological order, labeled by role. Use a time window — not a count — to bound prompt size and LLM cost as sessions grow.
- current accepted tasks/timers (so the compiler knows what the cook is already tracking)

### 10.2 LLM + hybrid extraction

**Default LLM:** Claude Haiku (`claude-haiku-4-5-20251001`) via Anthropic API. Fast (~1–2s), cheap, good instruction following and structured output.

**Offline / no-internet fallback:** Ollama with `llama3.1:8b`. Run locally. Latency is higher (~3–5s on CPU, ~1s on GPU) but fully offline — useful for lab testing without internet.

**Structured output is mandatory.** The compiler must return a JSON list matching the suggestion schema (§7.2). Use:
- Anthropic API: `tools` parameter with the suggestion schema as a tool definition. The LLM calls the tool with validated JSON — no string parsing needed.
- Ollama: `format: "json"` with the schema in the system prompt.

Never parse free-form LLM text for suggestions. If the model returns invalid JSON or fails schema validation, fall back to transcript-only (§10.3).

**Hybrid extraction:** Use the LLM for action semantics, sequencing, checks, and warnings. Run deterministic parsing **before** the LLM call for durations, temperatures, and quantities (regex is fast and reliable for "10 minutes", "180°C", "two cups"). Inject the parsed values into the LLM prompt context so the model doesn't have to re-parse them.

**System prompt structure:**
1. Role: "You are a cooking instruction compiler. Extract actionable items from the conversation."
2. Suggestion schema (JSON Schema of §7.2)
3. Current accepted tasks (so you don't re-suggest what's already active)
4. Constraint: "Only extract items explicitly mentioned in the conversation. Do not invent steps."

### 10.3 Validation and fallback

- validate compiler output against the suggestion schema
- if invalid: emit **no suggestions** (transcript-only) and log an error
- never invent steps not present in the conversation

### 10.4 Tier selection rule

**Phase 1 (this spec) — cards only:**

- Always emit `instruction_suggestions` (card on task board).
- High-confidence items (`confidence >= 0.85`) may be visually distinguished in the UI but still require cook action.
- If safety-critical (`kind: warning`) OR `confidence < 0.5`: emit `clarification_request` instead.
- Never emit `overlay_upsert` in Phase 1.

**Phase 2 (with vision grounding):**

- If `confidence >= 0.85` AND not safety-critical AND vision grounding succeeds: emit `overlay_upsert` with world anchor.
- If grounding fails: fall back to suggestion card.
- If `kind: warning` OR `confidence < 0.5`: emit `clarification_request`.

---

## 11) MR UI behaviors (Phase 1 minimum)

### 11.1 Task board

Three columns:
- **Pending:** new suggestions land here on `instruction_suggestions`. Append to bottom — never reorder the list.
- **Active:** accepted suggestions move here. Timers start (or confirm-duration UI shows) on acceptance.
- **Done (optional):** accepted items move here on `mark_done`.

On `suggestion_update`: patch the item in-place in whichever column it currently lives (do not move it).
On `suggestion_remove`: remove the item from whichever column it is in. If the cook was mid-interaction with it, close that UI first.

### 11.2 Timers

- timer suggestions are shown as cards
- acceptance starts a timer OR opens a short “confirm duration” UI (choose one and keep consistent)

### 11.3 Interruption control

- Suggestions are emitted on `transcript_final` boundaries only (never on partials).
- Flicker and reorder prevention is guaranteed by the delta protocol (§7.3): the Quest appends, patches in place, and removes by ID — it never replaces or re-sorts its full list.

---

## 12) Errors and resilience

### 12.1 Error event (Hub → clients)

```json
{ "type":"error", "session_id":"abc", "t_ms":0, "code":"ASR_TIMEOUT", "message":"ASR chunk timed out" }

```

### 12.2 Expected behavior

- If ASR fails: call continues; transcript may pause; suggestions may pause.
- If compiler fails: transcript continues; no suggestions; session stable.
- If WS reconnects: client re-subscribes; hub continues.

---

## 13) Phase 2 hooks (do not block Phase 1)

- **Vision grounding:** Add RAM/SAM3 for object detection and world-anchored overlays. Hub calls vision service after cook accepts a suggestion with `needs_grounding=true`. If grounding fails, suggestion remains a normal card. Never block UX on grounding.
- **Auto overlays:** Introduce `overlay_upsert` events (§9.5) once UI pattern is validated and world anchoring is available.
- **Quest-forwarded ASR path (optional):** If you need Quest-side audio processing (spatial normalization, noise reduction) before ASR, implement the Quest → hub PCM forwarding path described in §2.1.1. Use a `ConcurrentQueue` for audio thread safety.

Items confirmed in Phase 1 (not Phase 2):
- Cook audio ASR (§2.1): Quest optionally streams cook mic PCM to `/ws/audio/{session}/cook`. Same code path as mentor audio; compiler gains full dialogue context.
- Mentor feedback echo (§5.3.1): Hub echoes compiled suggestions to mentor's events WS.