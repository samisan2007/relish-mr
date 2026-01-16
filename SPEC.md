```markdown
# SPEC — Phase 1 (Audio-First + MR Suggestions)
**Last updated:** 2026-01-15
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
- Forward received mentor audio (decoded WebRTC track) to hub for ASR (Phase 1 default)
### 2.1.1 Audio forwarding behavior (Phase 1)

Quest forwards mentor audio when:
- WebRTC peer connection state is `connected`
- Remote audio track is received and playing
- Hub WebSocket (`/ws/audio/...`) is connected

Quest stops forwarding when:
- WebRTC peer disconnects
- Remote audio track ends
- Hub WebSocket closes (Quest should attempt reconnect)

If hub WebSocket is disconnected but WebRTC call continues:
- Call audio continues normally (cook and mentor can still talk)
- ASR/suggestions pause (graceful degradation)
- Quest attempts WebSocket reconnect every 5 seconds
- Hub resumes ASR when Quest reconnects

### 2.2 Mentor client (Web link)
Responsibilities:
- Join by URL/session code
- WebRTC peer: send mentor mic, receive cook audio
- No separate PCM stream by default (Phase 1): mentor audio reaches the hub via Quest forwarding of the received WebRTC track.
- Fallback mode (only if needed): mentor can stream PCM to the hub for ASR.
- Minimal UI: Join, call status (optional transcript display)

### 2.3 Hub (FastAPI, Python)
Single-process hub does:
- sessions (create/join/leave)
- WebRTC signaling relay (offer/answer/ICE)
- PCM ingest → ASR → transcript events
- transcript → compiler → suggestion events
- logging to disk (JSONL)
- PCM ingest source (Phase 1 default): forwarded mentor audio from the Quest (not captured directly in the mentor browser)

---

## 3) Real-time channels

### 3.1 WebRTC (Cook ↔ Mentor)
Purpose: real conversational call (two-way audio).
Signaling: via hub WebSocket (below).
Media: audio only.

### 3.2 WebSockets (clients ↔ hub)
We use three WS routes:
- signaling (WebRTC negotiation)
- events (JSON messages)
- audio ingest (binary PCM to ASR)
Phase 1 default: `/ws/audio/...` is fed by the Quest forwarding the received mentor audio from WebRTC.

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
- Hub behavior: buffer → ASR → emit transcript events

Phase 1 default:
- `speaker=mentor` audio is provided by the **Quest** (forwarding the received mentor WebRTC track).
Fallback mode:
- mentor browser may stream PCM directly if Quest forwarding is unavailable.
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
- ASR ingest (default): Quest forwards decoded mentor audio (received via WebRTC) as PCM to `/ws/audio/{session}/mentor` → hub buffers 2–5s chunks → ASR emits `transcript_final`.
- ASR ingest (fallback): mentor browser streams PCM to `/ws/audio/{session}/mentor` if Quest forwarding is not feasible.

- Compilation: on each `transcript_final`, hub runs compiler (LLM + rules) → emits `instruction_suggestions` to cook.
- MR loop: cook accepts/dismisses/edits suggestions → hub logs actions; accepted timers start or prompt for duration (per UX choice).
- Fallback: if ASR or compiler fails, the call continues; hub emits transcript-only and logs errors (no session crash).

### 4.5.1 Expected latency profile (Phase 1, Quest forwarding)

Typical timeline from mentor speaks → cook sees suggestion:

T=0ms:    Mentor speaks into mic
T=20ms:   Browser captures and WebRTC encodes (Opus)
T=50ms:   Network transmission (LAN: ~20ms, Internet: ~50-200ms)
T=70ms:   Quest receives and decodes
T=75ms:   Cook HEARS audio (Quest audio output)
T=76ms:   Quest forwards PCM to hub via WebSocket (1ms processing)
T=85ms:   Hub receives forwarded PCM (10ms network)
T=2085ms: Hub has buffered 2 seconds of audio
T=2585ms: ASR completes (Whisper ~500ms)
T=2600ms: Compiler runs (~15ms)
T=2610ms: Hub emits instruction_suggestions
T=2620ms: Cook sees MR suggestion card

Total: ~2.6 seconds from speech to suggestion
Breakdown: 2.0s buffering + 0.5s ASR + 0.1s network/processing

Cook heard the audio at T=75ms, saw suggestion at T=2620ms
→ Suggestion appears ~2.5 seconds after cook heard it

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

### 5.3.1 Mentor feedback (Phase 1 optional, Phase 2 recommended)

Hub can optionally emit to mentor's events WebSocket:

{
"type":"instruction_suggestions_echo",
"session_id":"abc",
"t_ms":300,
"items":[...],  // Same as sent to cook
"note":"These suggestions were sent to the cook"
}

Mentor UI (minimal web interface) can display:

- Last transcript: "You said: 'Dice the onions'"
- Compiled: "→ Action card: 'Dice the onions'"
- Status: "Accepted by cook" | "Dismissed" | "Pending"

Implementation priority: Phase 2 (not blocking for Phase 1 testing)

Rationale: Allows mentor to verify system understood them correctly and
adjust language if needed. Supports RQ3 (does mentor adapt behavior?).

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

### 6.1.1 Quest audio forwarding (implementation note)

Unity's AudioFilterRead callback provides buffers at ~20ms intervals:

- At 48kHz stereo: ~1920 samples/channel/buffer = 7680 bytes (stereo int16)
- At 48kHz mono: ~1920 samples/buffer = 3840 bytes (mono int16)

Quest forwards each AudioFilterRead buffer as a single WebSocket message:

- No batching (to minimize latency)
- Typical send rate: 50 messages/second
- Hub buffers received frames until 2-5 second window for ASR

If WebSocket send buffer fills (network congestion):

- Drop oldest buffered frames (real-time priority)
- Log dropped frame count
- ASR will have gaps but session continues

### 6.2 ASR chunking

- buffer 2–5 seconds then transcribe → emit `transcript_final`
- partials are optional; only use them if stable

**Rule:** compile suggestions from `transcript_final` only (to avoid UI thrash).

---

## 7) Suggestion model (instruction compiler output)

### 7.0 Interaction policy (auto overlays vs suggestions vs confirm-first)
Not all outputs require explicit cook approval. (MAYBE)

- **Auto overlays (no approval):** High-confidence, low-risk cues may render immediately (e.g., subtle highlights/arrows). The cook can remove/edit/disable them.
- **Suggestions (cook chooses):** Medium-confidence or workflow-impacting items appear as cards (Accept/Edit/Dismiss).
- **Confirm-first:** Safety-critical or high-impact items require explicit confirmation before any persistent MR action.

### 7.1 Philosophy

The system outputs **suggestions**, not commands.

Low-risk overlays may render immediately when confidence is high, but the cook can always remove/edit/disable them; (MAYBE- higher-impact items remain explicit suggestions or require confirmation.)

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

### 7.3 Suggestion event (Hub → Cook)

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

## 9.5 Overlay events (auto-rendered MR cues)

Overlays are visual cues (highlight/arrow/label). They may be auto-rendered (Tier 1) or rendered after acceptance/confirmation.

### overlay_upsert (Hub → Cook)

Creates or updates an overlay.

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

---

### overlay_remove (Cook → Hub)

```json
{"type":"overlay_remove","session_id":"abc","t_ms":650,"id":"ov_001"}

```

### overlay_edit (Cook → Hub)

```json
{"type":"overlay_edit","session_id":"abc","t_ms":660,"id":"ov_001","patch":{"ttl_ms":8000,"text":"Red onions"}}

```

> Note: In Phase 1, anchor_hint can be a placeholder and overlays can be UI-anchored. In Phase 2, anchor_hint is resolved via grounding (RAM/SAM3 + scene graph) to a stable world anchor.
> 

```json
// Phase 1: UI anchor (screen-space)
"anchor_hint": {
  "type": "ui",
  "target_label": "onion",
  "position": "task_board"  // Or "top_left", "floating"
}

// Phase 2: World anchor (spatial)
"anchor_hint": {
  "type": "world",
  "target_label": "onion",
  "object_id": "obj_123",  // From vision grounding
  "confidence": 0.92
}
```

## 10) Compiler behavior (LLM + rules)

### 10.1 Inputs

- newest `transcript_final`
- short transcript window (last 5–15 finals)
- (optional) current accepted tasks/timers

### 10.2 Hybrid extraction (recommended)

- deterministic parsing for: durations, temps, quantities
- LLM for: action semantics, sequencing, checks, warnings

### 10.3 Validation and fallback

- validate compiler output against the suggestion schema
- if invalid: emit **no suggestions** (transcript-only) and log an error
- never invent steps not present in the conversation

### 10.4 Tier selection rule

Phase 1 (no vision grounding):

- If confidence >= 0.85 AND not safety-critical: emit overlay_upsert with UI anchor
    - UI anchor means: screen-space attached to task board or ingredient label (not world-anchored)
    - Cook can manually drag to reposition
- Else: emit instruction_suggestions (card on task board)
- If safety-critical OR confidence < 0.5: emit clarification_request

Phase 2 (with vision grounding):

- If confidence >= 0.85 AND vision grounding succeeds: emit overlay_upsert with world anchor
    - World anchor means: spatially locked to detected object (pot, cutting board, etc.)
- Else: fall back to UI anchor or suggestion card

---

## 11) MR UI behaviors (Phase 1 minimum)

### 11.1 Task board

- Suggested list/column: new suggestions land here
- Active list/column: accepted suggestions move here
- Done list (optional): accepted → marked done

### 11.2 Timers

- timer suggestions are shown as cards
- acceptance starts a timer OR opens a short “confirm duration” UI (choose one and keep consistent)

### 11.3 Interruption control

- suggestions are emitted on `transcript_final` boundaries
- avoid flicker and reorder storms

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

- Add cook-audio ASR for full dialogue context.
- Add vision grounding for overlays/suggestions when confidence is high; cook can always remove/disable overlays. Do not block UX on grounding failures.
- If grounding unavailable: suggestion remains a normal MR card.