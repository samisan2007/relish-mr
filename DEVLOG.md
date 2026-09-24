# Relish — devlog

Running log, newest first. Update after each completed batch or task, including
intermediate batches within a session: what changed, what it means, what's next.
For *what we're building*, see [SPECS.md](SPECS.md); for what comes next,
[PLAN.md](PLAN.md). Older entries are history: later entries supersede them.

---

## 2026-09-24 — Documentation and code cleanup

Removed stale, duplicated and wrong statements before the next experiments.
- **`temp-devlog.md` is merged into this log and deleted.** It was the older log
  from another machine, kept to be merged later. Its 2026-09-14 and 2026-09-22
  sections duplicated entries here. Its unique 2026-09-07 to 2026-09-10 history
  is condensed below, with conclusions that were later disproved marked.
- **Deleted `perception/sweep.py` and `perception/track_video.py`.** Both were
  unreferenced first-batch scripts. `sweep.py` needed a local `Test Data/`
  folder that no longer exists; the replay CLI
  (`keyframe_hybrid.py --backend sam3video`) and the UI's file tab cover
  `track_video.py`.
- **Corrected wrong statements:**
  - perception/README said YOLOE text takes 100-500 ms per frame (it is about
    30 ms), grounding takes ~1 s (a few hundred ms), SAM3 plateaus at ~1 fps
    regardless of config (3-3.6 fps at one or two objects), and the backbone
    dominates SAM3's frame time (the per-object tracker does).
  - An entry below said the YOLOE backends re-detect without tracking and
    restart their IDs. They keep IDs between ordinary frames and reset them
    only when fresh exemplars are installed. Two superseded estimates below
    now carry notes.
  - Code comments: the keyframe interval counts processed frames (every 10 at
    10 fps is ~1 Hz), SAM3 video costs ~70 ms per tracked object, and the Unity
    widget manager no longer assumes ~1 fps perception.
- **Trimmed:**
  - SPECS' perception section now links to the READMEs instead of repeating
    them.
  - Old SAM 3.1 fps figures left the acceptance target.
  - The "biggest lever" claim for world placement went, along with a reviewed
    link marked "not relevant" and a duplicate DeepSeek note.
  - PLAN lost its finished §0 and stale commit notes. It gained two known
    facts: SAM3's two-pen masks cause merges, and retired tracks still cost
    EdgeTAM passes.
  - The agent instructions no longer warn about an architecture that no
    longer appears anywhere in the repo.
- Removed Python `__pycache__` folders. Replay outputs in `perception/runs/` and
  the local model setups stay.

**Validation:** all 60 unit tests pass, and every relative link in the
Markdown docs resolves. No model or GPU run was needed. **Git:** committed and pushed on `dartf`. **Next:** PLAN §1,
the lost-object diagnosis.

## 2026-09-24 — Candidate checkpoint and comparison-plan review

The offline candidate harness, its tests and its docs are committed and pushed
as `ab7db58` on `dartf`. Weights, source checkouts, cases and runs stay in the
ignored `perception/candidates-local/`.
- On the same inputs, three objects took 32.8-33.2 ms per request with native
  EdgeTAM and 55.2-61.3 ms with Transformers EdgeTAM (entry below). The 77 ms
  keyframe-hybrid figure from clip replays includes association and our mask
  post-processing, so it is not the matching baseline.
- The native patch applies to upstream EdgeTAM revision
  `7711e012a30a2402c4eaab637bdb00a521302c91`: one `view` becomes `reshape` in
  `sam2/modeling/perceiver.py`. The harness verifies the source and records the
  patch in run manifests.
- PLAN §3 gained explicit checks:
  - native EdgeTAM on Windows without the optional CUDA extension;
  - the deployed request and transport cost;
  - pen-clip propagation and reseeding, scored against human-reviewed identities;
  - controlled frame skipping.
- EV-M's next checks: more food photos at fixed settings, `ball` against
  `meatball`, and non-food round objects as negatives.
- Standing rule, now in AGENTS.md: update the Markdown docs after every
  completed batch, not only at the end of a session.

Previous-frame association and hiding a track after one missed keyframe are
plausible causes of lost pens, not yet confirmed for each failure.
**Next:** PLAN §1.

## 2026-09-24 — Offline candidate preparation and GPU smokes

Prepared [candidate commands](perception/candidates/README.md) for official
EdgeTAM, SAM 2.1 tiny, Transformers EdgeTAM and EfficientSAM3 EV-M before the
handled-food recordings and Quest are available. Sources/checkpoints are pinned,
downloaded, and isolated in `relish-candidates:local`; inference succeeds with
network disabled. No UI backend or production tracking policy changed.

The harness freezes decoded frames and SAM3 seed proposals once, verifies their
hashes, runs fresh tracker sessions, and preserves masks, IDs, preview videos,
per-frame CSVs and environment manifests. Manifests include checkpoint/code hashes,
installed packages, GPU, CUDA, image ID, settings and failure/completion state.
Large source/checkpoint/result files stay in ignored `perception/candidates-local/`.
The previous reviewed baseline remains `5f76de4`, already pushed on `dartf`.

**Compatibility findings:**
- Official EdgeTAM already batches ordinary propagation. Its pinned implementation
  crashed with multiple objects at an expanded tensor's `.view(...)`. Setup applies
  exactly one `.reshape(...)` correction; runs verify it and record it in the manifest.
  Redundant timm pretrained initialization is skipped before strict checkpoint load.
- The HF configuration's default construction requested online timm metadata despite
  local model loading. It now uses the backbone configuration already packaged in
  the pinned checkpoint. Each session receives its own ID list: resetting the HF
  session otherwise clears the shared seeds and breaks repeat 2.
- EV-M strictly loads EfficientViT B1 + MobileCLIP S0, context 16. Its published image
  checkpoint is usable. Upstream's Stage-2 memory-weight release remains unchecked;
  a ready EV-M video tracker has not been established.

**Bounded smoke measurements, RTX 5070, FP16 autocast, no compilation:**
24 frames derived from `Media/meatballs_img.jpg`, translated 0-20 pixels, then held.
Same frame files and highest-scoring inspected seeds; case
`20260924-112344-240729`. Two independent sessions per tracker/object count.
Ranges below span their post-warmup means (first eight frames excluded).

| Implementation | Objects | Mean request ms | Request p95 ms | Peak allocated GiB |
|---|---:|---:|---:|---:|
| Native EdgeTAM + reshape fix | 1 | 28.3-30.8 | 30.2-39.4 | 0.416 |
| Transformers EdgeTAM | 1 | 28.6-32.5 | 30.3-38.4 | 0.357-0.360 |
| Native EdgeTAM + reshape fix | 3 | 32.8-33.2 | 34.8-35.1 | 0.442-0.443 |
| Transformers EdgeTAM | 3 | 55.2-61.3 | 62.7-70.0 | 0.374-0.377 |
| SAM 2.1 tiny | 3 | 74.6-74.9 | 75.8-76.0 | 0.623 |
| Native EdgeTAM + reshape fix | 10 | 69.2-74.1 | 71.8-80.0 | 0.815-0.816 |
| Transformers EdgeTAM | 10 | 145.8-151.5 | 151.4-185.3 | 0.437 |

These measure propagation plus CPU mask transfer, excluding preprocessing, seeding,
rendering and writes. They are not hybrid FPS or capture-to-display measurements.
Native postprocessing is disabled and full clip state retained; HF keeps current
presence gating, seed-score correction and pruning. The runs therefore do not isolate
batching alone. Native EdgeTAM's multi-object advantage warrants further comparison;
its cost still increases with object count and uses more allocated GPU memory.
The three-object final overlays were inspected and follow the translated objects.
This tests execution, not occlusion, identity recovery, new arrivals or handled food.

Tracker runs under `perception/candidates-local/runs/` (all start `20260924-`):
`113250-985309-edgetam` (3), `113328-973534-sam21tiny` (3),
`113952-657367-edgetam-hf` (3), `114027-527508-edgetam` (10),
`114104-699277-edgetam-hf` (10), `114142-660253-edgetam` (1),
`114229-422221-edgetam-hf` (1). Earlier failed runs remain for diagnosis.

**EV-M image smoke:** prompt `meatball` on the same original photo. Threshold 0.4
returned zero detections (cold 3448 ms, next request 61 ms). Threshold 0.1 returned
12 proposals, scores 0.123-0.257 (cold 3607 ms, next two requests 73.7/73.9 ms,
peak allocated 1.09 GiB). Visual inspection shows food coverage and a large false
positive over the hand. Raising the threshold enough to remove that hand mask
also removes a partially occluded meatball; do not treat threshold tuning as a
complete quality fix. These tiny samples are not recall or accuracy estimates.
Runs: `20260924-113035-002003-evm` and `20260924-113719-264327-evm`.

**Checks:** 60 unit tests pass, including shared-input integrity, strict checkpoint
loading and ID preservation across cleanup. PowerShell parsing and `git diff --check`
pass. GPU runs validated two fresh sessions for all tracker implementations, with
1/3/10-object scaling for both EdgeTAM paths. Preview encoding resizes 452x678 to
456x680; raw masks keep source dimensions. No long-duration or Quest test was run.

**Next:** annotate the existing lost-pen windows, compare native/HF masks through
real movement and reseeding, and diagnose association separately. Keep SAM 2.1 tiny
as the quality alternative and evaluate EV-M across more food photos before using
it for keyframes. The recording checklist and commands are ready for new footage.

## 2026-09-24 — Ordered plan, lost-state cooldown and preserved replay results

Added [PLAN.md](PLAN.md) with the next tasks and acceptance checks: establish a
baseline, diagnose lost objects, validate live timing/recovery, profile before
batching, then test handled food and connect the Quest. Alternative models get
bounded offline comparisons before UI integration. No final backend is selected.

**Documentation corrections:**
- Added `AGENTS.md` and replaced obsolete remote-mentor/audio Copilot instructions.
- Reconciled SPECS with moving food: capture-time pose does not remove food-motion
  latency; hand overlap must not reject held food; the 8-10 fps target stays.
- YOLOE tracks with persistent IDs on ordinary frames. Installing fresh exemplars
  resets the tracker; failed grounding does not. Earlier descriptions of it as
  not tracking, or resetting IDs every frame, were too broad.
- EdgeTAM caches shared image features; its per-object tracking work is sequential.
  Batching might improve utilization but does not remove that work or guarantee
  flat cost. Lost-pen causes are still hypotheses requiring labeled replay.
- Corrected FAST's webcam/3080 status, historical compilation context and devlog
  links. Historical measurements below are retained, not new benchmark results.

**Small implementation fixes:**
- Both YOLOE hybrids ground immediately at startup and once on the request after
  a new loss. Continuing absence waits 500 ms after the last attempt finishes.
  YOLOE keeps searching with the old exemplar between retries. Before the first
  seed, skipped requests show a waiting message and do not inflate inference FPS.
  The existing periodic refresh slider still applies while tracking. The delay
  is provisional; `retry_seconds=0` enables the old policy for comparison.
- Keyframe replay saves every run in a fresh UTC timestamp directory with a
  manifest (settings, clip/source hashes, host package/GPU details and Git state
  when available). It saves mean/p50/p95 request times, total processing time and
  completion/failure state; partial CSV data survives an exception or interrupt.
  Camera, writer and session cleanup runs on those paths too. Invalid inputs and
  zero-frame videos fail explicitly. This does not archive checkpoints or dirty code.
- Pinned Transformers 5.16.1, Ultralytics 8.4.138 and timm 1.0.30 to the locally
  installed versions behind our tracking APIs. No dependency install was needed.

**Validation:** 57 unit tests pass, including timed initial retries, repeated
loss, recovery with old exemplars, cooldown after slow grounding, UI waiting-state
statistics, unique run output and cleanup/partial CSV on failure and interruption.
The run also emitted asyncio unclosed-event-loop ResourceWarnings during UI tests;
these did not fail the suite and remain uninvestigated. No new GPU inference,
food-quality or performance claim is made. Existing uncommitted SAM 3.1 work was
preserved; batching, asynchronous grounding and association changes are not included.

**Next:** replay and annotate the lost-pen windows, then change one cause at a time.

## 2026-09-24 — Handled food is the requirement, compiled SAM 3.1 fixed, SAM 3.1 keyframes

**SPECS now states that the food moves.** Cooks knead meatballs, shape them and
hold them up to compare. The food changes shape and is hidden by the hands, so
perception has to track and measure it through that.
- World anchoring, low-rate detection and following the hand joints are
  complements, not replacements.
- Items 1, 4, 5 and 11 and the acceptance target were corrected to match. They
  had treated still food as the main case.

**Compiled SAM 3.1 froze on the webcam. It is fixed by compiling only the detector.**
- **Cause:** upstream also compiles tracker and matching functions for fixed
  sizes, which change with the number of objects and memory frames.
  - A headless empty-scene run looked fine: frame 1 compiled in 62 s, then
    360 ms per frame. With no objects, the tracker never runs.
  - With three pens coming into view, frames 1-4 took 21, 16, 86 and 32 s, and
    every later change in the count recompiled again.
- **Fix:** `compile_detector` in [worker.py](perception/sam31/worker.py) runs
  upstream's compile and then puts the five tracker and matching functions back
  to uncompiled.
- **Result** on 200 clip frames with 0-2 pens: frame 1 compiles for 18 s, then
  412 ms mean and 440 ms max, with no stalls.
  - The full compile ran at 424 ms and uncompiled at 500 ms on the same frames,
    so nothing was lost.
  - These timings are with the compile cache already built; a cold cache takes
    longer.

**New: keyframe hybrid with SAM 3.1 keyframes, uncompiled or compiled.**
- Two webcam menu entries and the replay backends `hybrid31` and `hybrid31c`.
- The Docker worker starts with the stream, and Stop can cancel it.
- The worker's image-only mode now treats every request as an independent
  picture. Before, a second request would have gone through the video tracker.
- Replay of clip 2, compiled: 9.6 fps and 19 IDs. Keyframes take 368 ms,
  against about 230 ms for SAM3 image. Frames between keyframes take 77 ms.
  - Output appeared on 814 of 1234 frames. The SAM3 keyframe run on the same
    code was interrupted, so the two can't be compared on coverage yet.
  - The thresholds (0.4 and 0.2) were tuned on SAM3's scores and have not been
    retuned for SAM 3.1.

**Webcam round-up (user, approximate):**

| Backend | fps | Notes |
|---|---|---|
| SAM3 video | ~3 | 1-2 pens |
| SAM3 image + ByteTrack | 4.5 | 2-3 pens |
| Keyframe hybrid, SAM3 | ~12 | Loses the pen when it moves |
| Keyframe hybrid, SAM 3.1 | 8-9 with 3 pens, 33 empty | |
| Keyframe hybrid, SAM 3.1 compiled | Same as uncompiled | |
| YOLOE text | 33 | |
| Hybrid, SAM 3.1 seed → YOLOE | 33 tracking, 11 lost | |
| DARTF | 5 | |

**What this means:**
- **Only the keyframe hybrids and the YOLOE backends reach the 8-10 fps target.**
  The YOLOE backends detect by exemplar each frame and keep IDs between frames,
  but reset them whenever fresh exemplars are installed.
- **Compiling makes no difference to the SAM 3.1 keyframe hybrid.** A keyframe
  is 1 frame in 10, so saving ~90 ms on it is ~9 ms per frame. EdgeTAM's
  per-object cost between keyframes dominates.
- **SAM 3.1 keyframes are slower than SAM3's.** Each goes through Docker and
  sets up fresh state. So the SAM 3.1 hybrid runs at 8-9 fps against 12.
- **Open: the keyframe hybrid loses a moving pen.** Two causes are possible:
  - EdgeTAM drops a fast-moving pen, and only the next keyframe, up to 10
    frames later, can recover it;
  - or SAM3 misses the motion-blurred pen at a keyframe, and the new
    `retire_after=1` hides it until a keyframe sees it again.
  - A replay can separate the two.

**Validation:** all 52 unit tests pass. Compiled SAM 3.1 was probed headless on
the clips and then on the webcam, where compiled and uncompiled ran at the same
speed.

**Next:**
- Diagnose why a moving pen is lost, and fix it. A likely fix is running an
  early keyframe as soon as a shown track disappears.
- Cut EdgeTAM's per-object cost, or run keyframes off the main loop. Both are
  needed for plates with many pieces.

## 2026-09-24 — Keyframe hybrid live: double IDs, merged pens and ghost tracks

The first webcam run of the keyframe hybrid showed three tracking failures. All
three came from the tracker's bookkeeping, not from the models, and all are
fixed in [keyframe_hybrid.py](perception/keyframe_hybrid.py).

**Live fps depends on what is in view, in opposite ways for the two hybrids:**
- **YOLOE hybrids (SAM3 or SAM 3.1 seed):** about 33 fps while tracking, 8-10
  fps with nothing in view. Once YOLOE loses everything, the seeder re-grounds
  on every frame until it finds something again. This is deliberate, for the
  fastest re-detection; running the seeder less often while lost is an option.
- **Keyframe hybrid:** about 43 fps with nothing in view, about 12 fps with two
  pens. With no tracks, EdgeTAM is skipped. With tracks, it runs one pass per
  object per frame, because transformers loops over objects one at a time.
  - That costs about 25-35 ms per track on the clips.
  - A 13-meatball plate would then take about 350 ms per frame. That figure is
    extrapolated, not measured.
  - *Superseded:* the offline candidate runs later measured Transformers EdgeTAM
    propagation alone at about 150 ms with 10 objects. The clip figure also
    counted retired tracks and our own per-track work.
- So live runs only compare on the same content. Use the recorded clips.

**1. One pen, two IDs.** Nothing compared tracks against each other.
- How it happened:
  - A keyframe started a new track while the pen's old track was briefly lost.
  - EdgeTAM then found the old track again, so both sat on the same pen.
  - Each keyframe then gave SAM3's single detection to one of the two. When
    the winner alternated, neither was ever retired.
- Now, every frame, tracks whose masks nearly coincide (IoU ≥ 0.6) are
  reduced to one: a live track over a retired one, then the older ID.
  - The dropped track can't be matched again, and the next compaction removes it.
- A same-pen duplicate had been shown on 140 of 658 frames of clip 1 and 726 of
  1234 of clip 2. Now it is 1 and 5.
- The earlier ID counts (9 and 19) were low partly because the duplicate tracks
  absorbed re-detections that would otherwise have started new IDs.

**2. Two touching pens as one track.** When one hand holds both pens, EdgeTAM
spreads one track over both.
- **The first version of fix 1 made this worse.** It detected duplicates by
  containment, so it removed the second pen's own track, which lies inside the
  spread one. It now uses IoU.
- **Matching:** a pair still needs containment ≥ 0.3, for the rotating-pen case
  above. Pairs are now taken in IoU order, so each pen's own track claims its
  detection first. The spread track is re-seeded with the remaining pen.
- **Memory reset:** if SAM3's mask disagrees with a re-seeded track's EdgeTAM
  mask (IoU < 0.5), that track's recent memory is cleared. Otherwise it spreads
  again from its two-pen history within a few frames.

**3. Ghost tracks.** After a pen moves, EdgeTAM can hold a track on pen-like
background, such as a shelf edge, at 0.99 confidence. Such a track stayed
visible until SAM3 had missed it on 3 keyframes (30 frames). Now:
- SAM3 reports detections down to 0.2 (`keep_threshold`).
- Only detections of 0.4 or more start or re-seed a track.
- Detections from 0.2 to 0.4 only confirm an existing track.
- A track with no support at all is hidden at that keyframe (`retire_after`
  3 → 1).
- A hidden track can still be matched, so a pen that SAM3 misses once comes
  back under its own ID.

Clip 2 (3 pens, 1234 frames), before and after fixes 2 and 3. Merged and ghost
tracks are counted on the frame before each keyframe, against SAM3's
detections on it:

| | Before | After |
|---|---|---|
| Tracks covering two separate pens | 3 | 2 |
| Ghost tracks (no SAM3 detection ≥ 0.2) | 21 | 16 |
| Distinct IDs | 23 | 20 |
| fps | 9.5 | 10.8 |

- On clip 1, ghosts went from 10 to 8 and IDs from 7 to 9, at 12 fps both times.
- Duplicate frames stayed at 0 on both clips.
- The clips rarely show two pens held together, so they cannot confirm the
  merge fix. The live test is that check.
- Some counted ghosts may be real pens that SAM3 missed entirely. They were not
  checked by eye.

**Found, not fixed: SAM3 sometimes returns one mask covering two touching pens.**
This happened on 5 keyframes of clip 2. Once, the two-pen mask scored above both
single pens and started a track. Preferring the separate masks would also split
a pen that SAM3 returns as a whole plus two fragments, so measure how often
that happens first.

**Validation:** all 52 unit tests pass, 3 of them new: duplicate ranking,
matching a spread track, and the IoU dedupe. The CLI replay runs with the new
`--keep-threshold` flag.

**Not yet verified:** a live webcam run with these fixes.

**Next:**
- Test live with both pens in one hand.
- Decide how to handle SAM3's two-pen masks.
- Run SAM3 keyframes on a worker so they stop stalling the stream.

## 2026-09-24 — Keyframe hybrid: SAM3 image + EdgeTAM, first like-for-like clip comparison

Two recorded webcam clips now sit in `Media/`: `pen_test_vid.mp4` (2 pens, 658
frames) and `pen_vid_test_x3.mp4` (up to 3 pens, 1234 frames). Both have
crossing, rotation, hand occlusion and motion blur. They are the first fixed
benchmark: every backend below saw the same frames, on the RTX 5070.

**New backend: keyframe hybrid** ([keyframe_hybrid.py](perception/keyframe_hybrid.py)),
SPECS item 11.
- SAM3 image mode (fp16) runs every 10th frame. EdgeTAM, a distilled SAM 2
  tracker with memory, carries the masks through the frames in between.
- Keyframe detections are matched to live tracks by mask overlap, so a
  re-detection **keeps the track's ID** instead of restarting it, which is the
  YOLOE hybrid's flaw.
- Also in the webcam menu, with a "run SAM3 every N frames" slider.
- The script replays a clip through any backend, writing an annotated video, a
  per-frame ID CSV and an ID-timeline PNG to `perception/runs/` (gitignored).

| Backend | Clip 1 fps | Clip 1 IDs (2 pens) | Clip 2 fps | Clip 2 IDs (3 pens) |
|---|---|---|---|---|
| SAM3 video fp16 | 3.0 | 2 | 2.9 | 3 (OOM at frame 942) |
| SAM3 image + ByteTrack | 4.4 | 47 | 4.3 | 132 |
| **Keyframe hybrid, every 10** | **11.6** | **9** | **9.0** | **19** |

Hybrid timing breakdown:
- Keyframes take about 230 ms (SAM3 plus seeding).
- Frames between keyframes take 60 ms on clip 1 and 95 ms on clip 2. EdgeTAM's
  cost grows with the number of tracks.
- SAM3 runs inline, so each keyframe stalls the stream. Running it on a worker
  would bring the average toward the between-keyframe rate.

**What the ID counts mean, checked by eye on the overlays:**
- **SAM3 video** holds identities best, but its low count hides a failure mode:
  at clip 1 frame 500 it merges two crossing pens into one mask.
- **SAM3 image + ByteTrack** is unusable, as on 2026-09-10.
- **The hybrid's remaining breaks** come from two causes:
  - a pen disappearing fully into the hands and reappearing, which SAM3 then
    detects as new;
  - bursts of extra SAM3 detections, which are retired after 3 unconfirmed
    keyframes but still consume IDs.

Two fixes halved the hybrid's IDs (18 → 9 and 36 → 19) at the same speed:
1. **Match on containment (intersection over the smaller mask), not IoU.** A
   pen rotated from horizontal to vertical between keyframes. EdgeTAM kept only
   its top while SAM3 returned all of it, so their IoU was low and a new ID
   started.
2. **Drop SAM3 duplicates before seeding.** SAM3 image mode returns a whole pen
   plus fragments of it, and each would seed its own track.

**Fixes and findings:**
- **transformers 5.16 EdgeTAM bug, worked around in our code.** A mask-seeded
  object's score is 1-D and a propagated object's is 2-D, and `forward()`
  concatenates them. Any keyframe that re-seeds some tracks but not others
  crashed.
- **The EdgeTAM streaming session keeps every frame and output.** It is now
  pruned to the memory the model actually reads.
- **Dead tracks are compacted.** Once 4 or more accumulate, the session
  restarts with only the live tracks, under the same IDs.
- **Existing issue, not fixed: SAM3 video streaming grows GPU memory every
  frame.** It ran out of memory at frame 942 of clip 2. The current webcam
  SAM3 backend should hit the same limit after about 5 minutes.
- **New dependency: `timm`**, for EdgeTAM's backbone. Added to requirements.
- The `kernels` package is still not installed. Without it SAM3 video skips its
  NMS and hole-filling post-processing.

**Validation:** all 49 unit tests pass, 5 of them new, covering matching,
containment and dedupe. A UI-path smoke test with real models tracked 40 clip
frames under a single ID. Switching to another backend released the hybrid's
1,866 MiB of VRAM.

**Not yet verified:** a live webcam run of the hybrid, and any keyframe interval
other than 10.

**Next:**
- Test the hybrid live on the webcam.
- Try keyframe intervals of 5 and 20.
- Consider keeping a track alive through full hand occlusion (hide it rather
  than release its ID) so the pen keeps its ID when it reappears.

## 2026-09-22 — SAM 3.1 + YOLOE hybrid, re-grounding, and a drift guard

Added **Hybrid (SAM 3.1 seed -> YOLOE track)** to the webcam menu. The hybrid's
seeder was already duck-typed on `start_stream`/`track_frame`, which `Sam31Runner`
already satisfies, so the model work was small; the work was lifecycle. The
session is now a `dict` subclass carrying `reset_inference_session()`, so the two
cleanup paths that already existed stop no-opping and actually close the SAM 3.1
container — without it every Stop stranded one. `start_stream` forwards
`on_started`, so Stop can cancel a container mid-boot as it can for the plain
3.1 and DARTF backends.

The seeder now stays open for the whole stream and re-grounds, instead of seeding
once and freezing its exemplars. Two triggers: a selectable frame interval (a new
webcam slider) and, always, YOLOE returning nothing. On the 3080 at 640x480 a
quiet YOLOE frame is ~35ms against ~750ms for a SAM 3.1 re-ground, giving ~28 fps
at interval 0, ~21 at 60 and 9.0 measured at 10. Keeping the worker alive is what
makes this affordable — a re-ground is one inference, not a ~40s container boot.

The interval defaults to off, because re-grounding **restarts the track IDs**:
`YOLOE.predict` drops its predictor after `set_classes`, so installing fresh
exemplars rebuilds the tracker. Measured ids `[1] -> [2] -> [3] -> [4]`, one bump
per re-ground, and 3 distinct IDs against 1 on the same 40-frame clip at intervals
10 and 30. Since the run log reports distinct-ID counts as a quality number, a
short interval quietly inflates it.

Chased a live report of the hybrid locking onto a whole t-shirt while prompted for
a pen or a watch. SAM 3.1 is not the culprit: on that frame it returns nothing for
either noun, and where it is confident (knives) it returns 0.84-0.90, so it emits
clean exemplars or none. YOLOE seeded on knives and shown a person also returns
nothing, so drift is not automatic — it needs a thin exemplar whose embedding
generalizes badly. The real defect was structural: YOLOE matches the exemplar, not
the words, nothing downstream could tell a box was wrong, and the only corrective
trigger was *zero* detections, so a confidently-wrong box at 0.29 (over the 0.15
floor) was an absorbing state. Boxes more than `drift_scale` (default 8x, ~2.8x
linear) the biggest seeded box in area are now rejected; the frame then reads as
lost and re-grounds, so drift corrects itself. Area alone will not catch a drift
onto something similarly sized — that still needs the interval.

Unverified: none of this has been through a live webcam run here. The numbers come
from real SAM 3.1 and YOLOE inference on stills, and the drift guard is a heuristic
keyed to the size mismatch in the reported screenshot rather than tuned against
that scene.

## 2026-09-22 — SAM 3.1 and native DARTF FP16 on the RTX 3080

Installed both backends on the home RTX 3080 and ran their startup checks; an
earlier session on this PC could start neither, because both setups were local
to the 5070. DART is pinned to `16fada39` with SM86 engines built locally into
`DART/dartf/assets`; SAM 3.1 uses source `660a5e9e` and checkpoint `daa63191`.
Engines, weights and compile caches are GPU- and host-specific and stay
uncommitted, so each new PC still needs its own setup run.

Native DARTF FP16 holds four IDs through a translated test frame and restarts
cleanly, at 449 ms per frame including Docker transfer. SAM 3.1 tracks the same
four objects at 2.11 fps eager, with masks confirmed by eye rather than by hit
counts alone.

Fixed SAM 3.1 compiled mode, which failed here with an Inductor
`FileNotFoundError` on a file that was present on disk. The Inductor and Triton
caches sat on a Windows bind mount, which does not make one compile worker's
writes visible to another in time. Both launchers now use the Docker volume
`relish-sam31-compile-cache`. Compiled mode then reaches 3.42 fps against 2.68
eager on the same path, after roughly 2.8 minutes of one-time compilation that
lands on the second frame, because the first frame runs eagerly. The apparent
"loads, shows one frame, freezes" symptom was that compile, not a hang.
*Partly superseded:* with objects in view, compilation also repeated whenever
the object count changed. Fixed on 2026-09-24 by compiling only the detector.

On a warm cache, frame 0 took 4.4 s, frame 1 141 s (nine components compiled at
`max-autotune`) and frame 2 29 s, then 0.27-0.34 s per frame. Steady frames 3-5
took 373 ms eager and 293 ms compiled. Frame 1 returns zero objects in both
modes and IDs return at frame 2, which is the live path's normal behaviour.

Both figures are below the 8-10 fps acceptance target, and both were measured on
a stock photo translated across a synthetic frame. That PC had no `Media/`
folder, so `tests/smoke_sam31_ui.py` could not run as written.

## 2026-09-22 — DARTF FAST webcam transport

A watch test on this PC put FAST at about 3 fps, SAM3 at about 2.9 fps and the
SAM3-seeded YOLOE hybrid at about 24 fps, the last being its YOLOE phase rather
than a like-for-like tracker result.

FAST's slowdown turned out to be transport, not the model. The adapter sent a
raw 1008x1008 RGB frame and full boolean masks through Docker's Windows pipes on
every request: 334 ms per request, of which model and tracking were only 108 ms.
Sending lossless PNG at the camera resolution, resizing inside the worker and
bit-packing the output masks cut requests to 159 ms (6.3 fps) with model time
unchanged at 107 ms. The Gradio webcam generator measured 6.35 fps on the same
input and produced an overlay with ID 1.

This measures transport, not watch accuracy or identity continuity through
motion and occlusion, which are still unevaluated.

## 2026-09-15 — SAM 3.1 in picture, video and webcam testing

Added **SAM 3.1** and **SAM 3.1 (compiled)** to the picture, video-file and live
webcam model menus. The existing launchers remain `run_sam_img.cmd` (port 7860)
and `run_sam_vid.cmd` (port 7861). Models now load on selection; choosing 3.1
releases that UI process's other model caches. The image confidence slider
controls detection admission and filters results.

The new adapter reuses DARTF's tested Docker transport and cleanup. It processes
one incoming frame at a time, preserving Object Multiplex tracking state without
future-frame prefetch. It retains 32 frames of memory plus each bucket's first
conditioning frame, caps tracking at 16 regions, and starts fresh IDs on every
run. The limited history may affect longer occlusions. Stop can cancel a webcam
worker during loading or compilation; failed runs also release their resources.
File and webcam inference share a concurrency limit. File results separate
playback FPS, total processing speed, and frame-request timing after the first
eight frames; the latter still includes any later compilation and Docker transfer.

Added a separate [SAM 3.1 photo/recorded-video launcher](perception/sam31/README.md)
using Meta's pinned Object Multiplex source and checkpoint. It reuses the Linux
GPU environment, runs BF16 with PyTorch attention, and saves annotated H.264
videos, visible IDs, per-frame timings and GPU memory measurements. Eager and
compiled inference run successfully on the 12 GB RTX 5070. Compilation of the
initial crowded-scene shapes took about ten minutes; later shapes can compile
again, including after a tracking-state reset.

On the translated meatball tray, pass 2 measured **3.88 fps eager** and
**5.65 fps compiled**, with about **5.0 GiB peak PyTorch allocations**. All 12
region IDs persisted and restarted consistently across two fresh passes.
Visual inspection found hand/wrist false positives, so 12 is not a count of
correctly identified food objects. The single-food photo with both `meatball`
and `ball` prompts returned no detections and correctly failed the synthetic
smoke check. Their empty-output speeds are not successful tracking results.

These are offline propagation timings: the first eight and final prefetched
frames are excluded, as are decoding, prompting, overlays, encoding and Windows
transfer. The detector still prefetches one frame. This configuration does not
yet meet the 8-10 fps target, and synthetic translation does not test real hand
occlusion or identity swaps. Real recorded food clips are the next quality test.

Validation: all 30 unit tests pass; Python and PowerShell syntax checks pass.
Both UIs serve their model menus over HTTP. Normal and compiled UI smokes pass
picture, 40-frame video, two synthetic webcam runs, mask/video dimensions, ID
restart and camera/worker cleanup. The normal-mode webcam also detects food
entering after an initially empty view. Picture output contains 13 regions at threshold 0.5;
video retains 12 region IDs after confirmation. The physical camera at index 0
opens and reads 640 x 480 frames. Visual inspection still shows hand/wrist false
positives; these checks do not establish tracking quality during real occlusion.

The first compiled UI file test measured 2.25 fps for frame requests after the
first eight frames, including transfer and later compilation; the tracking and
encoding loop took 155.2 seconds for 40 frames including worker startup. Short
compiled webcam runs also include compilation and are not steady-speed
benchmarks. These UI results are distinct from the faster offline measurements
above. Saved outputs and exact commands are in the ignored `sam31-local/` folder
and [SAM 3.1 instructions](perception/sam31/README.md). Real food clips remain the
next speed/quality check before choosing a backend for the 8-10 fps target.

Normal UI inference measured 2.41 fps on the file's requests after the first eight
frames, and 2.58/2.63 fps on the final four requests of the two synthetic webcam
runs. Both use Windows/Docker transfer. The food enters on frame 3 and is
confirmed by frame 5. These short functional checks remain below the target.

## 2026-09-14 — RTX 3080 FAST handoff

Prepared a separate DARTF FAST build and recorded-video test for the home RTX
3080. [RTX3080.md](perception/dartf/RTX3080.md) has the commands: build the Docker
images, log into Hugging Face, check the GPU, build engines and test a clip.
The launcher saves headless timings, IDs and an optional annotated-video pass.
The current webcam DARTF option remains the tested native FP16 backend; FAST is
tested through its own launcher before further webcam integration.

The FAST recipe uses the full W8A8 backbone, a fused single-prompt mask head,
upstream lightweight tracking and pipelining. Source and model revisions are
pinned. Shipped activation scales and all 16 upstream calibration images feed
CPU GPTQ, avoiding a large GPU calibration engine. Assets and credentials stay
under the ignored `perception/dartf-local/` directory. Engine builds reject the
5070 and previously cached plans from another GPU/runtime.

Validation: the SM86 Docker image and custom CUDA plugins compile on this PC;
the missing CUTLASS utility include path was corrected. The pinned weights and
all 19 calibration/check images download successfully. All 22 unit tests pass,
including the existing backend regressions, GPU guards and launcher argument
handling. Backbone, text and fused-head CPU exports pass, along with ONNX
validation, all 32 blocks' quantization-site counts, three FP32 references and
a one-block GPTQ smoke check. The backbone rewrite relative L2 error is
5.789e-06. Full GPTQ and INT8 graph rewriting, target TensorRT builds, numerical
verification and FAST inference await the RTX 3080 at home. No 3080 FPS or
accuracy claim is made.

## 2026-09-14 — fp16 default and optional DARTF experiment

**SAM3 now starts with `fp16 autocast, 1008px`.** The preload and webcam dropdown
agree; fp32 remains selectable. SAM3 image + ByteTrack and this default fix were
committed and pushed as `60fe315` on `sam3-image-bytetrack`.

The six pen runs are in the 2026-09-10 entry below. SAM3 fp16
reported 3.8 fps, image + ByteTrack 4.5 fps, Hybrid 21.6 fps and text YOLOE
31.7 fps. These used different frames. Hit counts and distinct IDs do not prove
that both pens kept their identities, so the final tracking choice remains open.

**Added DARTF to the webcam menu on branch `dartf`, as an optional experiment.**
It uses a separate GPU Docker process, local FP16 TensorRT engines and the native
SAM3 memory tracker. The Windows environment and existing backend defaults stay
intact. This is the full-memory FP16 reference path; W8A8 calibration and custom
INT8 plugins are outside this experiment. Setup and resume commands are in
[perception/dartf/README.md](perception/dartf/README.md).

**Validation:** all 17 unit tests pass, covering the existing YOLOE/Hybrid
regressions and DARTF packet handling, instance conversion and worker cleanup.
A model-stubbed UI check verifies the fp16 default and camera release when
startup fails. Docker GPU access, ONNX exports and all six TensorRT engines
succeeded on the RTX 5070. The real `tests/smoke_dartf.py` check also passes:
13 confirmed IDs persist through the translated meatball image, masks and
overlays have the expected dimensions, and a fresh stream restarts IDs at 1.
The last four frames average 549 ms (1.8 fps), including Windows/Docker transfer.
This 13-object synthetic test does not establish live two-pen performance or
identity correctness through occlusion.

**Next:** compare the backends on the same recorded two-object
clip with crossing, rotation and occlusion. Keep SAM3 fp16 as the default while
these checks remain open. Quest transport and pixel-to-world projection still
need implementation.

## 2026-09-10 — Webcam backend comparison, one or two pens

Six separate webcam runs with the prompt "pen", from the webcam tab's run log.
They did not see the same frames:

| # | Backend | fps | ms | Hits | IDs |
|---|---|---|---|---|---|
| 1 | SAM3 fp16 | 3.8 | 264 | 64/91 (70%) | 1 |
| 2 | SAM3 image + ByteTrack | 4.5 | 224 | 111/132 (84%) | 18 |
| 3 | SAM3 image + ByteTrack | 4.5 | 224 | 111/137 (81%) | 12 |
| 4 | Hybrid (SAM3 seed → YOLOE) | 21.6 | 46 | 324/347 (93%) | 14 |
| 5 | YOLOE (text prompt) | 31.7 | 32 | 185/270 (69%) | 0 |
| 6 | SAM3 image + ByteTrack | 4.5 | 224 | 100/147 (68%) | 14 |

- Hits count frames with at least one instance, and IDs count distinct IDs.
  Neither shows whether both pens were found or kept their identities.
- YOLOE's zero IDs: Ultralytics 8.4.138 defaults to TrackTrack, which starts IDs
  above 0.7 confidence and needs three matched observations. Text detections are
  accepted from 0.1, which can explain hits without IDs.
- The Hybrid's 14 IDs mix SAM3 seed IDs with YOLOE IDs; nothing maps one to the
  other at handoff.
- The webcam default was set to `fp16 autocast, 1008px` and verified: the
  preload, the dropdown and the file tab share one model.

## 2026-09-09 — Why SAM3 looked like ~1 fps

Two causes, neither a limit of the model:
1. The webcam tab defaulted to fp32. fp16 autocast is about 2.3x faster; the
   default was fixed on 2026-09-10.
2. Cost is linear in the number of tracked objects, and the headline runs used
   a plate of 13-14 meatballs.

Fitted on live webcam frames (RTX 5070, after the memory-bank warm-up):

```
fp32:  513 ms fixed + 140 ms per tracked object
fp16:  207 ms fixed +  70 ms per tracked object
```

So SAM3 video tracks one or two objects at 3.0-3.6 fps in fp16, with memory
propagation intact. By hand on a pen: fp32 1.6 fps, fp16 3.4 fps.

Forward-hook profile of one fp16 frame with 13 objects:

| Submodule | Params | ms/frame | Share |
|---|---|---|---|
| `detector_model` | 840.4 M | 50.5 | 4.7% |
| `tracker_model` | 11.7 M | 815.4 | 75.0% |
| `tracker_neck` | 7.8 M | 4.8 | 0.4% |
| Pre/post-processing | — | ~216 | 19.9% |

The small tracker costs 16x the detector, because memory attention runs once
per object while the backbone runs once per frame. `session.get_obj_num()`
matched the returned instance count, so no hidden tracks inflate the cost.

Left open: in two uncontrolled hand runs, fp32 hit on 71% of frames and fp16 on
42%. Whether fp16 costs detection sensitivity was never checked.

## 2026-09-09 — SAM3 image + ByteTrack, VRAM leaks and the DART review

Branch `sam3-image-bytetrack`: SAM3's image detector on every frame, with
Ultralytics' ByteTrack for IDs, prompted by DART's report of ~11 fps. Added to
the webcam menu.
- **Speed:** 234 ms per frame in fp16 (4.3 fps), against 1038 ms for SAM3 video
  with 13 objects, on `meatballs_img.jpg` in a 640x480 frame.
- **Tracking:** IDs come from box overlap, with no memory.
  - A synthetic sweep on ~30 px meatballs kept IDs up to 16 px of movement per
    frame and lost them at 32 px, about half an object width.
  - Live with two pencils, IDs churned constantly, detection counts flickered,
    and any normal-speed movement lost the track.
  - **Re-detection plus overlap association does not work at 4 fps for
    anything hand-held.**
- **VRAM leaks made the webcam tab freeze after a few runs; fixed.**
  - The SAM3 session was never released on Stop (~1350 MiB per run).
  - The image model stayed resident after switching backends.
  - Windows spills overflow to system RAM, so this looked like a freeze.
  - `gc.collect()` is needed before `torch.cuda.empty_cache()`, because the
    model's modules form reference cycles.
- **DART review, corrected on 2026-09-10:** DART speeds up detection and uses
  ByteTrack for IDs. DARTF also exports SAM3's memory tracker to TensorRT, which
  the profile above makes worth testing; that led to the DARTF experiment.
  Upstream's ~6 ms per object on an RTX 4090 is not a prediction for our GPUs.
- Untried at the time: BoT-SORT with optical-flow camera-motion compensation,
  retuning ByteTrack for 4 fps, and ReID, which cannot tell identical objects
  apart.
- The entry also argued that world anchoring might make image tracking
  unnecessary for food on a counter. *Superseded:* the food is handled and
  moves (SPECS), so it has to be tracked.

## 2026-09-09 — widget scaffold, and no on-device inference

**Rejected the Meta Image Segmentation building block.** Its only model provider
is Yolo11n-seg: 80 fixed COCO classes, so no food nouns, and strictly weaker than
what already runs on the PC. Passthrough Camera Access — the part worth having —
was already installed. Nothing to add.

**Built the widget scaffold** so there is something to plug perception into.
`Assets/Relish/`:

- `PortionWidget` — translucent sphere, sized by real-world diameter in metres
  (`SetDiameter`), with show/hide, minimize/restore, a green success pulse, a
  billboarded debug label and gizmos. No perception knowledge in it at all.
- `Hands` — resolves `OVRHand` from the hand-tracking blocks; pose, pinch state,
  pinch point. Uses the public `GetHand()` / `PointerPose` API, not the internal
  `HandType` field, so it survives SDK updates.
- `RelishWidgetBuilder` — `Relish > Create Widget Prefab`. Generates the prefab
  and a transparent URP Lit material instead of hand-authored YAML.

Not verified in-headset yet. Run the menu item, drop the prefab in the scene, and
use the component's right-click context menu in Play mode to exercise it.

**Reviewed the scaffold against what perception actually emits** — a
`list[Instance]` per frame (mask, box, score, obj_id): many objects, ~1 fps,
frequent dropouts. Three correctness fixes went in:

- The transparent material now goes through URP's own
  `BaseShaderGUI.SetupMaterialBlendMode` instead of a hand-rolled blend recipe
  that missed the alpha-blend factors URP 17 reads.
- The pinch point is the skeleton's index fingertip (either skeleton version),
  not the pointer-pose origin, which sits several centimetres back from the fingers.
- The "TMP Essentials missing" warning no longer throws in exactly the case it
  warns about.

Gaps that follow from perception's shape rather than from product decisions:
there is no `obj_id → widget` registry (one prefab, N objects), `Place()` snaps
(at 1 fps that is a teleport per second), and there is no stale/lost state even
though losing the object is the devlog's main finding. Recommended: one small
`PortionWidgets` manager that spawns, smooths, ages out and retires widgets, so
the prefab stays dumb. Interactions, UI and further states wait on deciding
what a "portion" communicates — Meta ISDK already has grab/poke/ray, so none of
that needs building.

**Built `PortionWidgets`**, the manager. `Report(frame)` takes
`Detection { Id, Position, Diameter, Score }`, spawns one prefab per new id,
lerps position between reports (frame-rate independent, `smoothing` per
second), dims via the widget's new `SetStale` after `staleAfter` seconds unseen
and destroys after `retireAfter`. Debug labels read `#id  cm  score`.
`Detection` is the contract with perception; the transport's only job is to
fill it in. Add an empty GameObject with `PortionWidgets` on it; the prefab
wires itself on add. In Play mode, press the inspector button *Test: fake frame in front
of camera* — the test methods are inspector buttons now, via a small editor
that reflects over `[ContextMenu]` — three spheres appear, dim
at 1.5 s, vanish at 4 s.

**Next:** decide the tracking approach (the four options in the entry below); then
the Quest ↔ PC transport and the pixel → world projection that fills `Detection`.

## 2026-09-07 — First tracking findings (largely superseded)

Built: SAM3 image and video testers (file and webcam tabs), the replay benchmark
`bench_realtime.py`, and `yoloe_runner.py` with YOLOE text prompts and a
SAM3-seeded YOLOE hybrid.

Findings at the time:
- SAM3's per-frame cost ramps over ~7-8 frames as its memory bank fills, so
  short benchmarks measured the ramp. fp16 autocast is ~2.3x faster than fp32.
- YOLOE text was weak on "meatball" (about 0.17 confidence). YOLOE with a visual
  exemplar scored 0.9+ but lost the object on pose changes.
- *Wrong or premature:* "~1 fps is SAM3's practical ceiling" came from fp32 on a
  13-object plate (see 2026-09-09). "The hybrid only reproduces YOLOE's
  ceiling" predates the pipeline fix below.

**Pipeline fix.** YOLOE received RGB frames where it expects BGR, and the hybrid
re-applied the seed boxes to each new frame without keeping the seed image. The
hybrid now keeps the seed image, extracts visual embeddings once and reuses them.
A synthetic check (`tests/smoke_hybrid.py`) seeded 13 instances and found 10 after
translation.

The four options considered:
1. a periodic-refresh hybrid with a classical tracker between refreshes;
2. adaptive multi-exemplar YOLOE, built on 2026-09-22 as re-grounding;
3. accepting SAM3's rate;
4. faster SAM3 builds, later tried as DARTF.

The keyframe hybrid (2026-09-24) is the memory-based version of option 1.
