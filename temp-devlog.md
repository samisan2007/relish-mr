# Real-time food tracking — findings, progress, options

## Backend comparison on the webcam, one or two pens — 2026-09-10

Six separate webcam runs with prompt "pen", holding one and/or two pens,
from the webcam tab's run log. These are not replays of identical frames:

| # | backend | fps | ms | hits | **ids** |
|---|---|---|---|---|---|
| 1 | SAM3 fp16 | 3.8 | 264 | 64/91 (70%) | **1** |
| 2 | SAM3 image + ByteTrack | 4.5 | 224 | 111/132 (84%) | 18 |
| 3 | SAM3 image + ByteTrack | 4.5 | 224 | 111/137 (81%) | 12 |
| 4 | Hybrid (SAM3 seed → YOLOE) | 21.6 | 46 | 324/347 (93%) | 14 |
| 5 | YOLOE (text prompt) | 31.7 | 32 | 185/270 (69%) | 0 |
| 6 | SAM3 image + ByteTrack | 4.5 | 224 | 100/147 (68%) | 14 |

**What the columns measure.** `hits` counts frames with at least one returned instance;
it does not establish that both pens were found or that detections were correct. `ids`
counts distinct non-None IDs over the whole run, not ID switches. One ID could mean a
stable track, a missed second pen, or an ID moving between pens. Twelve to eighteen IDs
for one or two pens suggests fragmentation and/or false detections, but does not separate
those causes. FPS is derived from backend timing, excluding capture and UI rendering;
warm-up and, for Hybrid, SAM3 seeding are included.

**Keep SAM3 video fp16 as the default for further testing.** Its 3.8 fps run and low ID
count are encouraging, but do not close the tracking question in [SPECS.md](SPECS.md).
Replay a clip containing two pens, crossing, rotation and brief occlusion, and check
each pen's identity and missed frames before selecting a final backend.

**SAM3 image + ByteTrack has not earned a switch.** The three runs gain only 0.7 fps
over video SAM3 while reporting 12-18 IDs. That agrees with earlier observations of
fragmentation under motion. The 68-84% hit range also leaves detection coverage open.

**Hybrid remains a useful speed candidate.** 21.6 fps and 93% hit frames are promising,
with 14 IDs still requiring investigation. Its log combines SAM3 seed IDs with YOLOE
IDs without an identity mapping at handoff, so that total is not a pure YOLOE track count.

**YOLOE's zero IDs needs a threshold/confirmation check.** Raw detections can be returned
without confirmed tracks; this metric alone does not prove a broken ID conversion.
Installed Ultralytics 8.4.138 defaults to `tracktrack.yaml` because our YOLOE wrappers
do not select a tracker explicitly. Text detections are accepted from 0.1 confidence,
but TrackTrack starts IDs above 0.7 and requires three matched observations for tracks
appearing after the first frame. This can explain hits without IDs; saved confidence
traces are needed to confirm it for this run. A useful next comparison is explicitly
selected ByteTrack versus TrackTrack on the same clip. No tracker thresholds changed.
Text YOLOE also keeps its tracker across Start/Stop, so a fresh process is needed for
independent runs until that reset is implemented.

**Default verified — 2026-09-10.** `video_ui.py` selects `fp16 autocast, 1008px`,
preloads that same config, and reuses it on Start. The shared model also serves the
video-file tab and Hybrid seeding. fp32 stays available for webcam comparisons.
Seven existing regressions pass, plus a model-stubbed UI check of the dropdown,
preload reuse, fp32 selection, tuned options and restoration of the default.

## Why SAM3 looked like ~1 fps — two causes, both fixable — 2026-09-09

Every "SAM3 is ~1 fps" figure in this log came from two compounding factors, neither of
which is a hard limit of the model.

**1. The webcam tab previously defaulted to fp32 (fixed 2026-09-10).** The config dropdown started on `CONFIGS[0]`,
`"baseline (fp32, 1008px, device_map)"`. Runs that left it unchanged used full precision.
This log concluded back on 2026-09-07 that fp16 is ~2.3x faster, but the UI default
had not adopted it. Manually selected fp16 runs already used the faster path.

**2. Cost is linear in the number of tracked objects,** and the headline benchmarks were
run on a plate of 13-14 meatballs — the worst case, not the operating point.

Fitted on live webcam frames (RTX 5070, steady state past the memory-bank ramp):

```
fp32:  513 ms fixed + 140 ms per tracked object
fp16:  207 ms fixed +  70 ms per tracked object
```

| precision | objects | ms/frame | fps |
|---|---|---|---|
| fp32 | 1 | 653 | 1.5 |
| fp32 | 3 | 934 | 1.1 |
| fp16 | 0 | 207 | 4.8 |
| fp16 | 1 | 275 | 3.6 |
| fp16 | 3 | 406 | 2.5 |

The fp32 model predicts 2333 ms at 13 objects; this log's earlier measurement on the
full plate was 2387 ms. Same law, both scenes.

**Confirmed by hand on the webcam**, tracking a pen:

| run | fps | ms | hits | ids |
|---|---|---|---|---|
| fp32 baseline | 1.6 | 642 | 88/124 | 1 |
| fp16 autocast | 3.4 | 294 | 102/242 | 2 |

**So SAM3 video mode at one or two objects is 3.0-3.6 fps with memory propagation
intact** — the thing that tracks through rotation and pose change, which nothing else
tried here can do. For comparison the SAM3-image + ByteTrack backend gets 4.3 fps but
loses ids on any real motion. At low object counts that hack buys nothing.

**Where the time goes.** Forward-hook profile of one steady-state fp16 frame at 13
objects:

| submodule | params | ms/frame | share |
|---|---|---|---|
| `detector_model` | 840.4 M | 50.5 | 4.7% |
| `tracker_model` | 11.7 M | 815.4 | **75.0%** |
| `tracker_neck` | 7.8 M | 4.8 | 0.4% |
| pre/post-processing | — | ~216 | 19.9% |

The 11.7 M-parameter tracker costs 16x the 840 M-parameter detector, because memory
attention runs once per tracked object while the backbone runs once per frame. At ~70 ms
per object, the tracker is worth profiling further. Cross-hardware comparisons with
DARTF do not establish whether this implementation is efficient.

Also checked: `session.get_obj_num()` equals the returned instance count exactly, so
there is no hidden pool of internal tracks inflating the cost. What you see is what you
pay for.

**DARTF review correction — 2026-09-10.** This profile argues against optimizing only
the detector at 13 objects. It does not rule out DARTF, which also exports the memory
tracker to TensorRT (see the updated review below). The recorded 4.7% share applies to
this scene and profile, not automatically to the one-pen operating point.

**Two changes worth making:**

1. **Default the webcam config to fp16 — done 2026-09-10.** The dropdown and
   preloaded model now use the same fp16 config.
2. **Cap the tracked object count.** Nothing needs 13 tracked meatballs, and each one
   costs ~70 ms. Secondary to the fp16 default at low object counts, but it is what
   makes a full plate viable at all.

**Unresolved:** the two hand runs above differ in hit rate — fp32 hit on 71% of frames,
fp16 on 42%. That may be scene variation across two uncontrolled runs of different
lengths, or fp16 may genuinely cost detection sensitivity. Worth a controlled check
before treating fp16 as strictly better; a faster backend that sees the object less
often is not obviously a win.

## SAM3 image mode + ByteTrack — 2026-09-09

Branch `sam3-image-bytetrack`. Prompted by [DART](https://github.com/mkturkcan/DART),
which reports SAM3's *image* detector at ~11 fps with ByteTrack for ids. The cheap
version of that idea, using our own `Sam3Runner` and ultralytics' already-installed
`BYTETracker`, is a fourth entry in the webcam Model dropdown. No DART code, no new
dependency.

**Outcome: 4.5x faster, but the tracking is not usable.** Live test below. Keeping the
branch for the speed measurement and as a base for the GMC/retuning options at the end.

**Speed — 4.5x faster than SAM3 video.** Measured on the RTX 5070, `meatballs_img.jpg`
pasted into a 640x480 frame, steady state after the first frame:

| Backend | ms/frame | fps |
|---|---|---|
| SAM3 video, fp16, 13 objects | 1038 | 0.96 |
| SAM3 image + ByteTrack, fp32 | 514 | 1.9 |
| **SAM3 image + ByteTrack, fp16** | **234** | **4.3** |

fp16 autocast is worth 2.2x here, same as in video mode. No memory-bank ramp, so the
first frame is the only slow one (~1.3s, CUDA warm-up) instead of a 7-8 frame climb.

**Tracking holds, but only for slow motion.** Ids come from IoU association with no
temporal memory, so they survive exactly as long as consecutive detections overlap.
Swept per-frame displacement on ~30px-wide meatballs:

| px/frame | ids confirmed | ids surviving 6 frames |
|---|---|---|
| 0 | 14 | 14 |
| 8 | 12 | 12 |
| 16 | 11 | 11 |
| 32 | 1 | 1 |
| 48 | 4 | 0 |

Ids collapse past ~16px/frame, roughly half an object width. At 4.3 fps that is the
open question: whether real cook/headset motion stays under ~half an object width per
234ms. Untested on live footage — the sweep is synthetic translation.

**Not measured yet:** rotation, occlusion recovery, and whether the 3 detections that
never got confirmed ids are a threshold-tuning issue. The one live webcam run of this
backend was taken while the GPU was thrashing (see the VRAM entry below) and is void.

**Verdict on DART:** not needed to get this far. Its extra speed comes from TensorRT and
pruned/distilled backbones, which is where to look only if 4.3 fps proves insufficient
*and* the motion limit above turns out to be the binding constraint rather than the
frame rate itself.

Run the check from `perception/` with
`python tests/smoke_sam3_image_track.py ../../Media/meatballs_img.jpg meatball`.

### VRAM leaks made the webcam tab freeze after a few runs — fixed

Symptom: after starting and stopping a few times, any backend crawls (one run logged
7411 ms/frame). Cause was 11.9 GiB of 12 GiB held while idle. On Windows the driver
spills the overflow to system RAM instead of failing, so it reads as a freeze rather
than an out-of-memory error.

Two leaks, both fixed:

- **The tracking session was never released.** `webcam_loop`'s `finally` only did
  `capture.release()`. The SAM3 session holds its memory bank and vision-feature cache
  on the GPU, and Stop cancels the generator rather than letting it return, so every
  run stranded ~1350 MiB. Now calls `session.reset_inference_session()`. Verified flat
  at 7794 MiB across four consecutive runs.
- **The image model stayed resident after switching backends,** a second full copy of
  SAM3 alongside the video model. Now released on switch, and loaded at fp16 (1618 MiB
  instead of ~3300). Verified: three load/release cycles each return to exactly
  4253 MiB.

**`gc.collect()` is required before `torch.cuda.empty_cache()`.** The model's modules
and accelerate hooks form reference cycles, so dropping the name leaves the weights
alive and `empty_cache()` finds nothing to return — the release is a silent no-op
without it. Note `get_bench_tracker` has the same omission and likely also fails to
reclaim when switching configs; not touched.

### DART / DARTF — corrected upstream review, 2026-09-10

**DART accelerates detection and offers ByteTrack IDs.** Its main project documents
Windows 11 / RTX 4080 testing. Faster detections may reduce the gap that hurts our
box association, but do not establish correct identities for moving pens.
Source: [DART README](https://github.com/mkturkcan/DART#tracking).

**DARTF is the more relevant temporal-tracking experiment.** It exports separate FP16
TensorRT tracker neck/init/step engines in addition to accelerating the detector.
Its native mode propagates SAM3 memory each frame; upstream reports about 6 ms/object
on RTX 4090 and 33 ms/object on Orin. These are not predicted RTX 5070 timings.
The lightweight default uses mask IoU, motion and query embeddings; its hybrid
propagates missed tracks. Neither is identical to our box-only ByteTrack or YOLOE Hybrid.
Source: [DARTF README](https://github.com/mkturkcan/DART/blob/main/dartf/README.md).

**Native mode is not proven equivalent to our Hugging Face pipeline.** The implementation
has its own association/refresh rules and defaults to `--sam3-prune 4`, restricting
memory keys. `--sam3-prune 0` selects full memory for a closer comparison.
Source: [video driver](https://github.com/mkturkcan/DART/blob/main/dartf/demo/run_video.py).

**Windows / RTX 5070 compatibility is unverified, not ruled out.** The supplied plugin
build uses Linux `.so` output, Linux TensorRT paths and CUTLASS 3.5.1. It documents
architectures 80/86/87/89/90; this is a comment, not a hard allowlist, and does not
validate SM120. Building and running the relevant engines needs a separate compatibility
check. Source: [plugin build script](https://github.com/mkturkcan/DART/blob/main/dartf/plugins/build.sh).

Keep the current fp16 default while evaluating that path separately. If the engines
run, compare native DARTF with full memory against our SAM3 on the same two-pen clip,
measuring identity continuity, misses, latency and memory use. The old backbone-only
dismissal was wrong: our profile makes accelerated memory tracking worth investigating.

### Live webcam result — the motion limit is real and binding

Tested by hand on the webcam, two pencils. Speed held at ~4 fps as measured.
Tracking did not survive contact with reality:

- Ids churn constantly. Two objects produced several ids.
- Detection count is unstable frame to frame — two pencils sometimes read as three
  objects. That is SAM3 per-frame detection noise, not an association failure: with no
  memory, every frame is an independent opinion and every false positive spawns an id.
- Any normal-speed movement loses the track outright. Only "really really slowly"
  keeps ids. That matches the 16px/frame synthetic threshold exactly.

Pencils are close to the worst case for IoU association — thin diagonal objects whose
boxes are mostly background, and two of them overlap heavily — so this is a harsher
test than meatballs on a counter. It is not harsher than a hand rolling a meatball.

**Conclusion: re-detection plus IoU association does not work at 4 fps for anything
hand-held.** The speed win is real; the tracker is not usable as-is. Same failure mode
as the YOLOE hybrid, reached from the opposite direction — no temporal memory, so
association degrades to whatever the geometry gives you between two distant frames.

### What that leaves

Cheap and untried, in the same file, no new dependency:

- **BOTSORT with `gmc_method: sparseOptFlow`** instead of ByteTrack. Global motion
  compensation estimates the frame-to-frame camera shift by optical flow and corrects
  the Kalman prediction before matching. Aimed squarely at the case where the *camera*
  moves and the scene is static — the headset case. Ships in ultralytics already.
- **ByteTrack's defaults assume ~30 fps.** `track_buffer: 30` is 7.5 seconds at 4 fps,
  and the Kalman process noise expects displacements 7-8x smaller than ours. Nothing
  was retuned for a 4 fps stream.
- **BOTSORT ReID** (`with_reid`, a standalone encoder via `build_encoder`) matches on
  appearance so it can re-associate across zero overlap. Worth knowing it cannot help
  with *identical* objects: two pencils, or 14 meatballs, look the same to it. It fixes
  "is this the same object" only when objects are distinguishable.

None of these fix a 234ms gap between frames. They make association smarter within it.

### The question this test actually raises

Every tracking approach tried so far solves *image-space* persistence: keep an id on a
region of pixels as it slides around the frame. Worth asking whether the product needs
that at all. Food on a counter does not move; the head does, and the Quest already
knows its own 6DoF pose to millimetres. If the widget anchors in world space (depth hit
or MRUK anchor, both listed as undecided in [SPECS.md](SPECS.md)), head motion stops
being a perception problem, and perception only has to answer "what is it and how big"
every second or so — which SAM3 video already does at 1 fps.

That reframing does not cover food being actively manipulated (see
`Media/Roll-Meatballs-in-Cupped-Hands.jpg` — a meatball moving in the cook's hands is
genuinely moving in world space). So the anchoring decision in SPECS.md and the
tracking decision are the same decision, and settling anchoring first may make most of
the tracking problem disappear.

## Review correction — 2026-09-07

The observations below predate two implementation fixes: YOLOE received RGB
NumPy frames where it expects BGR, and Hybrid reapplied seed boxes to each
new frame without keeping the seed image. Hybrid now preserves that image,
extracts visual embeddings from it once, and reuses them for later frames.
The earlier explanation of motion failures as an architectural ceiling is
therefore provisional; live rotation and occlusion need retesting with the
corrected pipeline. The webcam's tuned config now also forwards both tuning
options and reapplies the global cuDNN flag when switching cached configs.

Validation after the fixes: seven automated regressions pass; the tuned
SAM3 config detects 13 meatballs with both tuning options applied. A real
SAM3-to-YOLOE run on a translated local image seeds 13 instances and detects
10 after translation, keeps YOLOE IDs across the three follow-up frames,
extracts visual embeddings only once, and installs a fresh reference on
restart. This synthetic translation check does not validate live pose
changes, occlusion, or sustained performance. Run it from `perception/` with
`python tests/smoke_hybrid.py <image> <prompt>` using the project venv.

The remaining sections preserve the earlier findings and proposed directions.

*(Temp file at repo root — a devlog.md exists on another machine and will be
merged in later. Kept separate and tracked so it survives a pull/merge,
rather than living in the gitignored Documentation/ folder.)*

## What's built

- SAM3 image tester, SAM3 video tracker (file + live webcam tabs).
- `Sam3ImageTracker` in `sam3_runner.py` — SAM3 image detection per frame + ByteTrack ids, 4.3 fps (see the 2026-09-09 entry).
- `bench_realtime.py` — CLI benchmark, replays one captured clip through several configs for a fair comparison.
- Webcam tab: **Model** dropdown (SAM3 / YOLOE text-prompt / Hybrid), SAM3 **config** dropdown, and a run log of the last 8 runs.
- `yoloe_runner.py` — YOLOE text-prompt tracker, and a Hybrid tracker (SAM3 seeds once, YOLOE tracks after).

## Findings

**SAM3** — the only backend that's actually *reliable* (tracks through rotation/angle changes, understands niche food nouns like "meatball"). But:
- Per-frame cost **ramps for ~7-8 frames** as its memory bank fills, then plateaus — short benchmarks were measuring the ramp, not real speed. Fixed the benchmark's warmup to account for this.
- **Confirmed steady-state**: fp32 0.42 fps (2387ms/frame) vs fp16 (via `torch.autocast`) 0.96 fps (1038ms/frame) — fp16 is the right choice, ~2.3x faster.
- **No further speed lever survived testing**: resolution override breaks (hardcoded token-grid buffers), `num_maskmem` reduction breaks (tied to a fixed-shape learned weight), `torch.compile` is dead on Windows (no Triton), device_map/mask_size/cudnn.benchmark/conditioning-frame-cap all made no meaningful difference. **~1fps is the practical ceiling** with cheap levers.

**YOLOE (text prompt)** — fast (~100-500ms/frame) but its MobileCLIP vocabulary is too weak for specific food nouns ("meatball" maxes ~0.17 confidence even on the largest checkpoint). Not usable alone for this project's actual prompts.

**YOLOE (visual exemplar)** — excellent in isolation (0.9+ confidence given a good example box) but it's re-detection-by-similarity to a static snapshot, not real tracking with memory.

**Hybrid (SAM3 seeds once -> YOLOE tracks)** — works as code, but **live testing (moving pen, moving eye) showed it degrades to plain YOLOE's exact weakness** once seeded: loses the object on angle/pose change, because it inherited YOLOE's re-detection mechanism, not SAM3's temporal memory. Currently just reproduces YOLOE's ceiling with extra steps.

**Bugs fixed along the way**: SAM3 dtype crash (full weight-cast broke on real detections, fixed via autocast), ByteTrack `None`-ID crash in the run log, a `.gitignore` gap that almost committed a 599MB file, camera buffer lag (`CAP_PROP_BUFFERSIZE=1` — cheap fix applied, not yet confirmed it holds on Windows).

## The core tension

The only reliable backend (SAM3) is slow (~1fps). Every fast alternative tried breaks under motion because it lacks real temporal memory — it re-detects per frame by static similarity rather than tracking. That's a property of the model, not a config bug, so no amount of hybrid wiring around YOLOE fixes it by itself.

## Options

| # | Option | What changes | Tradeoff |
|---|--------|--------------|----------|
| 1 | **Periodic-refresh hybrid v2** | SAM3 re-grounds on a cadence (e.g. every 300-500ms) as authoritative correction; a classical CV tracker (CSRT/KCF) — motion/appearance continuity, not semantic matching — fills the gap between corrections | Structural fix, doesn't depend on YOLOE's weak spot at all. More to build; gives a box not a mask between corrections (approximate diameter); can drift/fail under occlusion |
| 2 | **Adaptive multi-exemplar Hybrid** | Keep the current Hybrid, but periodically add newly-confirmed detections as more visual exemplars so YOLOE's matching set covers more angles over time | Smallest code change. Still the same re-detection mechanism — may still fail on a genuinely novel pose; risk of polluting the exemplar set with a bad detection |
| 3 | **Accept ~1fps, design around it** | Use SAM3 alone at its real rate | No further engineering. Worth checking first whether ~1fps is an actual proven blocker for real cook/HMD movement speed, or an untested assumption |
| 4 | **Push SAM3's raw speed further** | Attempt ONNX/TensorRT export, or check for a smaller/distilled SAM3 checkpoint | Bigger, uncertain-payoff investment — every cheap lever is already exhausted |

**Option 1 vs 2, precisely**: Option 1 changes *what does the tracking* — swaps YOLOE's semantic re-detection for a fundamentally different algorithm (motion/appearance continuity) that doesn't care what the object is, so it doesn't break on unseen angles. Option 2 keeps the exact same mechanism and just gives it more reference photos — a mitigation, not a structural fix.

No direction has been chosen yet.
