# Perception — setup and experiments

Current direction and acceptance checks: [PLAN.md](../PLAN.md).
Requirements: [SPECS.md](../SPECS.md). Results: [DEVLOG.md](../DEVLOG.md).

Offline preparation for official EdgeTAM, SAM 2.1 tiny and EfficientSAM3 EV-M:
[candidate setup and comparison commands](candidates/README.md).

Tested on Windows with Python 3.12 and an NVIDIA RTX 5070. The webcam tools use
OpenCV's Windows DirectShow backend.

## One-time setup

Run these commands in PowerShell from the repository root (`relish-mr/`).
Install Python 3.12 first if `py -3.12 --version` cannot find it.

1. Create the environment, then enter `perception/`:

   ```powershell
   py -3.12 -m venv perception\.venv
   cd perception
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   ```

2. Install the tested PyTorch/CUDA combination:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   ```

   This is the CUDA 12.8 wheel for the local NVIDIA setup. For other hardware
   or drivers, choose the matching wheel from the
   [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

3. Install the remaining dependencies and check the environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m pip check
   .\.venv\Scripts\python.exe -c "import torch; from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor; print('SAM3 imports OK; CUDA available:', torch.cuda.is_available())"
   ```

   On an NVIDIA machine, the last command should report `CUDA available: True`.
   A CPU-only installation can run inference, but will be much slower.

   Transformers, Ultralytics and timm are pinned to the installed versions used
   by our private tracking APIs and EdgeTAM workaround. Upgrade deliberately with
   regression and real-model checks; the remaining dependencies are not locked.

4. Request access to the gated model at https://huggingface.co/facebook/sam3
   (instant-ish approval after accepting the license).
5. Create a token at https://huggingface.co/settings/tokens (read scope) and log in:

   ```powershell
   .\.venv\Scripts\hf.exe auth login
   ```

## Run the tester

From the `perception/` folder:

1. **Start** — models load on first selection. From the repo root just run
   `.\run_sam_img.cmd` (or double-click it); the equivalent from here is:

   ```powershell
   .\.venv\Scripts\python.exe ui.py
   ```

2. **Test** — open <http://127.0.0.1:7860>. Drop in an image, type a noun
   phrase ("meatball", "pot", "lemon"), choose **SAM3**, **SAM 3.1** or
   **SAM 3.1 (compiled)** and hit Submit. Toggle masks/boxes and adjust the
   threshold. SAM 3.1 requires its [Docker setup](sam31/README.md); compiled
   first use can take several minutes.

3. **Quit** — press `Ctrl+C` in the terminal. Closing the browser tab does
   not stop it.

One-off runs from the command line instead (pays the ~10s model load each time):

```powershell
.\.venv\Scripts\python.exe test_sam3.py photo.jpg "meatball"   # saves photo_sam3.png
.\.venv\Scripts\python.exe sweep.py                            # all Test Data images
```

## Video tracking

Tracks every instance of the concept through the clip, keeping stable IDs:

```powershell
.\.venv\Scripts\python.exe track_video.py "..\Test Data\AriaEverydayActivities_1.mp4" "mug"
.\.venv\Scripts\python.exe track_video.py clip.mp4 "meatball" --max-frames 300 --stride 2
```

`--stride` samples every Nth frame (default 3) and `--max-frames` caps the clip
(default 150) — decoded frames are held in RAM, so a full 1408x1408 video would
otherwise need several GB. Writes `<video>_<prompt>.mp4` next to the source.

Or use the UI (`run_sam_vid.cmd` from the repo root), which has two tabs:

- **Video file** — choose SAM3 offline propagation or SAM 3.1's forward-only
  tracker (normal or compiled). The results separate playback FPS from actual
  processing speed. Start SAM 3.1 with 60 frames and stride 1.
- **Webcam (live)** — streaming inference with selectable backends (see
  below), plus a **SAM3 config** dropdown (precision/dispatch/conditioning-frame
  knobs — same options as `bench_realtime.py`, only used when Model = SAM3).
  Defaults to **SAM3 / fp16 autocast, 1008px**, loading on first selection.
  The video-file tab and Hybrid's SAM3 seed share that fp16 model;
  fp32 remains selectable for webcam comparisons.
  Stop logs frame-request fps/ms/hit-rate into a rolling log of the last 8 runs,
  so you can flip backends/configs and compare by eye against a real moving
  object instead of only trusting fixed-clip numbers. Switching model/config
  takes effect on the next Start and reloads (~5-10s SAM3, longer for YOLOE's
  first-ever download). Streaming disables the heuristics that prune duplicate
  tracks, so expect more false positives than the file tab. Capture requests a
  1-frame OpenCV buffer (`CAP_PROP_BUFFERSIZE`) so the feed doesn't fall
  further and further behind real time as inference lags the camera's native
  rate — not guaranteed to hold on Windows' DSHOW backend; if the lag comes
  back, the next step is a background capture thread instead.

First ever run downloads ~3.5 GB of model weights to the Hugging Face cache.

## Model backends (webcam tab)

**SAM 3.1** and **SAM 3.1 (compiled)** are available in the picture, video-file
and webcam model menus. See [setup, UI tests and measurements](sam31/README.md).
The UI uses a causal Docker worker with bounded object memory; the standalone
offline benchmark has different timing and lookahead. Live frame-request timing
includes Docker transfer. Stop releases the worker even during compilation.

**DARTF FAST (W8A8 TensorRT)** is available in the Webcam tab on an RTX 3080
after the [FAST build setup](dartf/RTX3080.md). It uses the optimized detector,
segmentation head and lightweight tracker. The other DARTF entry is native FP16.
The live adapter sends lossless PNG frames and packed masks through Docker.
It processes one frame per request; the recorded benchmark additionally overlaps
frames. On the 3080, a repeated single-dog input improved from 3.0 to 6.35 frame
requests/sec after fixing the transfer overhead (2026-09-22). This is a wiring
and speed check, not a watch-tracking or motion-quality measurement.

**DARTF (native SAM3, FP16 TensorRT)** is an experimental Docker backend with
locally built engines. Real inference and a synthetic translation/restart smoke
check pass; live tracking quality still needs evaluation. See
[DARTF setup and remaining checks](dartf/README.md). It runs full-memory SAM3
tracking with an FP16 TensorRT detector, without the upstream W8A8 INT8 recipe.

**Keyframe hybrid (SAM3 image -> EdgeTAM)** runs SAM3's image model every N
frames (webcam slider, default 10) and lets EdgeTAM, a small SAM 2-style tracker
with memory, carry the masks in between. Keyframe detections are matched to the
live tracks by mask overlap, so IDs persist across keyframes instead of
restarting.
- Tracks that land on the same object are reduced to one.
- A track that SAM3 gives no support at a keyframe is hidden. Detections from 0.2
  count as support; only detections of 0.4 or more start or re-seed a track.
- Speed falls with each tracked object: shared image features are cached, but
  object-specific memory/decoder work runs sequentially. With no tracks it is skipped.
  The interval counts processed frames: every 10 at 10 fps is about 1 Hz, not 2-3 Hz.
- **SAM 3.1 variants:** the keyframes can come from SAM 3.1's image mode instead,
  uncompiled or compiled. It runs in its Docker worker, one per Start, and each
  keyframe is an independent picture. Replay one with
  `python keyframe_hybrid.py clip.mp4 pen --backend hybrid31c`.

Needs `timm` (in requirements). See the 2026-09-24 DEVLOG entries.

`yoloe_runner.py` adds two alternatives to SAM3, evaluated after finding SAM3
plateaus around 1 fps at steady state regardless of config (see below):

The earlier quality comparisons were recorded before fixing swapped color
channels in both YOLOE backends and incorrect visual references in Hybrid.
Those observations need live retesting before drawing conclusions about the
models' relative robustness.

- **YOLOE (text prompt)** — Ultralytics' YOLOE, fast (~100-500ms/frame,
  independent of internal resolution the way SAM3 is not). But its
  open-vocabulary text encoder (a lightweight MobileCLIP model) is noticeably
  weaker than SAM3's on specific food nouns: "meatball" tops out around 0.17
  confidence even on the largest checkpoint (yoloe-11l-seg), vs. SAM3 finding
  it cleanly. Don't take a single fixed-clip empty-scene benchmark's word for
  a model's quality — this only showed up once tested against a real object.
- **Hybrid (SAM3 seed -> YOLOE track)** and **Hybrid (SAM 3.1 seed -> YOLOE
  track)** — a grounding model finds the prompt (slow, ~1s, but reliable on
  niche nouns), then every instance it found seeds YOLOE's visual-exemplar
  mode. The exact seed image is copied in BGR and supplied as `refer_image` at
  the first YOLOE step; the resulting embeddings are reused on later frames
  with `persist=True`. See the [YOLOE visual-prompt API](https://docs.ultralytics.com/models/yoloe/#visual-prompts)
  for how reference images install persistent embeddings. The two entries
  differ only in the seeder: the SAM3 one shares the video tab's in-process
  fp16 model, the SAM 3.1 one keeps a Docker worker open for the stream so a
  re-ground costs one inference rather than a container boot.

  Both ground immediately at startup and on the request after a new loss. If
  still lost, attempts wait 500 ms after the previous attempt finishes. YOLOE
  keeps searching with existing exemplars between attempts; before the first
  seed there is no YOLOE search. Cooldown-only frames are excluded from the UI's
  inference statistics. `HybridVideoTracker(..., retry_seconds=0)` restores the
  old retry-every-request behavior for comparison; 0.5 is a provisional default.

  While tracking, the webcam re-ground slider optionally refreshes every N frames.
  It is off by default because
  installing fresh exemplars rebuilds YOLOE's tracker and **restarts the track
  IDs**. On an RTX 3080 at 640x480 a quiet YOLOE frame is ~35ms against ~750ms
  for a SAM 3.1 re-ground, so the interval sets the average rate: ~28 fps at 0
  (lost only), ~21 fps at 60, ~9 fps at 10 (the last measured directly).
  These are historical tracking-phase numbers, not measurements of the new
  lost-state policy. Ordinary frames use persistent IDs; failed grounding does
  not install exemplars or reset the tracker.

  YOLOE matches the exemplar embedding, not the words, so a thin exemplar (a
  pen, a watch strap) generalizes badly — observed locking onto a whole torso
  at 0.29, over the 0.15 confidence floor. Nothing downstream can tell that box
  is wrong, so a bad frame used to be absorbing: the only corrective trigger
  was zero detections. Boxes more than `drift_scale` (default 8x) the biggest
  seeded box in area are now rejected, which empties the frame and re-grounds.
  A drift onto something *similar in size* to the real object is not caught by
  area alone; that needs the interval. Live robustness through pose change and
  occlusion is still an open question.

## Regression checks

From `perception/`, run the color and hybrid-handoff tests without model
downloads, GPU inference, or camera access:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a real-model check, use an image containing a concept SAM3 can detect:

```powershell
.\.venv\Scripts\python.exe tests\smoke_hybrid.py "path\to\photo.jpg" "meatball"
```

Add `--sam31` to ground with the SAM 3.1 Docker worker instead of in-process
SAM3, and `--reground 2` to force scheduled re-grounds inside the run and print
the track IDs either side of them.

This requires cached SAM3 weights and a local `yoloe-11l-seg.pt` (or `--model`
pointing to a local checkpoint). It translates the image away from its seed
coordinates, checks that visual embeddings are extracted once per stream,
renders overlays in memory, and checks a restart with a new reference. It
does not measure sustained speed or validate live rotation and occlusion.

## Recorded backend comparison

From `perception/`, use the same input for each backend:

```powershell
.\.venv\Scripts\python.exe keyframe_hybrid.py ..\..\Media\pen_test_vid.mp4 pen --backend hybrid
```

Backends: `hybrid`, `hybrid31`, `hybrid31c`, `sam3video`, `sam3image`.
Each run creates a fresh UTC timestamp directory under `runs/` (or `--out`).
It saves an annotated MP4, per-frame CSV, an ID timeline when IDs exist, and
`environment.json`: settings, clip and source hashes, host versions/GPU, Git state
when available, completion/failure state, frame count and request p50/p95/mean.
Partial CSV data is flushed; errors and interrupts release the worker and video
resources. A forcibly killed process can leave the manifest marked `running`.

Request timings exclude decode, drawing, encoding and startup. Total processing
includes those costs and cleanup; it excludes manifest setup and timeline drawing.
Warm request statistics exclude ten frames when the clip is long enough; short
runs include startup requests and are not steady-state benchmarks. Playback FPS
comes from the source clip. None of these is capture-to-display latency.

Commit/review dirty changes before a baseline: hashes identify code but do not
archive it. Checkpoint revisions and worker package versions still need separate
recording for model comparisons. Inspect masks and physical identities; output
coverage and ID counts alone do not establish quality. See [PLAN.md](../PLAN.md).

## Historical SAM3 configuration benchmark

The findings below describe the original Windows SAM3 path. Linux SAM 3.1 now
supports detector-only compilation; the historical Windows limitation is not a
claim that compilation is unavailable for all backends.

`bench_realtime.py` captures one webcam clip, then replays those exact frames
through several `Sam3VideoTracker` configs (precision, dispatch mode, internal
processing resolution, `torch.compile`) so speed comparisons aren't muddied by
webcam variance between runs. The processor always resizes every frame to a
fixed 1008x1008 before it reaches the model, so capture resolution isn't the
cost knob — precision and that internal resize target are.

```powershell
.\.venv\Scripts\python.exe bench_realtime.py --prompt "phone" --capture-seconds 15
```

Move the object during the 3s countdown so the clip has real motion to judge
mask quality against. Or from the repo root: `run_sam_bench.cmd "phone" 15`.

Findings on an RTX 5070, verified against a real detection (not just an
empty-scene timing run — see caveat below):

- **fp16/bf16 via `torch.autocast`** — ~2.2x speedup, same instance count as
  fp32. Precision must go through autocast, not a full `dtype=` weight
  conversion at load time: the video session creates some tensors (memory-bank
  slots, object queries) as plain fp32 outside the parameter tree, so casting
  stored weights leaves them mismatched the instant a track is actually
  created (`Input type (float) and bias type (struct c10::Half) should be the
  same`) — it only shows up once something matches the prompt, which is why
  an empty-scene smoke test won't catch it. `video_runner.py` now handles
  this correctly.
- **`device_map="auto"` vs. plain `.to("cuda")`** — no measurable difference.
- **Processor resize override (504px/288px)** — confirmed broken, not just
  unoptimized: the video session's memory-bank/position-embedding buffers are
  hardcoded to the default 72x72 (1008px/14) token grid, so a real detection
  throws a tensor-size mismatch. Removed from `CONFIGS`; capture/display
  resolution was never the actual cost knob anyway (see above).
- **`torch.compile`** — confirmed dead (missing Triton, no official Windows
  support), fails on the very first forward pass every time. Removed from
  `CONFIGS` rather than left as a config that always shows FAILED.

**Per-frame cost is not flat — this changes the earlier numbers above.**
Timing ramps up over roughly the first 7-8 frames as the memory bank
(`tracker_config.num_maskmem`, default 7) fills, then plateaus around **2x**
the first frame's cost. Measured on fp16: ~530ms first frame -> **~1030ms
steady state** (not the ~270-500ms a short burst suggests). All the earlier
config numbers above were measured on short bursts that only partially
crossed this ramp, so they understate real sustained cost — `--warmup`
defaults to 10 now (was 3) and `--capture-seconds` to 15 (was 8) so a normal
run actually reaches steady state before it starts timing.

`num_maskmem` looked like the obvious lever to shrink that ramp, but it's
tied to a learned positional-embedding weight shaped exactly `[7, 1, 1, 64]`
— reducing it throws a checkpoint size-mismatch at load time, not usable.
Two knobs that don't break correctness — `torch.backends.cudnn.benchmark`
and capping `tracker_config.max_cond_frame_num` to 1 — only bought ~5%
steady-state improvement combined (`fp16 ... tuned` in `CONFIGS`); most of
the per-frame cost is the fixed-size vision backbone, which neither touches.
There's no remaining config-level lever that meaningfully beats fp16/bf16
autocast — further speedup likely means a different architecture (see the
YOLOE section) or restructuring how often SAM3 actually needs to run rather
than tuning its own knobs further.

Caveat: any timing run where the prompt never matches anything only exercises
the "no detection" code path, which is measurably cheaper and can hide bugs
that only trigger once a track is created (as above). Use a prompt you know
will hit, and check the `hits` column, not just fps.

The same configs are also selectable live in `video_ui.py`'s Webcam tab,
which logs each run's fps/hit-rate so you can compare a few by eye against a
real moving object (see below).

## Layout

| File | Responsibility |
|---|---|
| `sam3_runner.py` | Model load + text-prompt inference (returns numpy masks/boxes/scores + timing) |
| `geometry.py` | Mask → centroid / diameter / area; pure numpy, unit-testable |
| `viz.py` | Debug overlay rendering |
| `test_sam3.py` | CLI smoke test wiring the above together |
| `sweep.py` | Runs a set of prompts over `Test Data/`, saves overlays to `Test Data/results/` |
| `ui.py` | Gradio tester — image + prompt + threshold, see the mask |
| `video_runner.py` | Video tracking (`Sam3VideoModel`) — offline `track()` and live `track_frame()` |
| `track_video.py` | CLI: track a concept through a video, write an annotated `.mp4` |
| `video_ui.py` | Gradio video tracker — "Video file" and "Webcam (live)" tabs |
| `bench_realtime.py` | Benchmarks precision/resolution/compile configs against one captured webcam clip |
| `keyframe_hybrid.py` | SAM3 image mode on keyframes + EdgeTAM mask tracking between them; also replays a clip through any backend (`python keyframe_hybrid.py clip.mp4 pen --backend sam3video`) |
| `yoloe_runner.py` | YOLOE text-prompt and seeded-hybrid trackers (SAM3 or SAM 3.1 seeder) — alternatives to `Sam3VideoTracker` in the webcam tab |

Work is logged in [DEVLOG.md](../DEVLOG.md); follow [PLAN.md](../PLAN.md) for next steps.
