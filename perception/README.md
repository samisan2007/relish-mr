# Perception — Batch 1: SAM 3 local test

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
  Stop logs that run's fps/ms/hit-rate into a rolling log of the last 8 runs,
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

  Both re-ground when YOLOE returns nothing, and optionally every N frames via
  the webcam tab's re-ground slider. The schedule is off by default because
  installing fresh exemplars rebuilds YOLOE's tracker and **restarts the track
  IDs**. On an RTX 3080 at 640x480 a quiet YOLOE frame is ~35ms against ~750ms
  for a SAM 3.1 re-ground, so the interval sets the average rate: ~28 fps at 0
  (lost only), ~21 fps at 60, ~9 fps at 10 (the last measured directly).

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

## Real-time benchmark

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
| `yoloe_runner.py` | YOLOE text-prompt and seeded-hybrid trackers (SAM3 or SAM 3.1 seeder) — alternatives to `Sam3VideoTracker` in the webcam tab |

Work is logged in `../Documentation/devlog.md`.
