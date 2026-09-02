# Perception — Batch 1: SAM 3 local test

Python 3.12 venv (PyTorch CUDA wheels don't exist for the system Python 3.14).

## One-time setup

1. Request access to the gated model at https://huggingface.co/facebook/sam3
   (instant-ish approval after accepting the license).
2. Create a token at https://huggingface.co/settings/tokens (read scope) and log in:

   ```powershell
   .\.venv\Scripts\hf.exe auth login
   ```

## Run the tester

From the `perception/` folder:

1. **Start** — takes ~10s to load the model. From the repo root just run
   `.\run_test.cmd` (or double-click it); the equivalent from here is:

   ```powershell
   .\.venv\Scripts\python.exe ui.py
   ```

2. **Test** — open <http://127.0.0.1:7860>. Drop in an image, type a noun
   phrase ("meatball", "pot", "lemon"), hit Submit. Toggle masks/boxes and
   drag the threshold to taste. Each run takes ~0.5s.

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

- **Video file** — the offline path above, better quality.
- **Webcam (live)** — streaming inference. Enter a prompt, pick a **Benchmark
  config** (precision/resolution/compile — same options as `bench_realtime.py`
  below), press Start. Stop logs that run's fps/ms/hit-rate into a rolling
  log of the last 8 runs, so you can flip configs and compare by eye against
  a real moving object instead of only trusting the fixed-clip numbers.
  Switching config takes effect on the next Start and reloads the model
  (~5-10s) the first time it's used. Streaming disables the heuristics that
  prune duplicate tracks, so expect more false positives than the file tab.

First ever run downloads ~3.5 GB of model weights to the Hugging Face cache.

## Real-time benchmark

`bench_realtime.py` captures one webcam clip, then replays those exact frames
through several `Sam3VideoTracker` configs (precision, dispatch mode, internal
processing resolution, `torch.compile`) so speed comparisons aren't muddied by
webcam variance between runs. The processor always resizes every frame to a
fixed 1008x1008 before it reaches the model, so capture resolution isn't the
cost knob — precision and that internal resize target are.

```powershell
.\.venv\Scripts\python.exe bench_realtime.py --prompt "phone" --capture-seconds 8
```

Move the object during the 3s countdown so the clip has real motion to judge
mask quality against. Or from the repo root: `run_sam_bench.cmd "phone" 8`.

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
- **`torch.compile`** — still fails, now on a missing Triton install (Triton
  doesn't officially support Windows). Left in `CONFIGS` as an experimental
  config that fails cleanly rather than crashing the run.

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

Work is logged in `../Documentation/devlog.md`.
