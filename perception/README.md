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

First ever run downloads ~3.5 GB of model weights to the Hugging Face cache.

## Layout

| File | Responsibility |
|---|---|
| `sam3_runner.py` | Model load + text-prompt inference (returns numpy masks/boxes/scores + timing) |
| `geometry.py` | Mask → centroid / diameter / area; pure numpy, unit-testable |
| `viz.py` | Debug overlay rendering |
| `test_sam3.py` | CLI smoke test wiring the above together |
| `sweep.py` | Runs a set of prompts over `Test Data/`, saves overlays to `Test Data/results/` |
| `ui.py` | Gradio tester — image + prompt + threshold, see the mask |
| `video_runner.py` | Video tracking (`Sam3VideoModel`) — text prompt → masks with stable object IDs |
| `track_video.py` | CLI: track a concept through a video, write an annotated `.mp4` |

Work is logged in `../Documentation/devlog.md`.
