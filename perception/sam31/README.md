# SAM 3.1 on the RTX 5070

Meta's official SAM 3.1 Object Multiplex tracker is available in the image,
video-file and live webcam UIs, plus a separate offline benchmark. It uses the
existing Linux GPU environment and the tested Windows/Docker frame transport.

## Setup

From `relish-mr/` in PowerShell, with Docker Desktop running Linux containers:

```powershell
powershell -ExecutionPolicy Bypass -File perception\sam31\run.ps1 -Action Setup
```

This builds `relish-sam31:local`, downloads the pinned official source and the
3.5 GB checkpoint, and reuses completed downloads. It reads the existing Hugging
Face login from its default location. If access is denied, accept the terms at
[facebook/sam3.1](https://huggingface.co/facebook/sam3.1), log in using the existing
perception environment's `hf auth login`, then repeat Setup.

Source, model cache, compiler cache and results are ignored under
`perception/sam31-local/`. Setup does not install packages into the Windows venv.

## Test in the UI

Setup is already complete on this RTX 5070. Keep Docker Desktop running Linux
containers, and run one GPU test at a time. From the repository root:

| Mode | Launch | Test |
| --- | --- | --- |
| Picture | `run_sam_img.cmd` (port 7860) | Upload a photo, select **SAM 3.1**, enter a food noun and Submit. |
| Video file | `run_sam_vid.cmd` (port 7861) | In **Video file**, select **SAM 3.1**, upload a clip, enter a noun and Track. Start with 60 frames and stride 1. |
| Webcam | `run_sam_vid.cmd` (port 7861) | In **Webcam (live)**, select **SAM 3.1**, enter a noun, use camera index 0 and Start. Stop records the run and releases the camera/worker. |

Each mode also offers **SAM 3.1 (compiled)**. Start with normal mode for quick
checks. Compiled first frames can take several minutes; webcam Stop can cancel
model loading or compilation. A fresh worker loads for every picture/file/Start,
so model startup is paid again; downloaded weights and compiled kernels are
cached. SAM3 remains the default and loads only when selected. Selecting SAM 3.1
releases other model caches in that UI process to leave room on the 12 GB GPU.

Picture mode uses the 3.1 checkpoint on one image. Its confidence slider controls
detection admission and filters the returned regions. Both video modes retain
the official object memory while processing only the current frame, with no
future-frame prefetch. They cap tracking at 16 regions and retain the latest 32
memory frames plus the first conditioning frame per object bucket. This bounded
history can affect long occlusions. No reverse propagation or editing past
frames is provided by the UI adapter. Each Start resets IDs.

File results distinguish output playback FPS, total processing speed (including
startup and encoding), and frame-request speed after the first eight frames
(including Docker transfer and any later compilation). Webcam timing includes
the frame request and transfer, but excludes
camera capture, drawing and browser delivery. The initial run log also includes
compilation/first-frame costs. Check visible IDs through motion and brief hand
occlusion; faster empty output is still a recognition failure.

The GPU smoke exercises the real UI handlers with a synthetic camera source:

```powershell
cd perception
.\.venv\Scripts\python.exe tests\smoke_sam31_ui.py
.\.venv\Scripts\python.exe tests\smoke_sam31_ui.py --compile
```

The initial compiled UI smoke passed picture segmentation, 40 video frames and
two webcam Start/Stop cycles. It measured 2.25 fps on file frame requests after
the first eight frames (including transfer and later compilation), and 155.2 s
for the tracking/encoding loop including worker startup. Twelve region IDs persisted after
confirmation. Short compiled webcam runs were dominated by compilation. These
measurements do not establish 8-10 fps live tracking. The physical camera at
index 0 also opened and supplied 640 x 480 frames. Motion/occlusion quality in a
real food clip remains unverified.

Normal mode also passes all three handlers and two webcam restarts, including
food appearing after three initially empty frames. File requests after the first
eight frames measured 2.41 fps; the final four webcam requests measured 2.58 and
2.63 fps. These include Windows/Docker transfer and are short functional checks.
Artifacts: `sam31-local/ui-smoke-20260915-114526/` (compiled) and
`sam31-local/ui-smoke-20260915-115520/` (normal).

## Offline photo/clip benchmark

Close other GPU inference runs before testing. This card has 12 GB of VRAM.

The synthetic smoke test moves the existing meatball photo across a black frame:

```powershell
powershell -ExecutionPolicy Bypass -File perception\sam31\run.ps1 -InputPath ..\Media\meatballs_img.jpg -Prompt meatball -Frames 24 -Render
```

For a practical check, record a short clip with Windows Camera: one or two pens
or pieces of food, with rotation, crossing and a brief hand occlusion. Close
Camera and run:

```powershell
powershell -ExecutionPolicy Bypass -File perception\sam31\run.ps1 -InputPath "C:\Videos\food.mp4" -Prompt "tomato" -Frames 60 -Render
```

It processes the first 60 frames, twice, using a fresh tracking state each time.
The limit is 120 frames per run to bound video memory. `-MaxObjects` caps tracked
instances (1–16, default 16); it is recorded with each result. A single noun can
match multiple instances. Start with one prompt and a few objects.

`-Compile` requests upstream's PyTorch compilation. Its first pass can take much
longer and use more memory; compare the second pass after compilation. Failures
are reported, so an unsuccessful compiled run is not a performance result.
The first compiled tray test took about ten minutes to compile its initial
prompt and tracking shapes. Cached kernels help later runs, but a new shape can
still trigger compilation, including when a new tracking state is created.

## Measured on this RTX 5070 (2026-09-15)

Two fresh passes over 24 translated-photo frames at 640 x 480, using the
configuration below. This table uses pass 2, frames 8 through 22, and PyTorch's
peak allocated GPU memory:

| Photo / prompt | Mode | Tracked regions | Steady FPS | Peak GiB |
| --- | --- | ---: | ---: | ---: |
| Meatball tray / `meatball` | Eager | 12 | 3.88 | 5.03 |
| Meatball tray / `meatball` | Compiled | 12 | 5.65 | 5.01 |
| Single food ball / `meatball` | Compiled | 0 | Recognition failed | 4.38 |
| Single food ball / `ball` | Compiled | 0 | Recognition failed | 4.38 |

The tray retained IDs 0 through 11 in both modes and both passes, but visual
inspection found hand/wrist false positives. These are tracked regions, not a
ground-truth count of food. The single-food negative runs processed empty output
at 7.13 and 7.10 fps; those are not successful tracking results. The current successful
test is below the 8-10 fps target.

Local run folders: eager tray `20260915-080827-743481`, compiled tray
`20260915-081039-394486`, single-food negatives `20260915-082323-890435`
and `20260915-082612-680908`.
The tray summaries were recomputed from their saved per-frame timings to exclude
the final prefetched frame; their original console summaries included it.

## Results and interpretation

Each run creates `perception/sam31-local/runs/<timestamp>/` containing:

- `environment.json`: exact revisions, GPU, precision and replay settings.
- `pass-1/summary.json`, `pass-2/summary.json`: timing, peak PyTorch VRAM and ID
  summaries. Steady timing excludes the first eight propagation frames and the
  final frame, whose detector result is already cached with no next-frame work.
- `tracks.json` in each pass: per-frame IDs, boxes, scores and timings.
- First/last annotated PNGs, and `annotated.mp4` when `-Render` is selected.
- The decoded JPEG inputs and model construction log for reproduction.

Timing includes GPU-synchronized model propagation and upstream postprocessing.
Input decoding, initial prompting, our overlays/video encoding, and transfer
back to Windows are excluded. MP4 playback uses the source frame rate (10 fps
for synthetic input); it does not show how fast inference ran. These are offline
replay measurements, not measured webcam or Quest latency.
The upstream detector still prefetches one frame ahead in this configuration.

Inspect the visible IDs and masks. A persistent ID in the synthetic test checks
wiring and basic continuity; it does not prove identity correctness during real
occlusion. Empty results in a real clip remain a recognition failure even if
processing is fast.

## Exact offline benchmark configuration

- Official source: `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.
- `facebook/sam3.1` checkpoint: `daa63191845a41281374e725f4c9e51c7a824460`.
- BF16 autocast, 1008 px model input, 16 slots per multiplex bucket.
- `use_fa3=False`: PyTorch attention, avoiding the optional FlashAttention 3
  extension. Eager inference is the default.
- Grounding and postprocessing run one frame at a time; the 15-frame hotstart
  output delay is disabled. This changes the demo's initial pruning behavior.
  It avoids the official demo's 16-frame lookahead/batches for this live-oriented
  replay on a 12 GB GPU.
- The pinned generic session wrapper forwards `offload_state_to_cpu` to a
  multiplex method that does not accept it. The benchmark calls the upstream
  model's `init_state`, `add_prompt` and `propagate_in_video` directly, and clears
  each state after use. No upstream source patch is required for this mismatch.

See [Meta's release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
