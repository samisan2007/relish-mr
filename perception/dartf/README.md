# DARTF webcam experiment

For the optimized FAST experiment on the home RTX 3080, follow
[RTX 3080 setup and recorded-video test](RTX3080.md). It has a separate image,
asset directory and launcher; the native FP16 webcam setup below stays available.

The webcam menu's **DARTF (native SAM3, FP16 TensorRT)** runs an FP16 TensorRT
backbone, detector, segmentation head and SAM3 memory tracker in a GPU Docker
container. It uses full memory (no spatial or query pruning). This is an FP16
reference experiment, not DARTF's advertised W8A8 INT8 configuration.

**Status, 2026-09-14:** ready for experimental webcam testing on this machine.
Docker GPU access, ONNX exports, all six TensorRT engines and the real smoke
check below pass on the RTX 5070. The translated meatball image retains 13
confirmed IDs; a fresh stream restarts IDs at 1. Its last four frames average
549 ms (1.8 fps) including transfer. This is a synthetic 13-object check, not a
live two-pen benchmark or an occlusion test. All 17 unit tests and the
model-stubbed UI check also pass. Keep SAM3 fp16 as the default while comparing
tracking quality on the same recorded two-object clip.

The normal Windows perception environment still owns the webcam and overlays.
Frames and masks travel over the container's standard input/output. The run log
includes that transfer cost, but excludes model initialization and UI rendering.
Each Start creates a new tracker; Stop removes its container and GPU state.

## Setup

Requires Docker Desktop with Linux containers and GPU access. The initial image,
PyTorch dependencies and official checkpoint require several GB of downloads.
TensorRT engines are generated locally for this GPU/runtime combination.

From the repository root, in PowerShell:

```powershell
git clone https://github.com/mkturkcan/DART.git ..\DART
git -C ..\DART checkout 16fada39054ac5058f6e7c1e8748cb9cc288f90c
.\perception\.venv\Scripts\hf.exe download facebook/sam3 sam3.pt --local-dir ..\DART\weights
docker build -t relish-dartf:local perception/dartf
```

The exporter also uses the Hugging Face SAM3 checkpoint already cached by the
existing image tester. With the default Hugging Face cache location:

```powershell
$dartRoot = (Resolve-Path ..\DART).Path
$perceptionRoot = (Resolve-Path perception).Path
$hfCache = Join-Path $env:USERPROFILE '.cache\huggingface\hub'
$dartAssets = Join-Path $dartRoot 'dartf\assets'
New-Item -ItemType Directory -Force $dartAssets | Out-Null
docker run --rm --gpus all --name relish-dartf-build `
  --mount "type=bind,source=$dartRoot,target=/dart,readonly" `
  --mount "type=bind,source=$perceptionRoot,target=/app,readonly" `
  --mount "type=bind,source=$hfCache,target=/hf/hub,readonly" `
  --mount "type=bind,source=$dartAssets,target=/assets" `
  --env HF_HUB_OFFLINE=1 --entrypoint python relish-dartf:local `
  /app/dartf/build_engines.py
```

Export/build is resumable: successful stages and finished plans are reused.
Use `--stage backbone|text|heads|tracker|plans` to rerun a particular stage while
diagnosing a failure. Tracker plans batch two objects at a time for the 12 GB GPU;
additional objects are processed in chunks, not discarded.

The source, checkpoint, image and all engines were prepared on the RTX 5070
test machine. A new PC needs its own setup and GPU-specific engine builds;
restart the UI after setup to select DARTF. These large local artifacts are not
committed. To continue an interrupted build elsewhere, rerun the `docker run`
command above after setup. A stage
marker is written only after that export succeeds, and engine files are renamed
into place only after a successful build. When changing export code or engine
profiles, use a fresh assets directory to avoid reusing stale outputs.

The expected default layout is:

```text
Relish/
  DART/
    weights/sam3.pt
    dartf/assets/       # generated plans, ONNX files, tokenizer and positions
  relish-mr/
    perception/dartf/
```

`DART_ROOT` and `DARTF_ASSETS` can override those locations. Restart the video UI
after setting them. Select DARTF, enter one text prompt (for example `pen`), and
press Start. Initialization can take longer than an ordinary SAM3 restart.

The first test should use a fixed image and then a recorded two-object clip.
Check identities through crossing, rotation and occlusion before comparing speed;
`hits` and distinct ID totals alone do not establish correct tracking.

From `perception/`, the synthetic translation and restart check is:

```powershell
.\.venv\Scripts\python.exe tests\smoke_dartf.py ..\..\Media\meatballs_img.jpg meatball
```

## Upstream

Based on [DART/DARTF](https://github.com/mkturkcan/DART/tree/16fada39054ac5058f6e7c1e8748cb9cc288f90c/dartf).
The setup script exports a named backbone trunk for the native tracker and saves
the temporal-position buffer missing from the upstream tracker export. Export
scripts run in separate processes because upstream temporarily changes PyTorch's
device behavior during CPU export.
