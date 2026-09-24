# Offline model preparation

Compare mask propagation before integrating a new backend into the UI. Official
EdgeTAM and SAM 2.1 tiny receive identical frozen SAM3 seed masks; Transformers
EdgeTAM is the reference implementation. EfficientSAM3 EV-M is tested separately
as a text-prompted image detector. Its upstream Stage 2 memory-weight release is
still unchecked in the pinned README; this harness does not test EV-M video.

## Setup

From the repository root in PowerShell, with Docker Desktop's Linux GPU support
and `relish-sam31:local` already prepared ([instructions](../sam31/README.md)):

```powershell
powershell -ExecutionPolicy Bypass -File perception/candidates/run.ps1 -Action Setup
```

Source and checkpoint revisions are pinned in [models.json](models.json).
Setup downloads public checkpoints into ignored `perception/candidates-local/`
and builds `relish-candidates:local`, leaving the Windows environment alone.
The base image is a local dependency, not a portable lockfile; each run records
the actual image ID and installed packages. Setup may access the network;
inference containers use `--network none` and offline Hugging Face settings.

Official EdgeTAM already batches ordinary object propagation. Its pinned source
has a multi-object crash: `expand(...).view(...)` cannot flatten a strided tensor.
Setup changes that single `view` to `reshape` in `sam2/modeling/perceiver.py`.
The harness verifies the exact patched contents and rejects unrelated source
changes. The manifest records the patch. This is not a new batching implementation.
Native postprocessing is disabled to avoid requiring optional compiled CUDA
extensions. The timm ImageNet initialization is skipped because the complete
EdgeTAM checkpoint replaces it with strict loading.

## Freeze a shared tracking case

Choose a video frame where the target objects are visible:

```powershell
perception/.venv/Scripts/python.exe -B perception/candidates/prepare.py ../Media/pen_test_vid.mp4 pen --start 150 --frames 60
```

This uses the existing cached SAM3 checkpoint. It saves numbered JPEGs, seed
masks/IDs, `seed-overlay.png`, and a manifest with hashes. **Inspect the overlay**:
these are detector proposals, not ground truth. Objects are ordered by decreasing
seed confidence; `-Objects N` selects the first N. Use an earlier/different start
or case if the selected proposals are wrong. A photo input instead creates a
small translated sequence, useful only for checking execution and timing.

Copy the printed case directory into `$casePath`:

```powershell
$casePath = 'perception/candidates-local/cases/PASTE-PRINTED-DIRECTORY'
powershell -ExecutionPolicy Bypass -File perception/candidates/run.ps1 -Action Track -Model edgetam-hf -InputPath $casePath -Objects 3
powershell -ExecutionPolicy Bypass -File perception/candidates/run.ps1 -Action Track -Model edgetam -InputPath $casePath -Objects 3
powershell -ExecutionPolicy Bypass -File perception/candidates/run.ps1 -Action Track -Model sam21tiny -InputPath $casePath -Objects 3
```

Run one GPU job at a time. Repeat with 1 and approximately 10-12 inspected objects
to examine scaling. The harness rejects changed frames/seeds and insufficient
seed counts. Cases contain 10-120 frames: native predictors retain clip state,
so this bounded harness is not a long-run memory test. Two independent sessions
run by default; change `-Repeats` if needed.

Each unique `candidates-local/runs/` folder contains `environment.json`, an
annotated MP4, per-frame masks/IDs and timing CSVs. Raw masks retain original
dimensions; the preview encoder may resize to codec-compatible dimensions.
Frame request mean/p50/p95 exclude the first eight frames, preprocessing,
initial seeding, rendering and file writes. CPU mask transfer is included.
Total wall time is recorded separately. Track overlay scores of 1 are placeholders.

The HF path keeps the current mask-seed score fix, presence gate and memory
pruning. Native paths use their own presence behavior and retain full clip state.
Thus implementation comparisons include these differences, not just batching.
No re-grounding, association, late arrivals or keyframe stalls are exercised.
Stable slot IDs and nonempty masks do not establish identity correctness.

## Test EV-M food vocabulary

```powershell
powershell -ExecutionPolicy Bypass -File perception/candidates/run.ps1 -Action Image -Model evm -InputPath ../Media/meatballs_img.jpg -Prompt meatball -Threshold 0.4
```

The fine-tuned checkpoint strictly loads EfficientViT B1 + MobileCLIP S0,
context length 16, 1008-pixel input. Missing required tensors are an error;
extra unused checkpoint heads are allowed. Results include masks, scores,
overlays, cold/warm request timings and peak allocated GPU memory.
Image requests include image preprocessing and text encoding, excluding model
loading and output writes. Compare visible food coverage and false positives
against SAM3 on the same photos, not throughput alone. Thresholds are not
calibrated across models. An empty fast result is not a successful detector.

## What the recordings still need to establish

Use short clips with an initially clear view, then kneading, crossing, touching,
full hand occlusion, departure and return. Include 1, 3 and many pieces; preserve
the original resolution and frame rate. Keep separate tuning and evaluation
clips, and label selected masks/physical identities independently of SAM3.

Before UI integration, evaluate missed food, merged masks, ghosts, ID switches
and recovery delay; then test re-grounding and state reset. Quest work adds
capture timestamps, pose/intrinsics, transport, dropped frames and display age.
These are separate from this offline preparation. See [PLAN.md](../../PLAN.md)
for acceptance criteria and [DEVLOG.md](../../DEVLOG.md) for measured results.
