# Relish — devlog

Running log, newest first. One entry per session: what changed, what it means,
what's next. For *what we're building*, see [SPECS.md](SPECS.md).

Detailed perception measurements and tracking experiments live in
[temp-devlog.md](temp-devlog.md).

---

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

## 2026-09-15 - SAM 3.1 in picture, video and webcam testing

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

The six pen runs are recorded in [temp-devlog.md](temp-devlog.md). SAM3 fp16
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

**Next:** decide the tracking approach (the four options); then the Quest ↔ PC
transport and the pixel → world projection that fills `Detection`.
