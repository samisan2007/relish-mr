# Relish — specs

Living document. What we are building and the decisions that are settled.
Update it when a decision changes, not when work happens (that's [DEVLOG.md](DEVLOG.md)).

**Status:** prototype. Nothing shipped, nothing frozen.

## The product

Mixed-reality cooking assistant on Quest. The headset sees the food through
passthrough, and the app tells the cook about portions in place, on the food,
rather than on a screen.

## Architecture

```
Quest (Unity)                        PC (Python)
  passthrough camera  ──frames──▶  perception: segment + track the food
  PortionWidget       ◀──sizes───  per-object mask → real-world diameter
```

Split brain, deliberately: the models that actually understand food nouns are
too heavy for the headset, so the Quest is a sensor and a display, and the PC
does the seeing.

### Settled

- **Unity 6000.6.0f1, URP, Meta XR SDK v205.**
- **Building blocks in the scene:** Camera Rig, Hand Tracking (left + right),
  Passthrough, Passthrough Camera Access.
- **No on-device inference.** The Meta Image Segmentation building block only
  offers Yolo11n-seg — 80 fixed COCO classes, no food nouns. Rejected 2026-09-09.
- **Perception runs on the PC** in Python (`perception/`), testing SAM3,
  SAM 3.1 Object Multiplex, YOLOE and DARTF.
- **The test UI defaults to SAM3 video, `fp16 autocast, 1008px`.** This is the
  current testing default, not a final production tracking choice.
- **The widget is decoupled from perception.** `PortionWidget` takes a world
  position and a diameter in metres and knows nothing about how they were derived.

### Open

- **Tracking approach.** No final backend is chosen. Recent pen runs report
  SAM3 fp16 at 3.8 fps, image + ByteTrack at 4.5 fps, Hybrid at 21.6 fps and
  text YOLOE at 31.7 fps. Those runs used different frames, and neither hit
  counts nor distinct IDs establish correct two-object tracking. Compare the
  same recorded clip through crossing, rotation and occlusion; measurements
  and limitations are in [temp-devlog.md](temp-devlog.md). DARTF is an optional
  experiment with a passing synthetic translation/restart check; live tracking
  quality still needs evaluation.
- **Food-tracking acceptance target.** Aim for at least 8-10 fps with useful
  masks and correct identities through motion, crossing and brief hand
  occlusion. SAM 3.1's initial successful offline test reached 5.65 fps compiled
  on the RTX 5070; on the home RTX 3080 the same worker path measured 2.68 fps
  eager and 3.42 fps compiled. Live and end-to-end timings must be measured
  separately. False positives and missed food instances remain unresolved, so no
  final backend is selected.
- **Quest ↔ PC transport.** Not built. No protocol, no codec, no latency budget.
- **Where the widget gets anchored.** Screen-space projection of the mask
  centroid, a depth hit, or an MRUK anchor — undecided.
- **What the widget actually communicates.** Diameter is a placeholder for
  "portion". Grams, calories, or a serving count are all unspecified.

## Unity side

`Assets/Relish/`

| Piece | What it is |
|---|---|
| `Scripts/PortionWidget.cs` | The widget. Sphere sized in real-world metres, minimize/restore, success pulse, stale dim, debug label + gizmos. |
| `Scripts/Hands.cs` | Finds the hand-tracking hands so nothing else hunts for the rig. Pose, pinch, pinch point. |
| `Scripts/PortionWidgets.cs` | The manager. `Report(frame)` of `Detection { Id, Position, Diameter, Score }` → spawns, lerps between reports, dims stale, retires lost. `Detection` is the contract with perception. |
| `Editor/RelishWidgetBuilder.cs` | `Relish > Create Widget Prefab` — generates the prefab and its transparent URP material. |
| `Editor/ContextMenuButtonsEditor.cs` | Turns every `[ContextMenu]` test method on the two scripts into an inspector button (Play mode only). |

The widget is a sphere *for now*. The API is the contract; the mesh is not.

## Perception side

`perception/` — SAM3 image + video testers, YOLOE trackers, a replay benchmark,
and a webcam UI that switches backends. See [temp-devlog.md](temp-devlog.md) for
the measurements and the reasoning behind them.

The webcam menu offers SAM3 video, SAM 3.1 (normal/compiled), SAM3 image + ByteTrack,
YOLOE text, Hybrid (SAM3 seed to YOLOE), Hybrid (SAM 3.1 seed to YOLOE), and
experimental DARTF. Either hybrid re-grounds when it loses the object, and on a
selectable frame interval; each re-ground restarts that run's track IDs. All adapters return a per-frame
`list[Instance]` containing a pixel mask, box, confidence and optional object ID.
This does not yet supply Unity's world position or diameter.

SAM 3.1 is also selectable in the picture and video-file modes. It uses the
pinned official 3.1 checkpoint and source in an optional Linux GPU Docker
worker, BF16 attention through PyTorch, and up to 16 tracked regions. Models
load on selection; choosing SAM 3.1 releases the UI process's other model caches.
Picture mode applies the confidence threshold to detection admission and output.
Both video modes process incoming frames causally, retain the model's object
memory, and discard memory older than 32 frames except the first conditioning
frame per bucket. This differs from the offline demo and may affect long
occlusions. Each run has fresh IDs, and Stop/failure closes its worker and camera,
including during compilation. File and webcam GPU work is serialized in the UI.
Compilation is opt-in and can take minutes on new shapes. Recorded-video
results distinguish playback FPS, total processing speed and request speed after
the first eight frames (which can still include later compilation). Webcam
request timing includes Docker transfer but excludes capture,
overlays and browser delivery. See [SAM 3.1 setup and tests](perception/sam31/README.md).

DARTF is optional and loads only when selected. Its native SAM3 memory tracker
and FP16 TensorRT detector run in a GPU Linux Docker container; the Windows
process handles camera capture and overlays. Engines must be built locally for
the GPU/runtime. Full memory is retained, with additional objects processed in
batches of two. This experiment does not implement the upstream W8A8 recipe.
Missing assets or startup failures must produce a UI error and release the
camera. Real inference, mask transfer and restart pass the synthetic smoke
check on both the RTX 5070 and the home RTX 3080, which builds its own SM86
engines. See [setup and validation](perception/dartf/README.md); identity
continuity through real motion and occlusion still needs evaluation before
recommending this backend.

A separate [RTX 3080 FAST experiment](perception/dartf/RTX3080.md) prepares the
W8A8 detector, fused mask head, lightweight tracker and frame pipeline. Its
first test is a recorded-video run through the upstream pipeline, with separate
headless and rendered timings plus saved IDs. It is not yet a webcam menu mode
or a selected production backend. SM86 engine execution and tracking quality
must be checked on the home GPU; the upstream RTX 4090 FPS claim is not a local
performance target or guarantee.
