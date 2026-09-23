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

## Ideas to explore

Survey of tools, repos and research, 2026-09-23. **None of this is decided.**
Each item is a candidate to test, with its source. Claims are the authors' own;
none has been measured here.

### 1. Anchor in world space and stop tracking in the image

This is the biggest lever, and [temp-devlog.md](temp-devlog.md) already raises
it ("The question this test actually raises"). Tag every frame sent to the PC
with its capture timestamp, camera pose and intrinsics. Then lift each detection
into world space on the Quest, using the pose from *that* frame. From there:

- Head motion stops being a perception problem, and network and inference
  latency stop moving widgets, because each result is placed using its own
  frame's pose.
- Identity becomes nearest-match by world distance with a gate, in
  `PortionWidgets`. Perception no longer has to supply IDs.
- Food sitting on a counter needs perception at roughly 1-4 Hz, not 8-10.
  The 8-10 fps target only applies to food moving in the world, such as food
  in the cook's hands (see item 5).

Tools already in the SDK we use:
- MRUK `PassthroughCameraAccess` gives precise frame timestamps for
  camera-to-world alignment, and both cameras at once.
  [PCA samples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples)
- MRUK `EnvironmentRaycastManager` (MRUK 85+) raycasts a 2D pixel to a world
  hit using the depth map. The CameraToWorld sample shows the pixel-to-ray
  math. [Environment raycast docs](https://developers.meta.com/horizon/documentation/unity/unity-mr-utility-kit-environment-raycast/)

Metric diameter is then `diam_px · depth / focal_px`. The PC can stay purely 2D.

### 2. Reuse an existing transport instead of designing one

- [Unity-QuestVisionStream](https://github.com/danieloquelis/Unity-QuestVisionStream):
  MIT, beta. PCA over WebRTC to a Python server that runs YOLO, Florence-2 or
  GroundingDINO, with detections returned into Unity as 3D tags. We would swap
  in SAM3. Its docs don't say whether frames carry the camera pose, so check.
- [QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit): MIT, on
  **MRUK/Meta XR 205, the same version we use**. It has samples for WebRTC
  streaming, camera-to-world raycasting, object detection with 3D markers, and
  stereo UV calibration for both cameras.
- The simplest alternative: a WebSocket carrying a JPEG plus a JSON header
  (timestamp, pose, intrinsics). At 1-4 Hz and 640 px we don't need a video
  codec.

### 3. Faster models in the SAM3 family

- [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) (Apache-2.0):
  - Distilled students of about 90M parameters (EV-M, RV-M, TV-M), with text
    prompts and video tracking. Stage-3 fine-tuned weights were released in
    June 2026.
  - SAM3-LiteText keeps SAM3's vision encoder and cuts the text encoder by 88%.
  - No speed figures are published, so we'd benchmark on the 3080 and check
    that food nouns still work.
- [EOVSAM](https://arxiv.org/abs/2608.02284) (hustvl, August 2026): runs SAM3's
  open-vocabulary segmentation in one pass. It claims up to 338× faster than
  vanilla SAM3, and code is released. It is the newest candidate. Its main
  risk: the authors may have measured at larger vocabularies than one prompt.
- [YOLOE-26](https://arxiv.org/abs/2602.00168): text-prompt accuracy +2-3 AP
  over the YOLOE-11 we use, and a drop-in change in Ultralytics. It is still a
  lightweight text encoder, so "meatball" may stay weak. Cheap to retest.
- SAM 3.1 (March 2026) is still Meta's latest checkpoint. Nothing newer was
  found. [Meta blog](https://ai.meta.com/blog/segment-anything-model-3/)

### 4. Mask trackers for moving food, only if needed

These are the memory-based version of the hybrid idea. The current YOLOE hybrid
re-detects each frame by matching an exemplar. These trackers instead propagate
a mask with temporal memory, can be seeded by a SAM3 mask, and don't depend on
the object's class.
- [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) (Meta, CVPR 2025):
  22× faster than SAM 2, 16 fps on an iPhone 15 Pro, J&F close to SAM 2.
  [Already in HF transformers](https://huggingface.co/docs/transformers/en/model_doc/edgetam).
- [EfficientTAM](https://arxiv.org/pdf/2411.18933): about 2× faster than SAM 2,
  with under 2 J&F lost.
- [StreamDAM](https://arxiv.org/html/2608.03912) (August 2026): paper only, no
  code. It runs a SAM 2-family tracker at about 30 fps and uses a learned
  "presence" signal to decide when to re-detect. Relevant to our re-ground
  triggers, which are currently a fixed interval plus "zero detections".

### 5. Use hand tracking

- **Reject hand and wrist false positives.** Project the Quest hand joints into
  each frame and drop masks that overlap them. It costs no extra inference.
- **Carry widgets while the cook holds the food.** A widget near a grasping hand
  follows the hand's joints at the tracking rate, which covers the
  rolling-a-meatball case without fast perception.
- Research, for reference only:
  - [FoodTrack](https://arxiv.org/abs/2505.04055): portion estimation for
    handheld food from egocentric video.
  - [EgoGrasp](https://arxiv.org/pdf/2601.01050): hand-object interaction in
    world space.
  - [In-hand object segmentation](https://arxiv.org/html/2509.26004v1).

### 6. Depth, for size and volume

- **Quest Depth API:** low resolution, and weak in the near field. Probably
  good enough to raycast a centroid, since the error is about half the object's
  height. [Depth API](https://developers.meta.com/horizon/documentation/unity/unity-depthapi-overview/)
- **Stereo on the PC:** PCA can stream both cameras.
  [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
  (NVIDIA, CVPR 2026) does real-time zero-shot stereo. That gives dense metric
  depth of the plate, which is what volume needs.
- **Monocular metric depth, fallback only:**
  [Depth Anything 3](https://arxiv.org/pdf/2511.10647) (DA3-metric) and
  [Depth Pro](https://arxiv.org/html/2410.02073v1).

### 7. What a "portion" means: grams, calories, servings

- **Ask a VLM.** A [2026 benchmark on Nutrition5k](https://pmc.ncbi.nlm.nih.gov/articles/PMC13483877/)
  of ten models:
  - Gemini 3.0 Flash was the most accurate on calories (MAE 80.7 kcal,
    CCC 0.767).
  - Gemini 3.1 Flash Lite was nearly as good at the lowest cost.
  - Claude Haiku 4.5 and GPT-5 Mini were also tested.
  - The authors rate these models good enough for consumer calorie tracking,
    not clinical use.

  Idea: send each object's crop together with its measured metric size.
- **Compute it from geometry.** Mask plus depth above the table plane gives
  volume, and a density table turns volume into grams.
  - [Size Matters](https://arxiv.org/html/2601.20051) (Purdue, 2026): MAPE of
    about 36% on MetaFood3D.
    [Code](https://gitlab.com/viper-purdue/size-matters)
  - [VolE](https://www.nature.com/articles/s41598-026-38756-5)
  - [MonoBite](https://link.springer.com/chapter/10.1007/978-981-95-5737-0_3)
  - [SAM 3D Objects](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)
    gives full shape, but not metric scale. Our depth would have to supply
    the scale.
- **Compare before and after.** [DietDelta](https://arxiv.org/pdf/2604.06352)
  estimates how much was taken or eaten.

### 8. Adjacent

- [Gemini Live API](https://github.com/google-gemini/gemini-live-api-examples):
  streams video and voice, about 600 ms to the first token. Could become a
  "what's this / how much is left?" voice layer.
- [Memory-augmented AR agents](https://arxiv.org/pdf/2508.08774): an agent that
  remembers where objects were and what happened to them, for help with
  multi-step tasks.

### Cheapest experiments, in order

1. Run the QuestCameraKit or PCA CameraToWorld sample in our scene and raycast
   a fixed pixel to a world point.
2. Build pose-tagged frame transport (item 2), with SAM3 image mode on the PC
   returning centroids and pixel diameters.
3. Match objects by world distance in `PortionWidgets`, then test head motion
   against a static plate.
4. Filter out masks that overlap the projected hands.
5. Benchmark EfficientSAM3, EOVSAM and YOLOE-26 on the same food clips. By this
   point those can be real Quest recordings with their pose logs.
6. Decide what a portion is, then try the VLM route before the geometry route.

If step 3 holds up, the tracking question under **Open** reduces to food in the
cook's hands. DARTF, the hybrids and the 8-10 fps target could then be shelved.
