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

### 9. Scene/activity understanding — what is the cook doing

A different question from segmentation: not "where is the food" but "what
action is happening, on what surface" (e.g. cutting lettuce on a wood board).
This is a captioning/VLM task, not a detector, so it doesn't need a dataset or
training. Send a frame (or a short recent window) to a vision-language model
at low rate, maybe 0.5-1 Hz, and ask directly.

- **[Gemini Robotics ER 2](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview)**
  is purpose-built for this: it does video "moment finding and progress
  classification," i.e. it can say what step of a task is happening right now,
  not just caption a still. Has a streaming Live API variant
  (`gemini-robotics-er-2-streaming-preview`). The closest off-the-shelf answer
  to this question. Pricing and latency aren't documented on that page.
- **Gemini 3 Flash / GPT-5**, general-purpose, prompted directly on sampled
  frames. Likely the same call already used for the portion/calorie question
  (item 7) could return the action too — one model call, two answers.
- **[DeepSeek `deepseek-flash`](https://api-docs.deepseek.com/guides/vision/)**:
  vision-capable, but **static images only, no video/streaming input**, and no
  documented object detection or grounding. Would work as a periodic-frame
  captioner (same shape as the sampled-frame approach above) but not for
  anything needing continuous video or spatial grounding. No pricing/latency
  in the docs; check separately before relying on it.
- **[Qwen3-VL](https://arxiv.org/pdf/2511.21631)** (4B/8B/30B/235B, open
  weights): the 8B is reported competitive with much larger closed models on
  video benchmarks. Relevant if this should run locally instead of a cloud
  API — the project already runs everything else on the local PC.
- **[Moondream 3](https://moondream.ai/blog/moondream-3-preview)** (9B MoE,
  2B active): small, fast, self-hostable, and does *grounded* reasoning — it
  can point at the part of the image it's reasoning about, not just emit text.
  Worth trying if latency/cost matters more than accuracy.
- Dedicated egocentric action-recognition research exists
  ([EPIC-KITCHENS-100](https://epic-kitchens.github.io/), verb+noun labels
  like "cut lettuce") but means training a closed-vocabulary classifier.
  A general VLM prompted well should beat this on open-ended food/action
  variety for an API call instead of a training run; only worth it if a VLM
  proves too slow/expensive/inaccurate in practice.
  [Vinci](https://arxiv.org/pdf/2412.21080) is a real-time egocentric
  assistant built for step-by-step task guidance, closer to this product's
  shape than a generic VLM, worth reading before dismissing this path.

### 10. Reviewed links: relevance notes (2026-09-23)

Ranked by usefulness to us.

- **[EPFL-Smart-Kitchen](https://cnai.epfl.ch/EPFL-Smart-Kitchen/): high.**
  29.7 h of 16 people cooking 4 recipes, with a HoloLens 2 egocentric camera
  plus 9 RGB-D cameras, hand pose, body motion, eye gaze and IMUs. It has
  60,189 dense action labels: 33 verbs × 79 nouns, 763 fine-grained actions.
  Benchmarks cover vision-language, action recognition and pose-based action
  segmentation. It is the closest public match to our setup: head-mounted MR,
  a kitchen, hand pose. Use it as a **test set**. We could run the item 9 VLMs
  on its egocentric clips and score them against real labels, instead of
  judging by eye. Its hand-pose-based action segmentation also maps onto the
  Quest hand joints we get for free. Data is on Zenodo and HF, code is on
  GitHub. The license isn't stated on the page, so check it before use.
- **Roboflow few-shot PoC ([LinkedIn post](https://www.linkedin.com/posts/patrickdeschere_automate2026-ugcPost-7475547155395452929-Ho1u/)): high, as a method.**
  - Zero-shot models missed a specific part. About 10 frames were hand-labeled
    with SAM-assisted "Smart Select", then a first RF-DETR model was trained.
    That model pre-labeled a few dozen more frames, which were corrected
    before retraining. The output fed a step checklist.
  - This is the fallback when SAM3 or YOLOE miss a food noun, which was a
    real problem here: "meatball" peaked at 0.17 on YOLOE. **We already
    generate the pre-labels.** SAM3 masks from our own clips can seed a small
    detector trained on our kitchen.
  - RF-DETR is a real-time detector from Roboflow, Apache 2.0, with a
    segmentation variant. A trained closed-set model would also run far
    faster than SAM3.
  - The checklist overlay itself is a product idea close to ours: "did the
    cook do each step?"
  - A caveat from the comments: in step checking, the false negative is the
    costly failure, because a missed step goes unnoticed.
- **[V-JEPA 2](https://ai.meta.com/research/vjepa/): medium, for later.**
  Meta's self-supervised video world model. It outputs features, not text, so
  it needs a trained probe or an LLM attached. Its paper reports
  state-of-the-art action *anticipation* on EPIC-KITCHENS-100: predicting the
  cook's next action, not just naming the current one.
  [Code](https://github.com/facebookresearch/vjepa2) ·
  [paper](https://arxiv.org/abs/2506.09985). More work than prompting a VLM.
  Worth it only if we want "you're about to ..." guidance and VLM latency is
  too high.
- **[Sapiens](https://github.com/facebookresearch/sapiens) /
  [Sapiens2](https://github.com/facebookresearch/sapiens2): low-medium.**
  Meta's human-centric models (0.3B-2B, 1024 px) for 2D pose including hand
  keypoints, body-part segmentation, depth and normals. Built for images of
  people seen from outside. Quest hand tracking already gives us better hand
  joints for free. Its body-part segmentation could clean up hand and arm
  false positives if projecting the joints (item 5) proves too coarse.
- **[Recognize Anything (RAM/RAM++)](https://recognize-anything.github.io/): low-medium.**
  An image tagger with 6,400+ tags, from 2023, designed to feed
  Grounded-SAM. Use case: a cheap "what's on the counter?" pass that suggests
  prompts for SAM3, so the cook doesn't have to name the food. A VLM (item 9)
  does the same job and also describes the action. Only worth it if VLM
  cost or latency becomes a problem.
- **[FastVLM](https://huggingface.co/apple/FastVLM-1.5B-int8): low.**
  Apple's compact VLM, with up to 7.9× faster time to first token than
  similar models. This checkpoint is MLX for iOS and macOS, and the Apple
  AMLR license restricts commercial use. It doesn't fit a Windows plus CUDA
  PC, and the Quest is Android. The idea is useful: a fast vision encoder
  that cuts image tokens. For local VLMs, Qwen3-VL and Moondream 3 (item 9)
  fit our stack.
- **[DeepSeek vision](https://api-docs.deepseek.com/guides/vision/): low-medium.**
  Covered in item 9. `deepseek-flash` handles static images only: no video,
  no grounding. It works as a frame captioner and nothing more.
- **[Cognition in the Wild](https://mitpress.mit.edu/9780262581462/cognition-in-the-wild/)
  (Hutchins, 1995): design framing, not a tool.** It introduced distributed
  cognition: thinking is spread across people, tools and the environment,
  not held only in one head. For us, the kitchen plus the headset form one
  cognitive system. Placing state *in the world* (portion markers on the
  food, step progress where the work happens, as in the Roboflow checklist)
  should beat telling the cook things they must remember. That supports
  world-anchored widgets (item 1) over a floating screen.
- **[arXiv 2601.12134](https://arxiv.org/html/2601.12134): not relevant.**
  This is "Human-Human-AI Triadic Programming", a study of pair programming
  with a shared AI. The only loose thread: a *shared* AI made pairs more
  accountable than personal AIs did. That might matter if two people cook
  together, but not now. Possibly the wrong link.

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
7. Prompt Gemini Robotics ER 2 (or the same VLM as step 6) on a sampled frame
   for the current action, alongside the portion question. Score it on
   EPFL-Smart-Kitchen's egocentric clips against their action labels.
8. If a food noun keeps failing zero-shot, use SAM3 to pre-label about 10
   frames from our clips, then train RF-DETR on them (the Roboflow loop).

If step 3 holds up, the tracking question under **Open** reduces to food in the
cook's hands. DARTF, the hybrids and the 8-10 fps target could then be shelved.
