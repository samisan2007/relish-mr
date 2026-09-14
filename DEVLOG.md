# Relish — devlog

Running log, newest first. One entry per session: what changed, what it means,
what's next. For *what we're building*, see [SPECS.md](SPECS.md).

Detailed perception measurements and tracking experiments live in
[temp-devlog.md](temp-devlog.md).

---

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
