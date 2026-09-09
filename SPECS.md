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
- **Perception runs on the PC** in Python (`perception/`), against SAM3 and YOLOE.
- **The widget is decoupled from perception.** `PortionWidget` takes a world
  position and a diameter in metres and knows nothing about how they were derived.

### Open

- **Tracking approach.** SAM3 is the only reliable backend and runs at ~1 fps;
  every faster option tested loses the object under motion. Four options are
  written up in [temp-devlog.md](temp-devlog.md); none is chosen.
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
| `Scripts/PortionWidgets.cs` | The manager. `Report(frame)` of `Detection { Id, Position, Diameter, Score }` → spawns, lerps between ~1 fps reports, dims stale, retires lost. `Detection` is the contract with perception. |
| `Editor/RelishWidgetBuilder.cs` | `Relish > Create Widget Prefab` — generates the prefab and its transparent URP material. |
| `Editor/ContextMenuButtonsEditor.cs` | Turns every `[ContextMenu]` test method on the two scripts into an inspector button (Play mode only). |

The widget is a sphere *for now*. The API is the contract; the mesh is not.

## Perception side

`perception/` — SAM3 image + video testers, YOLOE trackers, a replay benchmark,
and a webcam UI that switches backends. See [temp-devlog.md](temp-devlog.md) for
the measurements and the reasoning behind them.
