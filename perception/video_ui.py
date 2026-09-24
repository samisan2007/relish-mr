"""Interactive video tester: a clip from disk, or a live webcam feed.

    python video_ui.py    ->  opens http://127.0.0.1:7861

Separate port from ui.py. Models load on selection; run one GPU test at a time.
SAM3 files use offline propagation. SAM 3.1 files and webcam use the same causal
Docker worker. Webcam runs are logged for comparison.
"""

import gc
import tempfile
import time
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from bench_realtime import CONFIGS
from sam3_runner import Sam3ImageTracker, Sam3Runner
from sam31_runner import MODEL_SAM31_COMPILED, SAM31_CHOICES, Sam31Runner
from video_runner import Sam3VideoTracker, load_frames, open_writer
from viz import draw_instances
from yoloe_runner import HybridVideoTracker, YoloeVideoTracker

# File mode and the default webcam config share one lazily loaded SAM3 instance.
DEFAULT_CONFIG = next((c for c in CONFIGS if c.name == "fp16 autocast, 1008px"), None)
assert DEFAULT_CONFIG is not None, "renamed a config in bench_realtime.CONFIGS?"


def _build_tracker(cfg) -> Sam3VideoTracker:
    return Sam3VideoTracker(
        dtype=cfg.dtype,
        processor_size=cfg.processor_size,
        use_device_map=cfg.use_device_map,
        compile_model=cfg.compile_model,
        cudnn_benchmark=cfg.cudnn_benchmark,
        max_cond_frame_num=cfg.max_cond_frame_num,
    )


tracker = None  # Load on selection so Docker backends have the GPU available.

MODEL_SAM3 = "SAM3"
MODEL_SAM3_IMG = "SAM3 image + ByteTrack"
MODEL_KEYFRAME = "Keyframe hybrid (SAM3 image -> EdgeTAM)"
MODEL_KEYFRAME_31 = "Keyframe hybrid (SAM 3.1 image -> EdgeTAM)"
MODEL_KEYFRAME_31_COMPILED = "Keyframe hybrid (SAM 3.1 compiled -> EdgeTAM)"
KEYFRAME_31_CHOICES = [MODEL_KEYFRAME_31, MODEL_KEYFRAME_31_COMPILED]
KEYFRAME_CHOICES = [MODEL_KEYFRAME, *KEYFRAME_31_CHOICES]
MODEL_YOLOE = "YOLOE (text prompt)"
MODEL_HYBRID = "Hybrid (SAM3 seed -> YOLOE track)"
MODEL_HYBRID_31 = "Hybrid (SAM 3.1 seed -> YOLOE track)"
MODEL_DARTF = "DARTF (native SAM3, FP16 TensorRT)"
MODEL_DARTF_FAST = "DARTF FAST (W8A8 TensorRT)"
MODEL_CHOICES = [MODEL_SAM3, *SAM31_CHOICES, MODEL_SAM3_IMG, *KEYFRAME_CHOICES, MODEL_YOLOE, MODEL_HYBRID,
                 MODEL_HYBRID_31, MODEL_DARTF, MODEL_DARTF_FAST]
# Backends whose model runs in a Docker worker: their per-frame timing includes transfer,
# so the run summary reports request speed separately from total processing.
WORKER_CHOICES = [*SAM31_CHOICES, MODEL_DARTF, MODEL_DARTF_FAST]

# Webcam tab: one extra SAM3 config may be cached alongside `tracker` so switching between
# the baseline and one alternate is instant; switching to a different alternate reloads.
# YOLOE/Hybrid each load once and stay cached — they don't have SAM3's precision configs.
_bench_cache: dict[str, Sam3VideoTracker] = {}
_yoloe_tracker: YoloeVideoTracker | None = None
_hybrid_tracker: HybridVideoTracker | None = None
_hybrid31_tracker: HybridVideoTracker | None = None
# The image model is a separate ~3GB set of weights from the video model above, so it
# loads on first use rather than at import — pick another backend and you never pay it.
_sam3_image_tracker: Sam3ImageTracker | None = None
# Its own SAM3 image weights (fp16, ~1.6GB) plus EdgeTAM; released like the one above.
_keyframe_tracker = None
# EdgeTAM only; its SAM 3.1 keyframes come from a Docker worker opened per stream.
_keyframe31_tracker = None
_webcam_runs = {}  # Live worker handles stay server-side, outside serializable gr.State.


def get_bench_tracker(config_name: str) -> Sam3VideoTracker:
    global tracker
    cfg = next((c for c in CONFIGS if c.name == config_name), DEFAULT_CONFIG)
    # cuDNN's flag is global, so reapply it even when selecting a cached tracker.
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.name == DEFAULT_CONFIG.name:
        if tracker is None:
            tracker = _build_tracker(cfg)
        return tracker

    if config_name not in _bench_cache:
        _bench_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Loading webcam benchmark config: {cfg.name} ...")
        _bench_cache[config_name] = _build_tracker(cfg)
    return _bench_cache[config_name]


def get_active_tracker(model_choice: str, config_name: str):
    """Returns (tracker, label) for whichever backend is selected. label goes in the
    status line and run log so entries stay distinguishable across backends."""
    global tracker, _yoloe_tracker, _hybrid_tracker, _hybrid31_tracker, _sam3_image_tracker
    global _keyframe_tracker, _keyframe31_tracker

    if model_choice in SAM31_CHOICES or model_choice in (
            MODEL_DARTF, MODEL_DARTF_FAST, MODEL_HYBRID_31, *KEYFRAME_31_CHOICES):
        # These run their model inside a GPU Docker worker, so every Windows-side copy has
        # to leave the card first — otherwise the two compete for the same 12GB and the
        # driver spills to system RAM rather than failing.
        tracker = _yoloe_tracker = _hybrid_tracker = _sam3_image_tracker = _keyframe_tracker = None
        _bench_cache.clear()
        # The 3.1 hybrids are the exception: their YOLOE/EdgeTAM half is small and is
        # meant to sit alongside the worker, so keep it cached rather than reloading.
        if model_choice != MODEL_HYBRID_31:
            _hybrid31_tracker = None
        if model_choice not in KEYFRAME_31_CHOICES:
            _keyframe31_tracker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model_choice in KEYFRAME_31_CHOICES:
            from keyframe_hybrid import KeyframeHybridTracker

            if _keyframe31_tracker is None:
                print("Loading keyframe hybrid (EdgeTAM side; SAM 3.1 detects from its Docker worker)...")
                _keyframe31_tracker = KeyframeHybridTracker(Sam31Runner())
            _keyframe31_tracker.sam3 = Sam31Runner(compile_model=model_choice == MODEL_KEYFRAME_31_COMPILED)
            return _keyframe31_tracker, model_choice
        if model_choice == MODEL_HYBRID_31:
            if _hybrid31_tracker is None:
                print("Loading Hybrid (YOLOE side; SAM 3.1 grounds from its Docker worker)...")
                _hybrid31_tracker = HybridVideoTracker(seed_tracker=Sam31Runner())
            return _hybrid31_tracker, MODEL_HYBRID_31
        if model_choice == MODEL_DARTF:
            from dartf_runner import DartfVideoTracker

            return DartfVideoTracker(), MODEL_DARTF
        if model_choice == MODEL_DARTF_FAST:
            from dartf_runner import FastVideoTracker

            return FastVideoTracker(), MODEL_DARTF_FAST
        return Sam31Runner(compile_model=model_choice == MODEL_SAM31_COMPILED), model_choice

    if model_choice == MODEL_SAM3_IMG:
        if _sam3_image_tracker is None:
            print("Loading SAM 3 image model (separate weights from the video model)...")
            _sam3_image_tracker = Sam3ImageTracker(Sam3Runner(dtype=torch.float16))
        return _sam3_image_tracker, MODEL_SAM3_IMG

    if model_choice == MODEL_KEYFRAME:
        if _keyframe_tracker is None:
            from keyframe_hybrid import KeyframeHybridTracker

            print("Loading keyframe hybrid (SAM3 image fp16 + EdgeTAM)...")
            _keyframe_tracker = KeyframeHybridTracker(Sam3Runner(dtype=torch.float16))
        return _keyframe_tracker, MODEL_KEYFRAME

    # A second full copy of SAM3 lives here, so release it when another backend is
    # picked. Three resident copies overflow a 12GB card, and the driver then spills to
    # system RAM instead of failing — which reads as the whole app freezing. Same trade
    # as the config cache above: switching back reloads.
    if _sam3_image_tracker is not None or _keyframe_tracker is not None or _keyframe31_tracker is not None:
        _sam3_image_tracker = _keyframe_tracker = _keyframe31_tracker = None
        # gc.collect() before empty_cache(): the model's modules and accelerate hooks
        # form reference cycles, so dropping the name alone leaves the weights alive and
        # empty_cache() finds nothing to return. Without this the release is a no-op.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model_choice == MODEL_YOLOE:
        if _yoloe_tracker is None:
            print("Loading YOLOE...")
            _yoloe_tracker = YoloeVideoTracker()
        return _yoloe_tracker, MODEL_YOLOE

    if model_choice == MODEL_HYBRID:
        if _hybrid_tracker is None:
            print("Loading Hybrid (YOLOE side; SAM3 side reuses the baseline model)...")
            _hybrid_tracker = HybridVideoTracker(seed_tracker=get_bench_tracker(DEFAULT_CONFIG.name))
        return _hybrid_tracker, MODEL_HYBRID

    cam_tracker = get_bench_tracker(config_name)
    return cam_tracker, f"SAM3: {config_name}"


def run_file(video_path, prompt, max_frames, stride, show_masks, show_boxes,
             model_choice=MODEL_SAM3, progress=gr.Progress()):
    if not video_path or not prompt.strip():
        return None, "Provide a video and a prompt."

    started = time.perf_counter()
    try:
        active, label = get_active_tracker(model_choice, DEFAULT_CONFIG.name)
        frames, out_fps = load_frames(video_path, int(max_frames), int(stride))
    except Exception as error:
        return None, f"Failed to load '{model_choice}': {error}"
    out_path = Path(tempfile.mkdtemp()) / "tracked.mp4"

    writer = None
    seen_ids: set[int] = set()
    counts = []
    times = []
    results = active.track(frames, prompt.strip())
    try:
        for result in progress.tqdm(results, total=len(frames), desc="Tracking"):
            annotated = draw_instances(
                result.image, result.instances, show_masks=show_masks, show_boxes=show_boxes
            )
            if writer is None:
                writer = open_writer(out_path, out_fps)
            writer.append_data(np.asarray(annotated))
            seen_ids.update(i.obj_id for i in result.instances if i.obj_id is not None)
            counts.append(len(result.instances))
            times.append(result.inference_ms)
    except Exception as error:
        return None, f"Tracking failed on '{label}': {error}"
    finally:
        results.close()
        if writer is not None:
            writer.close()
    elapsed = time.perf_counter() - started

    if not seen_ids:
        return str(out_path), f'[{label}] No "{prompt}" found in {len(frames)} frames.'

    return str(out_path), "\n".join([
        f"[{label}] {len(frames)} frames; playback {out_fps:.1f} fps",
        f"Total processing: {elapsed:.1f}s ({len(frames) / elapsed:.2f} fps, including startup and encoding)",
        *([f"Frame requests after first 8: {1000 / np.mean(times[8:]):.2f} fps (includes Docker transfer and any later compilation)"]
          if model_choice in WORKER_CHOICES and len(times) > 8 else []),
        f"Distinct track IDs: {sorted(seen_ids)}",
        f"Objects per frame: min {min(counts)}, max {max(counts)}",
        "",
        "Inspect masks and IDs through motion and occlusion; ID counts alone do not verify identity.",
    ])


def webcam_loop(prompt, show_masks, show_boxes, camera_index, model_choice, config_name, run_state,
                reground=0, keyframe_every=10, request: gr.Request = None):
    """Grab frames from the local camera and yield annotated ones, until cancelled.

    Capture happens server-side (OpenCV) rather than in the browser, so the feed
    and its overlays are a single image. The camera must not be in use elsewhere
    — a browser tab holding it will block this.

    `run_state` accumulates this run's per-frame timings so the Stop button can
    summarize it into the log — it's read from gr.State, not the generator's
    return value, since Gradio cancels this generator rather than letting it
    finish normally.
    """
    if not prompt.strip():
        yield None, "Enter a prompt first.", run_state
        return

    try:
        cam_tracker, label = get_active_tracker(model_choice, config_name)
    except Exception as e:
        yield None, f"Failed to load '{model_choice}': {e}", run_state
        return

    if isinstance(cam_tracker, HybridVideoTracker):
        # Set it on the cached tracker rather than through the constructor, so moving the
        # slider takes effect on the next Start without reloading YOLOE. The interval goes
        # in the label because it changes both fps and the ID count the log reports.
        cam_tracker.reground_every = int(reground)
        if reground:
            label = f"{label} reground {int(reground)}"
    if model_choice in KEYFRAME_CHOICES:
        cam_tracker.keyframe_every = int(keyframe_every)
        label = f"{label} every {int(keyframe_every)}"

    capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Inference (hundreds of ms-1s/frame) is far slower than the camera's native rate, so
    # without this the driver queues frames faster than we consume them and read() drains
    # a growing backlog instead of returning the current one — the feed falls further and
    # further behind real time. Asking for a 1-frame buffer keeps read() on the latest
    # frame. Not guaranteed to hold with DSHOW on Windows — if the lag comes back, the next
    # step is a background capture thread instead of relying on this.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        yield None, (f"Could not open camera {int(camera_index)}. Close any other app or "
                     "browser tab using it, or try a different index."), run_state
        return

    session = None
    key = request.session_hash if request is not None else "local"
    control = {"stop": threading.Event(), "session": None}
    _webcam_runs[key] = control

    def on_started(started_session):
        control["session"] = started_session
        if control["stop"].is_set():
            started_session.failed = True
            started_session.reset_inference_session()
            raise RuntimeError("Stopped during startup")

    try:
        yield None, f"Starting {label}; compiled first frames can take several minutes. Stop cancels the run.", {}
        try:
            if isinstance(cam_tracker, Sam31Runner) or model_choice in (
                    MODEL_DARTF, MODEL_DARTF_FAST, MODEL_HYBRID_31, *KEYFRAME_31_CHOICES):
                session = cam_tracker.start_stream(prompt.strip(), on_started=on_started)
            else:
                session = cam_tracker.start_stream(prompt.strip())
                control["session"] = session
        except Exception as e:
            yield None, f"Failed to start '{label}': {e}", run_state
            return
        run_state = {"label": label, "prompt": prompt.strip(), "times": [], "counts": [], "ids": set()}
        while not control["stop"].is_set():
            ok, frame_bgr = capture.read()
            if not ok:
                yield None, "Lost the camera feed.", run_state
                return

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                instances, ms = cam_tracker.track_frame(session, frame)
            except Exception as e:
                yield None, f"Tracking failed on '{label}': {e}", run_state
                return

            if isinstance(cam_tracker, HybridVideoTracker) and ms == 0:
                # Cooldown-only frames did no inference; exclude them from timing.
                yield frame, f'[{label}] Waiting to retry grounding for "{prompt.strip()}".', run_state
                continue

            # obj_id can be None for a detection the tracker hasn't confirmed yet — ByteTrack
            # on the SAM3-image backend, and whatever ultralytics defaults to for YOLOE/Hybrid
            # (TRACKTRACK as of 8.4.138, not ByteTrack). Shows up on live motion, not on a
            # static frame.
            run_state["ids"].update(i.obj_id for i in instances if i.obj_id is not None)
            run_state["times"].append(ms)
            run_state["counts"].append(len(instances))
            recent = run_state["times"][-10:]
            avg = sum(recent) / len(recent)

            annotated = draw_instances(
                Image.fromarray(frame), instances,
                show_masks=show_masks, show_boxes=show_boxes,
            )
            yield np.asarray(annotated), (
                f'[{label}] "{prompt.strip()}" — {len(instances)} tracked now | '
                f"ids so far: {sorted(run_state['ids'])}\n"
                f"{ms:.0f}ms this frame, {avg:.0f}ms avg ({1000 / avg:.1f} fps) | "
                f"{len(run_state['times'])} frames"
            ), run_state
    finally:
        if _webcam_runs.get(key) is control:
            del _webcam_runs[key]
        capture.release()
        # The SAM3 session holds its memory bank and vision-feature cache on the GPU.
        # Stop cancels this generator rather than letting it return, so without an
        # explicit reset every start/stop cycle strands a run's worth of VRAM and the
        # app degrades after a few rounds. The dict-based backends have nothing to free.
        if hasattr(session, "reset_inference_session"):
            session.reset_inference_session()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def format_log(history: list[str]) -> str:
    if not history:
        return "(no runs yet — Start, let it track, then Stop to log a result)"
    return "\n".join(f"{i}. {line}" for i, line in enumerate(reversed(history), 1))


def finalize_run(run_state, history, request: gr.Request = None):
    """Stop-button handler: summarize the just-ended run from `run_state` into the log.

    Reads accumulated stats out of state rather than the generator's own cleanup,
    since Gradio's cancellation doesn't let the generator return a final value.
    """
    key = request.session_hash if request is not None else "local"
    control = _webcam_runs.get(key)
    if control is not None:
        control["stop"].set()
        session = control["session"]
        if hasattr(session, "process"):
            session.failed = True  # Interrupt a pending load/compile/frame request.
            session.reset_inference_session()
    times = run_state.get("times") if run_state else None
    if not times:
        return "Stopped.", history, format_log(history), {}

    avg = sum(times) / len(times)
    hits = sum(1 for c in run_state["counts"] if c > 0)
    summary = (
        f"{run_state['label']} | \"{run_state['prompt']}\" — "
        f"{1000 / avg:.1f} fps ({avg:.0f}ms avg) | "
        f"hits {hits}/{len(times)} | ids {len(run_state['ids'])}"
    )
    history = (history + [summary])[-8:]
    return "Stopped.", history, format_log(history), {}


with gr.Blocks(title="Relish video tracker") as demo:
    gr.Markdown("### Relish video tracker\n"
                "Name a thing. Every instance gets a mask and an ID that survives across frames.")

    with gr.Tabs():
        with gr.Tab("Video file"):
            with gr.Row():
                with gr.Column():
                    f_video = gr.Video(label="Video")
                    f_prompt = gr.Textbox(label="Prompt", placeholder="mug")
                    f_model = gr.Dropdown(choices=[MODEL_SAM3, *SAM31_CHOICES, MODEL_DARTF], value=MODEL_SAM3, label="Model")
                    gr.Markdown("Stride 1 for a benchmark comparable to a live run. "
                                "Compiled startup takes minutes.")
                    f_frames = gr.Slider(10, 1000, value=60, step=10, label="Frames to sample")
                    f_stride = gr.Slider(1, 10, value=3, step=1, label="Stride (every Nth frame)")
                    f_masks = gr.Checkbox(value=True, label="Segmentation masks")
                    f_boxes = gr.Checkbox(value=True, label="Bounding boxes")
                    f_run = gr.Button("Track", variant="primary")
                with gr.Column():
                    f_out = gr.Video(label="Tracked")
                    f_info = gr.Textbox(label="Results", lines=8)
            f_run.click(run_file,
                        [f_video, f_prompt, f_frames, f_stride, f_masks, f_boxes, f_model],
                        [f_out, f_info], concurrency_id="perception-gpu", concurrency_limit=1)

        with gr.Tab("Webcam (live)"):
            gr.Markdown("Live tracking. Free the camera first; Stop before changing model or config.")
            with gr.Accordion("Backends", open=False):
                gr.Markdown(
                    """
| Backend | Notes |
|---|---|
| SAM 3.1 | Object Multiplex, up to 16 regions. Compiled mode is slow to start. |
| SAM3 | Text grounding with object memory; survives rotation. Warms up over ~8 frames. |
| SAM3 image + ByteTrack | No memory — re-detects per frame, IDs by overlap. Drops IDs on fast motion. |
| Keyframe hybrid | SAM3 image mode every N frames; EdgeTAM (a light SAM 2) carries the masks between. Keyframes match to live tracks by mask overlap, so IDs persist. Each keyframe frame is slow (~250 ms), the rest fast. |
| Keyframe hybrid (SAM 3.1) | Same, with SAM 3.1 image mode for the keyframes, from its Docker worker. Slow to start; compiled adds a ~20 s pause on the first keyframe (longer on a cold cache). |
| YOLOE | Fast text detection; weak on specific food nouns. |
| Hybrid | SAM3 grounds, then YOLOE tracks; retries immediately after loss, then waits 500 ms between attempts. |
| | YOLOE matches the seeded exemplars, not the words, so boxes far larger than them are rejected as drift. |
| Hybrid (SAM 3.1) | Same, with SAM 3.1 grounding from its Docker worker. Slow to start, then YOLOE speed. |
| DARTF | FP16 TensorRT + native SAM3 memory, via Docker. Needs locally built engines. |
| DARTF FAST | W8A8 TensorRT + lightweight tracker, via Docker. Needs the RTX 3080 FAST build. |

Streaming keeps duplicate tracks the file tab would prune, so expect more false
positives here. Measurements live in `DEVLOG.md`.

**Re-ground interval.** A hybrid re-runs its grounding model on that frame. On an RTX
3080 at 640x480 a quiet YOLOE frame is ~35ms, while a SAM 3.1 re-ground frame is ~750ms
when it finds something (~400ms when it does not, since nothing is reinstalled), so the
interval sets the average rate:

| Every N frames | Average | What it feels like |
|---|---|---|
| 0 (lost only) | ~28 fps | No hitch until the track drops |
| 120 | ~24 fps | A pause every ~4s |
| 60 | ~21 fps | A pause every ~3s |
| 30 | ~17 fps | A pause every ~1s |
| 10 | ~9 fps | Pauses dominate |

Measured directly at N=10 (9.0 fps) and N=0; the rest follow from the two frame costs.
These tracking-phase measurements predate the lost-state cooldown. While lost,
retries wait 500 ms after the previous attempt finishes; YOLOE keeps searching
with its old exemplars. Before the first seed there is no YOLOE search, and
cooldown-only frames are excluded from the inference run log.
Successful re-grounds **restart the track IDs** — installing new exemplars rebuilds
YOLOE's tracker — so a short interval inflates the ID count in the run log: the same
40-frame clip logged 3 IDs at N=10 against 1 at N=30.
"""
                )
            w_view = gr.Image(label="Live", type="numpy", height=520)
            w_info = gr.Textbox(label="Status", lines=2)
            with gr.Row():
                w_prompt = gr.Textbox(label="Prompt", placeholder="pen", scale=3)
                w_camera = gr.Number(value=0, label="Camera index", precision=0, scale=1)
            with gr.Row():
                w_model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_SAM3, label="Model")
                w_config = gr.Dropdown(
                    choices=[c.name for c in CONFIGS], value=DEFAULT_CONFIG.name,
                    label="SAM3 config (ignored unless Model = SAM3)",
                )
            with gr.Row():
                w_masks = gr.Checkbox(value=True, label="Segmentation masks")
                w_boxes = gr.Checkbox(value=True, label="Bounding boxes")
            w_reground = gr.Slider(
                0, 300, value=0, step=10,
                label="Hybrid re-ground interval, frames (0 = only when the track is lost; "
                      "ignored unless Model is a Hybrid)",
            )
            w_keyframe = gr.Slider(
                1, 60, value=10, step=1,
                label="Keyframe hybrids: run SAM3 / SAM 3.1 every N frames (ignored for other models)",
            )
            with gr.Row():
                w_start = gr.Button("Start", variant="primary")
                w_stop = gr.Button("Stop")
            w_log = gr.Textbox(
                label="Run log (last 8 — Stop a run to record it)", lines=10, interactive=False
            )

            w_run_state = gr.State({})
            w_history = gr.State([])

            stream_event = w_start.click(
                webcam_loop,
                [w_prompt, w_masks, w_boxes, w_camera, w_model, w_config, w_run_state, w_reground,
                 w_keyframe],
                [w_view, w_info, w_run_state],
                show_progress="hidden",
                concurrency_id="perception-gpu", concurrency_limit=1,
            )
            w_stop.click(
                finalize_run, [w_run_state, w_history],
                [w_info, w_history, w_log, w_run_state],
                cancels=[stream_event],
            )

if __name__ == "__main__":
    demo.launch(server_port=7861, inbrowser=True, theme=gr.themes.Soft())
