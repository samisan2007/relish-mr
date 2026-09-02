"""Interactive SAM 3 video tester: a clip from disk, or a live webcam feed.

    python video_ui.py    ->  opens http://127.0.0.1:7861

Separate port from ui.py so the image and video testers can run side by side.
Both tabs use the same Sam3VideoModel; the file tab runs the offline path
(better quality). The webcam tab runs streaming inference and lets you pick
a speed/quality config per run (see bench_realtime.py for what each one
changes) and logs the last several runs so they're easy to compare.
"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from bench_realtime import CONFIGS
from video_runner import Sam3VideoTracker, load_frames, open_writer
from viz import draw_instances
from yoloe_runner import HybridVideoTracker, YoloeVideoTracker

print("Loading SAM 3 video model...")
tracker = Sam3VideoTracker()
print(f"Ready on {tracker.device} ({tracker.load_seconds:.1f}s)")

MODEL_SAM3 = "SAM3"
MODEL_YOLOE = "YOLOE (text prompt)"
MODEL_HYBRID = "Hybrid (SAM3 seed -> YOLOE track)"
MODEL_CHOICES = [MODEL_SAM3, MODEL_YOLOE, MODEL_HYBRID]

# Webcam tab: one extra SAM3 config may be cached alongside `tracker` so switching between
# the baseline and one alternate is instant; switching to a different alternate reloads.
# YOLOE/Hybrid each load once and stay cached — they don't have SAM3's precision configs.
_bench_cache: dict[str, Sam3VideoTracker] = {}
_yoloe_tracker: YoloeVideoTracker | None = None
_hybrid_tracker: HybridVideoTracker | None = None


def get_bench_tracker(config_name: str) -> Sam3VideoTracker:
    cfg = next((c for c in CONFIGS if c.name == config_name), CONFIGS[0])
    if cfg.name == CONFIGS[0].name:
        return tracker  # already loaded — avoid a redundant ~2GB VRAM copy

    if config_name not in _bench_cache:
        _bench_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Loading webcam benchmark config: {cfg.name} ...")
        _bench_cache[config_name] = Sam3VideoTracker(
            dtype=cfg.dtype,
            processor_size=cfg.processor_size,
            use_device_map=cfg.use_device_map,
            compile_model=cfg.compile_model,
        )
    return _bench_cache[config_name]


def get_active_tracker(model_choice: str, config_name: str):
    """Returns (tracker, label) for whichever backend is selected. label goes in the
    status line and run log so entries stay distinguishable across all three backends."""
    global _yoloe_tracker, _hybrid_tracker

    if model_choice == MODEL_YOLOE:
        if _yoloe_tracker is None:
            print("Loading YOLOE...")
            _yoloe_tracker = YoloeVideoTracker()
        return _yoloe_tracker, MODEL_YOLOE

    if model_choice == MODEL_HYBRID:
        if _hybrid_tracker is None:
            print("Loading Hybrid (YOLOE side; SAM3 side reuses the baseline model)...")
            _hybrid_tracker = HybridVideoTracker(sam3_tracker=tracker)
        return _hybrid_tracker, MODEL_HYBRID

    cam_tracker = get_bench_tracker(config_name)
    return cam_tracker, f"SAM3: {config_name}"


def run_file(video_path, prompt, max_frames, stride, show_masks, show_boxes,
             progress=gr.Progress()):
    if not video_path or not prompt.strip():
        return None, "Provide a video and a prompt."

    frames, out_fps = load_frames(video_path, int(max_frames), int(stride))
    out_path = Path(tempfile.mkdtemp()) / "tracked.mp4"

    writer = None
    seen_ids: set[int] = set()
    counts = []

    for result in progress.tqdm(
        tracker.track(frames, prompt.strip()), total=len(frames), desc="Tracking"
    ):
        annotated = draw_instances(
            result.image, result.instances, show_masks=show_masks, show_boxes=show_boxes
        )
        if writer is None:
            writer = open_writer(out_path, out_fps)
        writer.append_data(np.asarray(annotated))

        seen_ids.update(i.obj_id for i in result.instances)
        counts.append(len(result.instances))

    if writer:
        writer.close()

    if not seen_ids:
        return None, f'No "{prompt}" found in {len(frames)} frames.'

    return str(out_path), "\n".join([
        f"{len(frames)} frames tracked at {out_fps:.1f} fps",
        f"Distinct track IDs: {sorted(seen_ids)}",
        f"Objects per frame: min {min(counts)}, max {max(counts)}",
        "",
        "An ID that disappears and returns with the same number means the",
        "tracker re-acquired the object rather than treating it as new.",
    ])


def webcam_loop(prompt, show_masks, show_boxes, camera_index, model_choice, config_name, run_state):
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

    capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not capture.isOpened():
        yield None, (f"Could not open camera {int(camera_index)}. Close any other app or "
                     "browser tab using it, or try a different index."), run_state
        return

    session = cam_tracker.start_stream(prompt.strip())
    run_state = {"label": label, "prompt": prompt.strip(), "times": [], "counts": [], "ids": set()}

    try:
        while True:
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

            run_state["ids"].update(i.obj_id for i in instances)
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
        capture.release()


def format_log(history: list[str]) -> str:
    if not history:
        return "(no runs yet — Start, let it track, then Stop to log a result)"
    return "\n".join(f"{i}. {line}" for i, line in enumerate(reversed(history), 1))


def finalize_run(run_state, history):
    """Stop-button handler: summarize the just-ended run from `run_state` into the log.

    Reads accumulated stats out of state rather than the generator's own cleanup,
    since Gradio's cancellation doesn't let the generator return a final value.
    """
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


with gr.Blocks(title="SAM 3 video tracker") as demo:
    gr.Markdown("# SAM 3 video tracker\n"
                "Tracks every instance of a concept, keeping stable IDs across frames.")

    with gr.Tabs():
        with gr.Tab("Video file"):
            with gr.Row():
                with gr.Column():
                    f_video = gr.Video(label="Video")
                    f_prompt = gr.Textbox(label="Prompt", placeholder="mug")
                    f_frames = gr.Slider(10, 300, value=60, step=10, label="Frames to sample")
                    f_stride = gr.Slider(1, 10, value=3, step=1, label="Stride (every Nth frame)")
                    f_masks = gr.Checkbox(value=True, label="Segmentation masks")
                    f_boxes = gr.Checkbox(value=True, label="Bounding boxes")
                    f_run = gr.Button("Track", variant="primary")
                with gr.Column():
                    f_out = gr.Video(label="Tracked")
                    f_info = gr.Textbox(label="Results", lines=8)
            f_run.click(run_file,
                        [f_video, f_prompt, f_frames, f_stride, f_masks, f_boxes],
                        [f_out, f_info])

        with gr.Tab("Webcam (live)"):
            gr.Markdown(
                "Live feed with tracking drawn on it. Three backends to compare:\n"
                "- **SAM3** — reliable text-prompt grounding (handles niche nouns like "
                "\"meatball\" well) but slow; the config dropdown picks precision.\n"
                "- **YOLOE (text prompt)** — fast (~100-500ms/frame) but its MobileCLIP "
                "vocabulary is noticeably weaker on specific food nouns; low confidence "
                "even when it does find something.\n"
                "- **Hybrid** — runs SAM3 every frame until it grounds the prompt once, "
                "then switches to YOLOE's visual-exemplar tracking (fast, 0.9+ confidence) "
                "for every frame after. Expect a slow start, then a speed-up.\n\n"
                "Streaming disables the heuristics that prune duplicate tracks, so expect "
                "more false positives than the file tab. The camera must be free — close "
                "any browser tab or app already using it. Changing model/config only takes "
                "effect on the next Start (Stop first); the first use of a new one pays a "
                "load cost (~5-10s SAM3 config, longer for YOLOE's first-ever download)."
            )
            w_view = gr.Image(label="Live", type="numpy", height=520)
            w_info = gr.Textbox(label="Status", lines=2)
            with gr.Row():
                w_prompt = gr.Textbox(label="Prompt", placeholder="pen", scale=3)
                w_camera = gr.Number(value=0, label="Camera index", precision=0, scale=1)
            with gr.Row():
                w_model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_SAM3, label="Model")
                w_config = gr.Dropdown(
                    choices=[c.name for c in CONFIGS], value=CONFIGS[0].name,
                    label="SAM3 config (ignored unless Model = SAM3)",
                )
            with gr.Row():
                w_masks = gr.Checkbox(value=True, label="Segmentation masks")
                w_boxes = gr.Checkbox(value=True, label="Bounding boxes")
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
                [w_prompt, w_masks, w_boxes, w_camera, w_model, w_config, w_run_state],
                [w_view, w_info, w_run_state],
                show_progress="hidden",
            )
            w_stop.click(
                finalize_run, [w_run_state, w_history],
                [w_info, w_history, w_log, w_run_state],
                cancels=[stream_event],
            )

if __name__ == "__main__":
    demo.launch(server_port=7861, inbrowser=True)
