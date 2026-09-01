"""Interactive SAM 3 video tester: a clip from disk, or a live webcam feed.

    python video_ui.py    ->  opens http://127.0.0.1:7861

Separate port from ui.py so the image and video testers can run side by side.
Both tabs use the same Sam3VideoModel; the file tab runs the offline path
(better quality) and the webcam tab runs streaming inference (~1 fps).
"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from video_runner import Sam3VideoTracker, load_frames, open_writer
from viz import draw_instances

print("Loading SAM 3 video model...")
tracker = Sam3VideoTracker()
print(f"Ready on {tracker.device} ({tracker.load_seconds:.1f}s)")


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


def webcam_loop(prompt, show_masks, show_boxes, camera_index):
    """Grab frames from the local camera and yield annotated ones, until cancelled.

    Capture happens server-side (OpenCV) rather than in the browser, so the feed
    and its overlays are a single image. The camera must not be in use elsewhere
    — a browser tab holding it will block this.
    """
    if not prompt.strip():
        yield None, "Enter a prompt first."
        return

    capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not capture.isOpened():
        yield None, (f"Could not open camera {int(camera_index)}. Close any other app or "
                     "browser tab using it, or try a different index.")
        return

    session = tracker.start_stream(prompt.strip())
    seen: set[int] = set()
    times: list[float] = []

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                yield None, "Lost the camera feed."
                return

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            instances, ms = tracker.track_frame(session, frame)
            seen.update(i.obj_id for i in instances)
            times.append(ms)
            recent = times[-10:]
            avg = sum(recent) / len(recent)

            annotated = draw_instances(
                Image.fromarray(frame), instances,
                show_masks=show_masks, show_boxes=show_boxes,
            )
            yield np.asarray(annotated), (
                f'"{prompt.strip()}" — {len(instances)} tracked now | '
                f"ids so far: {sorted(seen)}\n"
                f"{ms:.0f}ms this frame, {avg:.0f}ms avg ({1000 / avg:.1f} fps) | "
                f"{len(times)} frames"
            )
    finally:
        capture.release()


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
                "Live feed with tracking drawn on it. Runs at roughly **1.4 fps** on an "
                "RTX 3080, so expect a slideshow rather than smooth video. Streaming also "
                "disables the heuristics that prune duplicate tracks, so expect more false "
                "positives than the file tab. The camera must be free — close any browser "
                "tab or app already using it."
            )
            w_view = gr.Image(label="Live", type="numpy", height=520)
            w_info = gr.Textbox(label="Status", lines=2)
            with gr.Row():
                w_prompt = gr.Textbox(label="Prompt", placeholder="pen", scale=3)
                w_camera = gr.Number(value=0, label="Camera index", precision=0, scale=1)
            with gr.Row():
                w_masks = gr.Checkbox(value=True, label="Segmentation masks")
                w_boxes = gr.Checkbox(value=True, label="Bounding boxes")
            with gr.Row():
                w_start = gr.Button("Start", variant="primary")
                w_stop = gr.Button("Stop")

            stream_event = w_start.click(
                webcam_loop, [w_prompt, w_masks, w_boxes, w_camera], [w_view, w_info],
                show_progress="hidden",
            )
            w_stop.click(lambda: "Stopped.", None, [w_info], cancels=[stream_event])

if __name__ == "__main__":
    demo.launch(server_port=7861, inbrowser=True)
