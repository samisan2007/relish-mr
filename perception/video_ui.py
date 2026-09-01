"""Interactive SAM 3 video tester: upload a clip, type a prompt, watch the tracks.

    python video_ui.py    ->  opens http://127.0.0.1:7861

Separate port from ui.py so the image and video testers can run side by side.
Tracking is slow (~1.2s per frame), so keep the frame count modest.
"""

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from video_runner import Sam3VideoTracker, load_frames, open_writer
from viz import draw_instances

print("Loading SAM 3 video model...")
tracker = Sam3VideoTracker()
print(f"Ready on {tracker.device} ({tracker.load_seconds:.1f}s)")


def run(video_path, prompt, max_frames, stride, show_masks, show_boxes, progress=gr.Progress()):
    if not video_path or not prompt.strip():
        return None, "Provide a video and a prompt."

    frames, out_fps = load_frames(video_path, int(max_frames), int(stride))
    out_path = Path(tempfile.mkdtemp()) / "tracked.mp4"

    writer = None
    seen_ids: set[int] = set()
    per_frame_counts = []

    for result in progress.tqdm(
        tracker.track(frames, prompt.strip()), total=len(frames), desc="Tracking"
    ):
        annotated = draw_instances(
            result.image, result.instances, show_masks=show_masks, show_boxes=show_boxes
        )
        if writer is None:
            writer = open_writer(out_path, out_fps)
        writer.append_data(np.asarray(annotated))

        ids = [i.obj_id for i in result.instances]
        seen_ids.update(ids)
        per_frame_counts.append(len(ids))

    if writer:
        writer.close()

    if not seen_ids:
        return None, f'No "{prompt}" found in {len(frames)} frames.'

    summary = [
        f"{len(frames)} frames tracked at {out_fps:.1f} fps",
        f"Distinct track IDs: {sorted(seen_ids)}",
        f"Objects per frame: min {min(per_frame_counts)}, max {max(per_frame_counts)}",
        "",
        "An ID that disappears and returns with the same number means the",
        "tracker re-acquired the object rather than treating it as new.",
    ]
    return str(out_path), "\n".join(summary)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Video(label="Video"),
        gr.Textbox(label="Prompt", placeholder="mug"),
        gr.Slider(10, 300, value=60, step=10, label="Frames to sample"),
        gr.Slider(1, 10, value=3, step=1, label="Stride (take every Nth frame)"),
        gr.Checkbox(value=True, label="Segmentation masks"),
        gr.Checkbox(value=True, label="Bounding boxes"),
    ],
    outputs=[gr.Video(label="Tracked"), gr.Textbox(label="Results", lines=8)],
    title="SAM 3 video tracker",
    description="Tracks every instance of the concept through the clip, keeping stable IDs. "
                "Roughly 1.2s per frame, so 60 frames takes about a minute.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_port=7861, inbrowser=True)
