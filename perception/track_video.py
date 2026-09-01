"""Track a concept through a video and write an annotated copy.

    python track_video.py "../Test Data/AriaEverydayActivities_1.mp4" "mug"
    python track_video.py video.mp4 "meatball" --max-frames 300 --stride 2

Writes <video>_<prompt>.mp4 next to the source and prints per-frame track IDs,
so you can see whether an object keeps its identity through occlusion.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from video_runner import Sam3VideoTracker, load_frames, open_writer
from viz import draw_instances


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM 3 video concept tracking")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("prompt", help='Concept to track, e.g. "mug"')
    parser.add_argument("--max-frames", type=int, default=150, help="Frames to sample (default 150)")
    parser.add_argument("--stride", type=int, default=3, help="Take every Nth frame (default 3)")
    args = parser.parse_args()

    video_path = Path(args.video)
    frames, out_fps = load_frames(video_path, args.max_frames, args.stride)
    print(f"Decoded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"output {out_fps:.1f} fps")

    tracker = Sam3VideoTracker()
    print(f"Model loaded in {tracker.load_seconds:.1f}s on {tracker.device}")
    print(f'Tracking "{args.prompt}"...\n')

    out_path = video_path.parent / f"{video_path.stem}_{args.prompt.replace(' ', '_')}.mp4"
    writer = None
    seen_ids: set[int] = set()
    per_frame_ms = []

    t_total = time.perf_counter()
    for result in tracker.track(frames, args.prompt):
        annotated = draw_instances(result.image, result.instances)
        if writer is None:
            writer = open_writer(out_path, out_fps)
        writer.append_data(np.asarray(annotated))

        ids = [i.obj_id for i in result.instances]
        seen_ids.update(ids)
        per_frame_ms.append(result.inference_ms)
        if result.frame_idx % 10 == 0:
            print(f"frame {result.frame_idx:4d}  {len(ids)} tracked  ids={ids}")

    if writer:
        writer.close()

    elapsed = time.perf_counter() - t_total
    print(f"\n{len(frames)} frames in {elapsed:.1f}s "
          f"({elapsed / len(frames) * 1000:.0f}ms/frame end to end)")
    print(f"Distinct track IDs over the clip: {sorted(seen_ids)}")
    print(f"Annotated video: {out_path}")


if __name__ == "__main__":
    main()
