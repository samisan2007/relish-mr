"""Exercise real DARTF engines, Windows/Docker transfer and restart without a camera.

From perception/: .venv/Scripts/python.exe tests/smoke_dartf.py photo.jpg meatball
Requires the DARTF container and locally built engines; see dartf/README.md.
Synthetic translation checks wiring and continuity, not live occlusion quality.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dartf_runner import DartfVideoTracker
from viz import draw_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("prompt")
    args = parser.parse_args()
    photo = Image.open(args.image).convert("RGB")
    photo.thumbnail((288, 432))
    pixels = np.asarray(photo)
    tracker = DartfVideoTracker()

    for run, frames in enumerate((12, 3), 1):
        print(f"Starting DARTF stream {run}", flush=True)
        session = tracker.start_stream(args.prompt)
        history, times = [], []
        try:
            for index in range(frames):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                left = 16 + index * 8
                frame[24:24 + pixels.shape[0], left:left + pixels.shape[1]] = pixels
                instances, ms = tracker.track_frame(session, frame)
                assert all(i.mask.shape == frame.shape[:2] and i.mask.any() for i in instances)
                assert all(i.obj_id is not None and np.isfinite(i.score) for i in instances)
                assert draw_instances(Image.fromarray(frame), instances).size == (640, 480)
                ids = {i.obj_id for i in instances}
                history.append(ids)
                times.append(ms)
                print(f"Frame {index}: {len(instances)} instances, ids {sorted(ids)}, {ms:.0f} ms", flush=True)
            assert history[-1], "DARTF confirmed no objects; check the prompt, image and engine outputs"
            assert history[2] & history[-1], "No confirmed ID survived the translation"
            if run == 2:
                assert min(history[-1]) == 1, "A new DARTF stream must restart IDs at 1"
            else:
                mean = float(np.mean(times[8:]))
                print(f"Settled tail: {mean:.0f} ms, {1000 / mean:.1f} fps including transfer", flush=True)
        finally:
            session.reset_inference_session()
    print("PASS: DARTF masks, IDs, transfer, translation and restart", flush=True)


if __name__ == "__main__":
    main()
