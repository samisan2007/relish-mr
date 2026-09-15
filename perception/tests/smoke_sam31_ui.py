"""Real GPU checks of the image, file and webcam handlers; synthetic camera input.

From perception/: .venv/Scripts/python.exe tests/smoke_sam31_ui.py [--compile]
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ui
import video_ui
from sam31.benchmark import prepare_frames
from sam31_runner import MODEL_SAM31, MODEL_SAM31_COMPILED
from video_runner import open_writer


class Progress:
    def tqdm(self, iterable, **kwargs):
        return iterable


class Camera:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def set(self, *args):
        return True

    def isOpened(self):
        return not self.released

    def read(self):
        frame = next(self.frames, None)
        return (False, None) if frame is None else (True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.released = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    model = MODEL_SAM31_COMPILED if args.compile else MODEL_SAM31
    root = Path(__file__).resolve().parents[1]
    output = root / "sam31-local" / ("ui-smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    photo = root.parents[1] / "Media" / "meatballs_img.jpg"
    prepare_frames(photo, output / "frames", 40)
    frames = [np.asarray(Image.open(path).convert("RGB")) for path in sorted((output / "frames").glob("*.jpg"))]
    print(f"Testing {model}; artifacts: {output}", flush=True)
    results = {"model": model}

    annotated, status = ui.run(Image.fromarray(frames[0]), "meatball", 0.5, True, True, model)
    assert annotated is not None and "found in" in status, status
    annotated.save(output / "image.png")
    results["image"] = status
    print("IMAGE PASS:", status.splitlines()[0], flush=True)

    video_path = output / "input.mp4"
    with open_writer(video_path, 10) as writer:
        for frame in frames:
            writer.append_data(frame)
    tracked_path, status = video_ui.run_file(str(video_path), "meatball", 40, 1, True, True, model, Progress())
    assert tracked_path is not None and "Distinct track IDs:" in status, status
    capture = cv2.VideoCapture(tracked_path)
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (480, 640, 3)
        count += 1
    capture.release()
    assert count == 40, count
    results["file"] = {"video": tracked_path, "status": status}
    print("FILE PASS:", status, flush=True)

    initial_ids = []
    for repeat in range(2):
        # Objects enter after startup, exercising creation of tracks on later frames.
        camera = Camera([np.zeros_like(frames[0])] * 3 + frames)
        stream = video_ui.webcam_loop("meatball", True, True, 0, model, video_ui.DEFAULT_CONFIG.name, {})
        states = []
        with patch.object(video_ui.cv2, "VideoCapture", return_value=camera):
            try:
                for _ in range(13):  # One startup status, then twelve live results.
                    overlay, status, state = next(stream)
                    if overlay is not None:
                        assert overlay.shape == frames[0].shape
                        states.append(state)
                assert states and states[-1]["counts"][-1] > 0, status
                initial_ids.append(min(states[-1]["ids"]))
                Image.fromarray(overlay).save(output / f"webcam-{repeat + 1}.png")
                _, history, _, _ = video_ui.finalize_run(state, [])
                results[f"webcam_{repeat + 1}"] = history[0]
                results[f"webcam_{repeat + 1}_fps_after_8"] = 1000 / float(np.mean(state["times"][8:]))
            finally:
                stream.close()
        assert camera.released
        assert not video_ui._webcam_runs
        print("WEBCAM PASS:", results[f"webcam_{repeat + 1}"], flush=True)
    assert initial_ids[0] == initial_ids[1] == 0, initial_ids
    (output / "results.json").write_text(json.dumps(results, indent=2))
    print("PASS: all UI paths, video decoding, camera cleanup and ID restart", flush=True)


if __name__ == "__main__":
    main()
