"""Exercise SAM3 image detection + ByteTrack on a translated local image, no camera.

From perception/: .venv/Scripts/python.exe tests/smoke_sam3_image_track.py photo.jpg meatball
Requires cached SAM3 weights and an image where SAM3 detects the prompt.
Checks that ids survive translation and that a new stream restarts them; it does
not check live motion, occlusion, or steady-state speed.
"""

import argparse
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from sam3_runner import Sam3ImageTracker, Sam3Runner
from viz import draw_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('prompt')
    args = parser.parse_args()

    photo = Image.open(args.image).convert('RGB')
    photo.thumbnail((288, 432))
    pixels = np.array(photo)

    def positioned(left):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[24:24 + pixels.shape[0], left:left + pixels.shape[1]] = pixels
        return frame

    print('Loading local SAM3 image weights', flush=True)
    tracker = Sam3ImageTracker(Sam3Runner())

    session = tracker.start_stream(args.prompt)
    # 16px/frame. ByteTrack associates by IoU, so ids only survive while consecutive
    # detections still overlap: measured on this image (~30px meatballs), ids hold at
    # 0/8/16px per frame and collapse at 32px+. Anything faster is a known limitation
    # of this backend, not a regression — don't "fix" the test by shrinking the step.
    ids_per_frame = []
    for index, left in enumerate(range(16, 16 + 6 * 16, 16)):
        frame = positioned(left)
        instances, ms = tracker.track_frame(session, frame)
        assert instances, f'SAM3 detected nothing on frame {index}'
        assert all(i.mask.shape == frame.shape[:2] for i in instances)
        assert draw_instances(Image.fromarray(frame), instances).size == (640, 480)
        ids = {i.obj_id for i in instances if i.obj_id is not None}
        ids_per_frame.append(ids)
        print(f'Frame {index} (left={left}): {len(instances)} detections; '
              f'ids {sorted(ids)}; {ms:.0f} ms', flush=True)

    assert ids_per_frame[-1], 'ByteTrack confirmed no track by the last frame'
    # ByteTrack needs a couple of frames to confirm, so compare the settled tail.
    survivors = ids_per_frame[1] & ids_per_frame[-1]
    assert survivors, (f'No id survived translation: {ids_per_frame[1]} -> '
                       f'{ids_per_frame[-1]}')
    print(f'Ids surviving the full translation: {sorted(survivors)}', flush=True)

    fresh = tracker.start_stream(args.prompt)
    restarted, _ = tracker.track_frame(fresh, positioned(16))
    assert restarted, 'Restarted stream detected nothing'
    assert min(i.obj_id for i in restarted if i.obj_id is not None) == 1, \
        'A new stream must restart ids at 1'
    print('SAM3 image + ByteTrack translation, overlays and restart: PASS', flush=True)


if __name__ == '__main__':
    main()
