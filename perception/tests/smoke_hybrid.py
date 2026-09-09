"""Exercise real hybrid inference on a translated local image, without a camera.

From perception/: .venv/Scripts/python.exe tests/smoke_hybrid.py photo.jpg meatball
Requires cached SAM3/YOLOE weights and an image where SAM3 detects the prompt.
This checks the handoff, not live motion or occlusion quality.
"""

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

from video_runner import Sam3VideoTracker
from viz import draw_instances
from yoloe_runner import HybridVideoTracker, YOLOE_MODEL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('prompt')
    parser.add_argument('--model', type=Path, default=Path(YOLOE_MODEL_ID))
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f'Local YOLOE weights are required: {args.model}')

    photo = Image.open(args.image).convert('RGB')
    photo.thumbnail((288, 432))
    pixels = np.array(photo)

    def positioned(left):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[24:24 + pixels.shape[0], left:left + pixels.shape[1]] = pixels
        return frame

    seed_frame, moved_frame = positioned(16), positioned(336)
    print('Loading local SAM3 and YOLOE weights', flush=True)
    sam = Sam3VideoTracker(dtype=torch.float16 if torch.cuda.is_available() else None)
    hybrid = HybridVideoTracker(sam, model_id=str(args.model))
    references = []
    original_get_vpe = YOLOEVPSegPredictor.get_vpe

    def recording_get_vpe(predictor, source):
        references.append(np.array(source, copy=True))
        return original_get_vpe(predictor, source)

    with patch.object(YOLOEVPSegPredictor, 'get_vpe', new=recording_get_vpe):
        session = hybrid.start_stream(args.prompt)
        seeds, ms = hybrid.track_frame(session, seed_frame)
        assert seeds, 'SAM3 must detect something to exercise the handoff'
        print(f'SAM3 seed: {len(seeds)} detections; {ms:.0f} ms', flush=True)
        predictor = None
        for index, frame in enumerate([moved_frame, moved_frame, positioned(320)]):
            instances, ms = hybrid.track_frame(session, frame)
            assert instances, 'YOLOE lost every detection after translation'
            assert any((i.box_xyxy[0] + i.box_xyxy[2]) / 2 > 320 for i in instances)
            assert len(references) == 1, 'Visual embeddings were recomputed on a later frame'
            np.testing.assert_array_equal(references[0], seed_frame[:, :, ::-1])
            if predictor is None:
                predictor = hybrid.model.predictor
            else:
                assert hybrid.model.predictor is predictor, 'Persistent tracker was replaced'
            assert all(i.mask.shape == frame.shape[:2] for i in instances)
            assert draw_instances(Image.fromarray(frame), instances).size == (640, 480)
            print(f'Moved frame {index}: {len(instances)} detections; '
                  f'IDs {[i.obj_id for i in instances]}; {ms:.0f} ms', flush=True)
        assert session['reference_image'] is None and session['visual_prompts'] is None

        new_session = hybrid.start_stream(args.prompt)
        new_seeds, _ = hybrid.track_frame(new_session, moved_frame)
        assert new_seeds
        new_instances, _ = hybrid.track_frame(new_session, seed_frame)
        assert new_instances
        assert len(references) == 2, 'New stream did not extract a new reference'
        np.testing.assert_array_equal(references[1], moved_frame[:, :, ::-1])
        assert hybrid.model.predictor is not predictor
        print(f'Restarted stream: {len(new_instances)} detections; fresh predictor', flush=True)
    print('Hybrid translation, embedding reuse, overlays and restart: PASS', flush=True)


if __name__ == '__main__':
    main()
