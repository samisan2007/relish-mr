"""Exercise real hybrid inference on a translated local image, without a camera.

From perception/: .venv/Scripts/python.exe tests/smoke_hybrid.py photo.jpg meatball
Add --sam31 to ground with the SAM 3.1 Docker worker instead of in-process SAM3, and
--reground 2 to force a mid-stream re-ground and see what it does to track IDs.
Requires cached weights and an image where the seeder detects the prompt.
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

from sam31_runner import Sam31Runner
from video_runner import Sam3VideoTracker
from viz import draw_instances
from yoloe_runner import HybridVideoTracker, YOLOE_MODEL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('prompt')
    parser.add_argument('--model', type=Path, default=Path(YOLOE_MODEL_ID))
    parser.add_argument('--sam31', action='store_true',
                        help='ground with the SAM 3.1 worker instead of in-process SAM3')
    parser.add_argument('--reground', type=int, default=0,
                        help='re-ground every N frames; 2 forces attempts inside this run')
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
    if args.sam31:
        print('Starting the SAM 3.1 worker and loading YOLOE weights', flush=True)
        seeder = Sam31Runner()
    else:
        print('Loading local SAM3 and YOLOE weights', flush=True)
        seeder = Sam3VideoTracker(dtype=torch.float16 if torch.cuda.is_available() else None)
    hybrid = HybridVideoTracker(seeder, model_id=str(args.model), reground_every=args.reground)
    references = []
    original_get_vpe = YOLOEVPSegPredictor.get_vpe

    def recording_get_vpe(predictor, source):
        references.append(np.array(source, copy=True))
        return original_get_vpe(predictor, source)

    moved_frames = [moved_frame, moved_frame, positioned(320)]
    # Scheduled re-grounding is the opposite check to the one-shot guarantees below, and
    # needs room for the seeder to miss a frame and be retried, so give it more frames.
    quiet = not args.reground
    if not quiet:
        moved_frames = [positioned(336 if i % 2 == 0 else 320) for i in range(8)]
    grounds = {'attempts': 0, 'hits': 0}
    inner_ground = hybrid._ground

    def counting_ground(session, frame):
        instances, ms = inner_ground(session, frame)
        grounds['attempts'] += 1
        grounds['hits'] += bool(instances)
        return instances, ms

    hybrid._ground = counting_ground
    sessions = []
    try:
        with patch.object(YOLOEVPSegPredictor, 'get_vpe', new=recording_get_vpe):
            sessions.append(hybrid.start_stream(args.prompt))
            session = sessions[-1]
            seeds, ms = hybrid.track_frame(session, seed_frame)
            assert seeds, 'the seeder must detect something to exercise the handoff'
            print(f'Seed: {len(seeds)} detections; {ms:.0f} ms', flush=True)
            predictor = None
            for index, frame in enumerate(moved_frames):
                instances, ms = hybrid.track_frame(session, frame)
                assert instances, 'YOLOE lost every detection after translation'
                assert any((i.box_xyxy[0] + i.box_xyxy[2]) / 2 > 320 for i in instances)
                if quiet:
                    assert len(references) == 1, 'Visual embeddings were recomputed on a later frame'
                    np.testing.assert_array_equal(references[0], seed_frame[:, :, ::-1])
                    if predictor is None:
                        predictor = hybrid.model.predictor
                    else:
                        assert hybrid.model.predictor is predictor, 'Persistent tracker was replaced'
                assert all(i.mask.shape == frame.shape[:2] for i in instances)
                assert draw_instances(Image.fromarray(frame), instances).size == (640, 480)
                print(f'Moved frame {index}: {len(instances)} detections; '
                      f'IDs {[i.obj_id for i in instances]}; {len(references)} groundings so far; '
                      f'{ms:.0f} ms', flush=True)
            assert session['reference_image'] is None and session['visual_prompts'] is None
            if not quiet:
                # The seeder may legitimately miss a scheduled frame, so assert on the
                # schedule (ours) and on the implication (a hit must reinstall exemplars)
                # rather than on the seeder getting lucky.
                assert grounds['attempts'] > 1, f'--reground {args.reground} never fired'
                if grounds['hits'] > 1:
                    assert len(references) > 1, 'a re-ground found objects but did not reinstall exemplars'
                    print(f"Re-grounded {grounds['hits'] - 1}x mid-stream "
                          f"({grounds['attempts'] - 1} attempts); note that installing new "
                          f'exemplars restarts the IDs printed above', flush=True)
                else:
                    print('Inconclusive: no scheduled re-ground found the object', flush=True)

            sessions.append(hybrid.start_stream(args.prompt))
            new_session = sessions[-1]
            new_seeds, _ = hybrid.track_frame(new_session, moved_frame)
            assert new_seeds
            new_instances, _ = hybrid.track_frame(new_session, seed_frame)
            assert new_instances
            if quiet:
                assert len(references) == 2, 'New stream did not extract a new reference'
                np.testing.assert_array_equal(references[1], moved_frame[:, :, ::-1])
                assert hybrid.model.predictor is not predictor
            print(f'Restarted stream: {len(new_instances)} detections; fresh predictor', flush=True)
    finally:
        # A SAM 3.1 seeder holds a container per stream; a failed assert must not strand it.
        for opened in sessions:
            opened.reset_inference_session()
    print('Hybrid translation, embedding reuse, overlays and restart: PASS', flush=True)


if __name__ == '__main__':
    main()
