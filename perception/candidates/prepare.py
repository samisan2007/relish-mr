"""Freeze decoded frames and SAM3 seed proposals for an offline tracker comparison."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_runner import Sam3Runner
from viz import draw_instances


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('prompt')
    p.add_argument('--start', type=int, default=0, help='First source video frame')
    p.add_argument('--frames', type=int, default=60)
    p.add_argument('--threshold', type=float, default=0.4)
    p.add_argument('--max-objects', type=int, default=16)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'candidates-local' / 'cases')
    args = p.parse_args()
    if not args.input.is_file() or not args.prompt.strip() or not 10 <= args.frames <= 120:
        p.error('Provide a file, prompt and 10-120 frames')
    if args.start < 0 or not 0 <= args.threshold <= 1 or not 1 <= args.max_objects <= 16:
        p.error('Require start >= 0, threshold in [0, 1], and max-objects in [1, 16]')
    case = args.out / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    frames_dir = case / 'frames'
    frames_dir.mkdir(parents=True)
    photo = cv2.imread(str(args.input))
    synthetic = photo is not None
    cap = None if synthetic else cv2.VideoCapture(str(args.input))
    files = []
    try:
        if cap is not None:
            if not cap.isOpened():
                raise ValueError('Cannot open video')
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        fps = 25.0 if synthetic else cap.get(cv2.CAP_PROP_FPS)
        for i in range(args.frames):
            if synthetic:
                shift = min(i, 20)
                frame = cv2.warpAffine(photo, np.float32([[1, 0, shift], [0, 1, 0]]),
                                       (photo.shape[1], photo.shape[0]))
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            path = frames_dir / f'{i:05d}.jpg'
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f'Could not write {path}')
            files.append(path)
    finally:
        if cap is not None:
            cap.release()
    if len(files) < 10:
        raise ValueError('Need at least 10 decoded frames; choose an earlier start')
    # Seed the exact saved JPEG consumed by all trackers, not a pre-encoding frame.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    detector = Sam3Runner(dtype=torch.float16)
    first = Image.open(files[0]).convert('RGB')
    result = detector.segment(first, args.prompt, args.threshold)
    seeds = sorted((d for d in result.instances if d.mask.any()), key=lambda d: d.score,
                   reverse=True)[:args.max_objects]
    if not seeds:
        raise RuntimeError('No seed proposals. Choose a visible-object start frame or another prompt.')
    for obj_id, seed in enumerate(seeds, 1):
        seed.obj_id = obj_id
    np.savez_compressed(case / 'seeds.npz', masks=np.stack([d.mask for d in seeds]),
                        ids=np.array([d.obj_id for d in seeds], dtype=np.int64))
    draw_instances(first, seeds).save(case / 'seed-overlay.png')
    metadata = {
        'input': str(args.input.resolve()), 'input_sha256': sha256(args.input),
        'synthetic_translation': synthetic, 'start': args.start, 'source_fps': fps,
        'prompt': args.prompt, 'threshold': args.threshold, 'objects': len(seeds),
        'seed_source': 'SAM3 proposals, not ground truth; inspect seed-overlay.png',
        'sam3_revision': getattr(detector.model.config, '_commit_hash', None),
        'frames': {path.name: sha256(path) for path in files},
        'seeds_sha256': sha256(case / 'seeds.npz'),
    }
    (case / 'case.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Prepared {len(files)} frames, {len(seeds)} seed proposals: {case}')
    print('Inspect seed-overlay.png before comparing. Synthetic translation checks wiring only.')


if __name__ == '__main__':
    main()
