"""Run the pinned DARTF FAST pipeline on JPEG frames arriving on stdin."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from build_engines import run
from fast import DART_COMMIT, HF_REVISION, ROOT, check_target


class UiReader:
    """Adapt the UI's lossless image packets to upstream's RGB frame reader."""

    def __init__(self, stream):
        self.stream = stream
        self.stdout = self

    def read(self, count):
        import cv2
        from dartf_worker import read_packet

        packet = read_packet(self.stream)
        if packet is None:
            return b''
        if set(packet) != {'png'}:
            raise ValueError('FAST expects one PNG frame')
        frame = cv2.imdecode(packet['png'], cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (480, 640, 3):
            raise ValueError('FAST expects a 640x480 color frame')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (1008, 1008), interpolation=cv2.INTER_LINEAR).tobytes()

    def kill(self):
        pass


def ui_source(source):
    decoder = next((line for line in source.splitlines() if line.startswith('else: dec_m = SeqReader(a.video)')), None)
    if decoder is None or source.count(decoder) != 1:
        raise RuntimeError('Upstream FAST decoder changed')
    source = source.replace(decoder, 'else: dec_m = UiReader(sys.stdin.buffer)')
    patches = {
        'import threading, queue': '''import threading, queue
import cv2
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/dartf')
from dartf_worker import write_packet
from live import UiReader
protocol = os.fdopen(os.dup(1), 'wb', buffering=0)
os.dup2(2, 1)
sys.stdout = sys.stderr''',
        't_start = time.time(); fi = 0': '''t_start = time.time(); fi = 0
write_packet(protocol, ready=True)''',
        '    if a.no_render:': '''    masks = np.stack([cv2.resize(np.asarray(t.mask.cpu().numpy() if hasattr(t.mask, 'cpu') else t.mask), (SW, SH), interpolation=cv2.INTER_LINEAR) > 0 for t in shown]) if shown else np.empty((0, SH, SW), bool)
    boxes = np.asarray([np.asarray(t.box) * [SW / OW, SH / OH, SW / OW, SH / OH] for t in shown], np.float32).reshape(-1, 4)
    write_packet(protocol, packed_masks=np.packbits(masks, axis=-1), boxes=boxes, scores=np.asarray([t.p for t in shown], np.float32), ids=np.asarray([t.id for t in shown], np.int64), inference_ms=(time.perf_counter() - t0) * 1000, vision_ms=(t1 - t0) * 1000, head_ms=(t2 - t1) * 1000)
    if a.no_render:''',
    }
    for anchor, replacement in patches.items():
        if source.count(anchor) != 1:
            raise RuntimeError(f'Upstream FAST source changed near {anchor!r}')
        source = source.replace(anchor, replacement)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='pen')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--ui', action='store_true')
    args = parser.parse_args()
    if not args.prompt.strip() or ',' in args.prompt or args.frames < 1:
        parser.error('Use one non-empty prompt and a positive frame count')

    out = Path('/assets')
    if subprocess.check_output(['git', '-C', str(ROOT.parent), 'rev-parse', 'HEAD'], text=True).strip() != DART_COMMIT:
        raise RuntimeError('Unexpected DART source revision')
    os.environ.update(LQ_HEAD_ROT='1', HF_HUB_OFFLINE='1')
    gpu = check_target(out)
    if not (out / 'verification.json').is_file():
        raise RuntimeError('Run Build first; a verified INT8 engine is required.')
    if not args.ui:
        result = out / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        result.mkdir(parents=True)

    upstream = ROOT / 'demo/run_video.py'
    source = upstream.read_text()
    if args.ui:
        source = ui_source(source)
    else:
        anchor = 'mot_f = open(a.mot_out, "w") if a.mot_out else None'
        if source.count(anchor) != 1:
            raise RuntimeError('Upstream MOT output changed; review the live output patch.')
        # The duplicate descriptor can close at the end without closing normal stdout.
        patch = 'mot_f = os.fdopen(os.dup(1), "w", buffering=1) if a.mot_out == "-" else open(a.mot_out, "w") if a.mot_out else None'
        source = source.replace(anchor, patch)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix='relish_live_',
                                     dir=upstream.parent, delete=False) as temp:
        temp.write(source)
        runner = Path(temp.name)
    try:
        if args.ui:
            subprocess.run([sys.executable, '-u', str(runner), 'pipe:0', '/tmp/unused.mp4', args.prompt,
                '--assets', str(out), '--vision', 'vision_int8.plan', '--img-pos', 'img_pos_c1.npy',
                '--groundmask', str(out / 'groundmask_c1_q32_phase_fp16.plan'), '--tracker', 'light',
                '--heads', 'masks', '--duration', '0', '--no-render', '--fps-in', '30',
                '--src-size', '640x480'], check=True)
            return
        run('python', '-u', runner, 'pipe:0', result / 'annotated.mp4', args.prompt,
            '--assets', out, '--vision', 'vision_int8.plan', '--img-pos', 'img_pos_c1.npy',
            '--groundmask', out / 'groundmask_c1_q32_phase_fp16.plan', '--tracker', 'light',
            '--heads', 'masks', '--pipeline', '--duration', '0', '--max-frames', args.frames,
            '--no-swipe', '--no-lower-third', '--gpu-name', gpu['name'],
            '--stats', result / 'headless.json', '--mot-out', '-', '--fps-in', '30',
            '--src-size', '640x480', '--no-render')
    finally:
        runner.unlink()
    data = json.loads((result / 'headless.json').read_text())
    if data['frames'] < 1:
        raise RuntimeError('No webcam frames were processed.')
    (result / 'environment.json').write_text(json.dumps({'gpu': gpu, 'dart_commit': DART_COMMIT,
        'hf_revision': HF_REVISION, 'input': 'webcam', 'prompt': args.prompt}, indent=2))
    print(f"headless: {1000 / data['wall_ms_per_frame']:.1f} fps, {data['frames']} frames", flush=True)
    print('Results saved in', result, flush=True)


if __name__ == '__main__':
    main()
