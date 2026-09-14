"""Prepare and benchmark upstream DARTF FAST in the SM86 Docker image.

Portable CPU exports can run elsewhere. Plans and inference require the target
GPU. The existing FP16 webcam backend and its assets are independent.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import urllib.error
import urllib.request

from build_engines import DART_COMMIT, export_text, run

DART = Path('/opt/dart')
ROOT = DART / 'dartf'
HF_REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'
STAGES = ('download', 'export', 'gptq', 'quantize', 'text', 'head', 'reference', 'plans', 'verify')
CHECK_IDS = ('000000473003', '000000397133', '000000037777')


def checked_gpu():
    import torch
    import tensorrt as trt

    if not torch.cuda.is_available():
        raise RuntimeError('Docker cannot see an NVIDIA GPU. Check Docker Desktop GPU access.')
    prop = torch.cuda.get_device_properties(0)
    if (prop.major, prop.minor) != (8, 6):
        raise RuntimeError(f'This image targets SM86 (RTX 3080); found {prop.name}, SM{prop.major}{prop.minor}. '
                           'CPU preparation is possible with --stage download/export/gptq/quantize/text/head/reference.')
    return {'name': prop.name, 'compute_capability': [prop.major, prop.minor],
            'memory_bytes': prop.total_memory, 'tensorrt': trt.__version__}


def check_target(out):
    gpu = checked_gpu()
    manifest = out / 'gpu.json'
    if manifest.exists() and json.loads(manifest.read_text()) != gpu:
        raise RuntimeError('These engines belong to another GPU/runtime. Use a fresh work directory.')
    return gpu


def calibration_from_scales(scales):
    """Convert the shipped, measured quantization scales into the quantizer's schema."""
    expected = {f'block{b}.{fam}' for b in range(32) for fam in ('qkv', 'proj', 'fc1', 'fc2')}
    if set(scales) != expected or any(not 0 < float(v) < float('inf') for v in scales.values()):
        raise ValueError('Expected all 128 positive, finite DARTF activation scales')
    return {f'calib.{key}': {'p99999': float(value) * 127}
            for key, value in scales.items()}


def download(out):
    from huggingface_hub import snapshot_download

    print('Downloading pinned SAM3 weights (existing cache files are reused)', flush=True)
    snapshot_download('facebook/sam3', revision=HF_REVISION,
                      allow_patterns=['*.json', 'model.safetensors', 'sam3.pt'])
    download_images(out)
    shutil.copyfile(DART / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz', out / 'bpe_simple_vocab_16e6.txt.gz')


def download_images(out):
    from PIL import Image

    ids = json.loads((ROOT / 'calib/calib_ids.json').read_text())
    images = out / 'images'
    images.mkdir(exist_ok=True)
    for iid in dict.fromkeys([*ids, *CHECK_IDS]):
        path = images / f'{iid}.jpg'
        if not path.is_file():
            print('Downloading COCO 2017 image', iid, flush=True)
            partial = path.with_suffix('.part')
            # Official COCO public S3 bucket; HTTPS endpoint with a valid AWS certificate.
            for split in ('val2017', 'train2017'):
                try:
                    with urllib.request.urlopen(f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{iid}.jpg', timeout=60) as src:
                        with partial.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 404 or split == 'train2017':
                        raise
            with Image.open(partial) as image:
                image.verify()
            partial.replace(path)


def checkpoint():
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download('facebook/sam3', 'sam3.pt', revision=HF_REVISION, local_files_only=True))


def export(out):
    import onnx

    run(sys.executable, ROOT / 'export/export_hf_had_nowin.py', DART, out / 'hadnw', 1008,
        out / 'images' / f'{CHECK_IDS[0]}.jpg')
    run(sys.executable, ROOT / 'export/rename_hf_io.py', out / 'hadnw/hf_backbone.onnx',
        out / 'vision_had.onnx', 'vision_had.onnx.data')
    (out / 'hadnw/hf_backbone.onnx.data').replace(out / 'vision_had.onnx.data')
    onnx.checker.check_model(str(out / 'vision_had.onnx'))
    graph = onnx.load(out / 'vision_had.onnx', load_external_data=False).graph
    dimensions = {tensor.name: tuple(tensor.dims) for tensor in graph.initializer}
    expected = {(1024, 1024): 128, (1024, 4736): 32, (4736, 1024): 32}
    sites = Counter(dimensions[node.input[1]] for node in graph.node
        if node.op_type == 'MatMul' and node.input[1] in dimensions
        and '_had' not in node.input[1] and dimensions[node.input[1]] in expected)
    if sites != expected:
        raise RuntimeError(f'Unexpected quantization sites before GPTQ: {dict(sites)}')


def gptq(out, threads):
    # The shipped scales avoid an all-intermediate-output FP32 calibration engine
    # that is too large for a 10 GB card. GPTQ still uses all 16 upstream images.
    run(sys.executable, ROOT / 'quant/gptq_actaware.py', '--images', out / 'images',
        '--ids', ROOT / 'calib/calib_ids.json', '--out', out / 'gptq', '--blocks', '0-31',
        '--act-blocks', '0-31', '--smooth', '0', '--hadamard', '--head-rot',
        '--act-scales', ROOT / 'calib/act_scales.json', '--threads', threads)


def quantize(out):
    scales = json.loads((out / 'gptq_act.json').read_text())['act_scales']
    (out / 'calibration.json').write_text(json.dumps(calibration_from_scales(scales)))
    run(sys.executable, ROOT / 'quant/quantize_hf.py', out / 'vision_had.onnx', out / 'q0.onnx',
        out / 'calibration.json', '--act', 'p99999', '--blocks', '0-31', '--act-blocks', '0-31',
        '--always-act-fams', 'proj', '--gptq-cache', out / 'gptq_cache.npz',
        '--act-override', out / 'gptq_act.json', '--bias-override', out / 'gptq_bias.json')
    for script, source, target, options in (
        ('quant/fuse_gelu.py', 'q0', 'q0g', []),
        ('quant/fold_rope.py', 'q0g', 'q1', []),
        ('plugins/splice_rope_plugin.py', 'q1', 'q2', []),
        ('plugins/splice_attn_plugin.py', 'q2', 'q3', []),
        ('plugins/splice_mlp_plugin.py', 'q3', 'q4', ['--gelu', 'tanh']),
        ('plugins/splice_block_plugins.py', 'q4', 'vision_int8', ['--qkv', '--attnproj']),
    ):
        run(sys.executable, ROOT / script, out / f'{source}.onnx', out / f'{target}.onnx', *options)


def reference(out, threads):
    import numpy as np
    import torch
    from transformers import Sam3Model
    from preprocess import load_image_tensor

    torch.set_num_threads(threads)
    model = Sam3Model.from_pretrained('facebook/sam3', revision=HF_REVISION,
                                     local_files_only=True, attn_implementation='eager').cpu().eval()
    with torch.inference_mode():
        for iid in CHECK_IDS:
            x, _ = load_image_tensor(out / 'images' / f'{iid}.jpg', 1008)
            features = model.vision_encoder(torch.from_numpy(x)).fpn_hidden_states
            np.save(out / f'reference_{iid}.npy', features[2].numpy())


def plans(out):
    gpu = check_target(out)
    manifest = out / 'gpu.json'
    if not manifest.exists() and any(out.glob('*.plan')):
        raise RuntimeError('Found engines without a GPU manifest. Use a fresh work directory.')
    manifest.write_text(json.dumps(gpu, indent=2))
    for name, options in (
        ('vision_int8', ['--plugins', os.environ['LQ_PLUGINS']]),
        ('groundmask_c1_q32_phase_fp16', []),
    ):
        target = out / f'{name}.plan'
        if target.exists():
            continue
        source = 'groundmask_c1_q32_phase' if name.startswith('groundmask') else name
        pending = out / f'{name}.building.plan'
        run(sys.executable, ROOT / 'runtime/build_engine.py', out / f'{source}.onnx', pending,
            '--no-int8', '--workspace-gb', '2', *options)
        pending.replace(target)


def verify(out):
    import numpy as np
    from preprocess import load_image_tensor
    from trt_util import Runner, load_engine

    check_target(out)
    runner = Runner(load_engine(str(out / 'vision_int8.plan')))
    results = []
    for iid in CHECK_IDS:
        pixels, _ = load_image_tensor(out / 'images' / f'{iid}.jpg', 1008)
        actual = runner({'images': pixels}, want=['fpn_2'])['fpn_2'].astype(np.float64).ravel()
        expected = np.load(out / f'reference_{iid}.npy', allow_pickle=False).astype(np.float64).ravel()
        rel = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        results.append({'image': iid, 'relative_l2': rel, 'cosine': cosine})
        print(results[-1], flush=True)
        # A gross numerical-error gate, not a claim of COCO accuracy parity.
        if not (np.isfinite(actual).all() and cosine >= 0.95 and rel <= 0.35):
            raise RuntimeError('INT8 backbone failed the FP32 sanity comparison; inspect outputs before testing.')
    (out / 'verification.json').write_text(json.dumps(results, indent=2))


def benchmark(out, video, prompt, frames, render):
    gpu = check_target(out)
    if not (out / 'verification.json').is_file():
        raise RuntimeError('Run Build first; a verified INT8 engine is required.')
    if not prompt.strip() or ',' in prompt:
        raise ValueError('Use one text prompt; these engines have a single-prompt head.')
    result = out / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    result.mkdir(parents=True)
    for mode in (['headless', 'rendered'] if render else ['headless']):
        stats = result / f'{mode}.json'
        run(sys.executable, ROOT / 'demo/run_video.py', video, result / 'annotated.mp4', prompt,
            '--assets', out, '--vision', 'vision_int8.plan', '--img-pos', 'img_pos_c1.npy',
            '--groundmask', out / 'groundmask_c1_q32_phase_fp16.plan', '--tracker', 'light',
            '--heads', 'masks', '--pipeline', '--duration', '0', '--max-frames', frames,
            '--no-swipe', '--no-lower-third', '--gpu-name', gpu['name'], '--stats', stats,
            '--mot-out', result / f'{mode}-tracks.txt', *(['--no-render'] if mode == 'headless' else []))
        data = json.loads(stats.read_text())
        if data['frames'] < 1:
            raise RuntimeError('No video frames were processed.')
        print(f"{mode}: {1000 / data['wall_ms_per_frame']:.1f} fps, {data['frames']} frames", flush=True)
    (result / 'environment.json').write_text(json.dumps({'gpu': gpu, 'dart_commit': DART_COMMIT,
        'hf_revision': HF_REVISION, 'input': str(video), 'prompt': prompt}, indent=2))
    print('Results saved in', result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/assets'))
    parser.add_argument('--stage', choices=('all', 'check', 'benchmark', *STAGES), default='all')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--prompt', default='pen')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or args.frames < 1:
        parser.error('threads and frames must be positive')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(DART), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != DART_COMMIT:
        raise RuntimeError('Unexpected DART source revision')
    os.environ.update(LQ_HEAD_ROT='1', OMP_NUM_THREADS=str(args.threads))
    # All upstream from_pretrained calls use the already downloaded pinned snapshot.
    if args.stage != 'download':
        os.environ['HF_HUB_OFFLINE'] = '1'
    if args.stage == 'check':
        print(json.dumps(checked_gpu(), indent=2))
    elif args.stage == 'all':
        check_target(args.out)  # Fail early on another GPU, including when stages are cached.
        fingerprint = hashlib.sha256(Path(__file__).read_bytes() +
            Path(__file__).with_name('build_engines.py').read_bytes()).hexdigest()
        recipe = args.out / 'recipe.sha256'
        if recipe.is_file() and recipe.read_text() != fingerprint:
            raise RuntimeError('The build recipe changed. Use a fresh work directory to avoid stale engines.')
        recipe.write_text(fingerprint)
        for stage in STAGES:
            marker = args.out / f'.fast_{stage}.done'
            if marker.is_file() and marker.read_text() == fingerprint:
                print('Already completed:', stage, flush=True)
                continue
            run(sys.executable, Path(__file__).resolve(), '--out', args.out, '--threads', args.threads, '--stage', stage)
            marker.write_text(fingerprint)
    elif args.stage == 'benchmark':
        if not args.video or not args.video.is_file():
            parser.error('--video must be an existing video file')
        benchmark(args.out, args.video, args.prompt, args.frames, args.render)
    elif args.stage == 'download':
        os.environ.pop('HF_HUB_OFFLINE', None)
        download(args.out)
    elif args.stage == 'export':
        export(args.out)
    elif args.stage == 'gptq':
        gptq(args.out, args.threads)
    elif args.stage == 'quantize':
        quantize(args.out)
    elif args.stage == 'text':
        export_text(SimpleNamespace(out=args.out, ckpt=checkpoint(), threads=args.threads))
    elif args.stage == 'head':
        import onnx

        run(sys.executable, ROOT / 'demo/export_ground_mask.py', '--dart', DART, '--ckpt', checkpoint(),
            '--out', args.out, '--buckets', '1', '--phase-conv', '--fused', '--threads', args.threads)
        onnx.checker.check_model(str(args.out / 'groundmask_c1_q32_phase.onnx'))
    elif args.stage == 'reference':
        reference(args.out, args.threads)
    elif args.stage == 'plans':
        plans(args.out)
    elif args.stage == 'verify':
        verify(args.out)


if __name__ == '__main__':
    main()
