"""Isolated candidate tests: fixed-seed propagation or EV-M image grounding."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from sam3_runner import Instance
from viz import draw_instances
from video_runner import open_writer

MODELS = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
EDGE_PATCH_FILE = 'sam2/modeling/perceiver.py'
EDGE_PATCH_BEFORE = 'self.latents_2d.unsqueeze(0).expand(B, -1, -1).view(-1, 1, C)'
EDGE_PATCH_AFTER = EDGE_PATCH_BEFORE.replace('.view(', '.reshape(')


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def setup(local):
    from huggingface_hub import hf_hub_download, snapshot_download
    for name, spec in MODELS.items():
        check_source(name, local, apply_patch=True)
        if 'repo_id' not in spec:
            if not (local / spec['checkpoint']).is_file():
                raise FileNotFoundError(local / spec['checkpoint'])
            continue
        destination = local / 'weights' / name
        if 'filename' in spec:
            hf_hub_download(spec['repo_id'], spec['filename'], revision=spec['checkpoint_revision'],
                            local_dir=destination)
        else:
            snapshot_download(spec['repo_id'], revision=spec['checkpoint_revision'],
                              allow_patterns=['*.json', '*.safetensors'], local_dir=destination)
    print('Pinned candidate checkpoints downloaded.')


def check_source(name, local, apply_patch=False):
    spec = MODELS[name]
    if 'directory' not in spec:
        return
    source = local / spec['directory']
    # Normalize Windows checkout line endings and bind-mount executable bits.
    git = ['git', '-c', f'safe.directory={source}', '-c', 'core.filemode=false',
           '-c', 'core.autocrlf=true', '-C', str(source)]
    revision = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    if revision != spec['revision']:
        raise RuntimeError(f'Unexpected {name} source revision: {revision}')
    allowed = ''
    if name == 'edgetam':
        original = subprocess.check_output(git + ['show', f'HEAD:{EDGE_PATCH_FILE}'], text=True)
        if original.count(EDGE_PATCH_BEFORE) != 1:
            raise RuntimeError('Pinned EdgeTAM reshape patch no longer matches')
        expected = original.replace(EDGE_PATCH_BEFORE, EDGE_PATCH_AFTER)
        path = source / EDGE_PATCH_FILE
        if apply_patch and path.read_text() == original:
            path.write_text(expected)
        if path.read_text() != expected:
            raise RuntimeError('EdgeTAM requires the exact recorded reshape patch; run Setup')
        allowed = f'M {EDGE_PATCH_FILE}'
    dirty = subprocess.check_output(git + ['status', '--porcelain'], text=True).strip()
    if dirty != allowed:
        raise RuntimeError(f'Unexpected candidate source changes: {dirty[:1500]}')


def load_case(path, objects):
    metadata = json.loads((path / 'case.json').read_text(encoding='utf-8'))
    files = [path / 'frames' / name for name in sorted(metadata['frames'])]
    if not 10 <= len(files) <= 120:
        raise ValueError('Prepared cases must contain 10-120 frames')
    if any(f.name != f'{i:05d}.jpg' for i, f in enumerate(files)):
        raise ValueError('Case frames must be consecutive numbered JPEGs')
    if any(sha256(f) != metadata['frames'][f.name] for f in files):
        raise ValueError('Case frames changed since preparation')
    if sha256(path / 'seeds.npz') != metadata['seeds_sha256']:
        raise ValueError('Seed masks changed since preparation')
    with np.load(path / 'seeds.npz', allow_pickle=False) as data:
        masks, ids = data['masks'], data['ids']
    with Image.open(files[0]) as first:
        width, height = first.size
    if (masks.dtype != np.bool_ or masks.ndim != 3 or masks.shape[1:] != (height, width)
            or ids.ndim != 1 or len(ids) != len(masks) or len(set(ids.tolist())) != len(ids)
            or not 1 <= objects <= len(ids) or not masks.reshape(len(masks), -1).any(axis=1).all()):
        raise ValueError('Need distinct IDs, nonempty boolean masks at source size, and enough seed objects')
    return metadata, files, masks[:objects], ids[:objects].tolist()


def instances(ids, masks):
    result = []
    for obj_id, mask in zip(ids, masks):
        ys, xs = np.nonzero(mask)
        if xs.size:
            result.append(Instance(mask, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                                   1.0, int(obj_id)))
    return result


def load_native(name, local):
    spec = MODELS[name]
    sys.path.insert(0, str(local / spec['directory']))
    from sam2.build_sam import build_sam2_video_predictor
    kwargs = dict(config_file=spec['config'], ckpt_path=str(local / spec['checkpoint']),
                  device='cuda', apply_postprocessing=False)
    if name == 'edgetam':
        import timm
        create_model = timm.create_model
        # The full checkpoint loads strictly. Avoid downloading redundant ImageNet
        # initialization which is entirely overwritten by those weights.
        with patch('sam2.modeling.backbones.timm.create_model',
                   side_effect=lambda name, **kw: create_model(name, **{**kw, 'pretrained': False})):
            return build_sam2_video_predictor(**kwargs)
    return build_sam2_video_predictor(**kwargs)


def native_frames(model, case, masks, ids):
    state = model.init_state(str(case / 'frames'), offload_video_to_cpu=True,
                             offload_state_to_cpu=False, async_loading_frames=False)
    for obj_id, mask in zip(ids, masks):
        model.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=mask)
    torch.cuda.synchronize()
    try:
        yield None  # preprocessing and seeding are outside per-frame timing
        for idx, obj_ids, logits in model.propagate_in_video(state):
            yield idx, list(obj_ids), (logits[:, 0] > 0).cpu().numpy()
    finally:
        model.reset_state(state)
        state.clear()


def hf_frames(model, processor, files, masks, ids):
    from keyframe_hybrid import KeyframeHybridTracker
    # Preprocess before timing, as the native predictors do in init_state.
    inputs = [processor(images=Image.open(f).convert('RGB'), device='cpu', return_tensors='pt') for f in files]
    session = processor.init_video_session(inference_device='cuda', video_storage_device='cpu')
    processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=list(ids),
                                               input_masks=list(masks), original_size=inputs[0].original_sizes[0])
    try:
        yield None
        for idx, data in enumerate(inputs):
            out = model(inference_session=session, frame=data.pixel_values[0].to('cuda'), frame_idx=idx)
            result = processor.post_process_masks([out.pred_masks.float()], original_sizes=data.original_sizes,
                                                  binarize=True)[0][:, 0].cpu().numpy()
            present = (torch.sigmoid(out.object_score_logits.float()).flatten() >= 0.5).cpu().numpy()
            result[~present] = False
            KeyframeHybridTracker._prune(session, idx)
            yield idx, list(out.object_ids), result
    finally:
        session.reset_inference_session()


def strict_image_checkpoint(model, checkpoint):
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = weights.get('model', weights)
    weights = {k.removeprefix('detector.').replace('student_trunk.', ''): v for k, v in weights.items()}
    wanted = model.state_dict()
    # Upstream silently tolerates missing tensors. A benchmark must not run partly
    # random weights; allow extra unused heads but require every constructed tensor.
    model.load_state_dict({k: v for k, v in weights.items() if k in wanted}, strict=True)


def load_evm(local):
    sys.path.insert(0, str(local / 'efficientsam3' / 'sam3'))
    import sam3.model_builder as builder
    from sam3.model.sam3_image_processor import Sam3Processor
    with patch.object(builder, '_load_checkpoint', side_effect=strict_image_checkpoint):
        model = builder.build_efficientsam3_image_model(
            checkpoint_path=str(local / MODELS['evm']['checkpoint']), device='cuda',
            backbone_type='efficientvit', model_name='b1', text_encoder_type='MobileCLIP-S0',
            text_encoder_context_length=16, load_from_HF=False, compile=False)
    return model, Sam3Processor(model, device='cuda')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['setup', 'track', 'image'])
    p.add_argument('--model', choices=list(MODELS), default='edgetam')
    p.add_argument('--local', type=Path, default=Path('/local'))
    p.add_argument('--input', type=Path)
    p.add_argument('--objects', type=int, default=3)
    p.add_argument('--prompt', default='meatball')
    p.add_argument('--threshold', type=float, default=0.4)
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    if args.action == 'setup':
        setup(args.local)
        return
    if not args.input or not args.input.exists() or not torch.cuda.is_available():
        p.error('Existing input and a CUDA GPU are required')
    if not 1 <= args.repeats <= 10 or not 0 <= args.threshold <= 1 or not args.prompt.strip():
        p.error('Invalid repeats, threshold or prompt')
    if (args.action == 'image') != (args.model == 'evm'):
        p.error('EV-M is an image detector; other models require a prepared tracking case')
    spec = MODELS[args.model]
    check_source(args.model, args.local)
    run = args.local / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f') + '-' + args.model)
    run.mkdir(parents=True)
    checkpoint = args.local / spec['checkpoint']
    checkpoint_files = [checkpoint] if checkpoint.is_file() else sorted(checkpoint.glob('*.safetensors'))
    manifest = dict(model=args.model, spec=spec, settings={**vars(args), 'local': str(args.local), 'input': str(args.input)},
                    gpu=torch.cuda.get_device_name(), precision='fp16 autocast', cuda=torch.version.cuda,
                    docker_image_id=os.environ.get('CANDIDATE_IMAGE_ID'), compilation=False,
                    packages={d.metadata['Name']: d.version for d in distributions()},
                    checkpoint_sha256={f.name: sha256(f) for f in checkpoint_files},
                    harness_sha256=sha256(__file__), status='running', passes=[])
    manifest['helper_sha256'] = {name: sha256(ROOT.parent / name) for name in
                                 ['sam3_runner.py', 'video_runner.py', 'viz.py', 'keyframe_hybrid.py']}
    if args.model == 'edgetam':
        manifest['source_patch'] = dict(file=EDGE_PATCH_FILE, before=EDGE_PATCH_BEFORE, after=EDGE_PATCH_AFTER)
    print(f'Results: {run}', flush=True)
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            if args.action == 'image':
                model, processor = load_evm(args.local)
                manifest['architecture'] = dict(vision='efficientvit_b1', text='MobileCLIP-S0', context_length=16,
                                                resolution=1008, parameters=sum(p.numel() for p in model.parameters()))
                processor.confidence_threshold = args.threshold
                image = Image.open(args.input).convert('RGB')
                manifest['input_sha256'] = sha256(args.input)
                for repeat in range(args.repeats):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    output = processor.set_text_prompt(args.prompt, processor.set_image(image))
                    masks = output['masks'][:, 0].cpu().numpy()
                    scores = output['scores'].cpu().tolist()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - t0) * 1000
                    found = instances(range(1, len(masks) + 1), masks)
                    for instance in found:
                        instance.score = scores[instance.obj_id - 1]
                    draw_instances(image, found).save(run / f'pass-{repeat + 1}.png')
                    np.savez_compressed(run / f'pass-{repeat + 1}.npz', masks=masks, scores=scores)
                    manifest['passes'].append(dict(request_ms=elapsed, detections=len(masks), scores=scores,
                                                   peak_gib=torch.cuda.max_memory_allocated() / 2**30))
            else:
                case, files, masks, ids = load_case(args.input, args.objects)
                manifest.update(case=case, selected_ids=ids,
                    timing='Forward fixed-seed propagation with CPU mask transfer; preprocessing/seeding and render/write excluded; not hybrid or live FPS.',
                    native_postprocessing=False, warmup_frames=8)
                if args.model == 'edgetam-hf':
                    from transformers import EdgeTamVideoModel, Sam2VideoProcessor, TimmWrapperConfig
                    from keyframe_hybrid import _fix_mask_seed_scores
                    # Config repr constructs a default vision config, which otherwise
                    # fetches timm metadata even with local_files_only. The complete
                    # backbone config is already included in the pinned checkpoint.
                    def local_backbone(repo_id, **kwargs):
                        if repo_id != 'timm/repvit_m1.dist_in1k':
                            raise ValueError(f'Unexpected backbone config request: {repo_id}')
                        config = json.loads((checkpoint / 'config.json').read_text())
                        return TimmWrapperConfig(**config['vision_config']['backbone_config'])
                    with patch('transformers.AutoConfig.from_pretrained', side_effect=local_backbone):
                        model = EdgeTamVideoModel.from_pretrained(checkpoint, local_files_only=True).to('cuda').eval()
                    _fix_mask_seed_scores(model)
                    processor = Sam2VideoProcessor.from_pretrained(checkpoint, local_files_only=True)
                else:
                    model = load_native(args.model, args.local)
                for repeat in range(args.repeats):
                    folder = run / f'pass-{repeat + 1}'
                    folder.mkdir()
                    torch.cuda.reset_peak_memory_stats()
                    iterator = hf_frames(model, processor, files, masks, ids) if args.model == 'edgetam-hf' else native_frames(model, args.input, masks, ids)
                    writer = None
                    timings, visible = [], []
                    try:
                        next(iterator)
                        writer = open_writer(str(folder / 'annotated.mp4'), case['source_fps'] or 25.0)
                        with (folder / 'tracks.csv').open('w', newline='') as f:
                            log = csv.writer(f)
                            log.writerow(['frame', 'request_ms', 'visible_ids'])
                            for _ in files:
                                torch.cuda.synchronize()
                                t0 = time.perf_counter()
                                idx, obj_ids, output_masks = next(iterator)
                                torch.cuda.synchronize()
                                elapsed = (time.perf_counter() - t0) * 1000
                                found = instances(obj_ids, output_masks)
                                timings.append(elapsed)
                                visible.append(len(found))
                                log.writerow([idx, elapsed, ' '.join(str(d.obj_id) for d in found)])
                                f.flush()
                                np.savez_compressed(folder / f'{idx:05d}.npz', ids=obj_ids, masks=output_masks)
                                writer.append_data(np.asarray(draw_instances(Image.open(files[idx]).convert('RGB'), found)))
                    finally:
                        iterator.close()
                        if writer is not None:
                            writer.close()
                    warm = timings[8:]
                    manifest['passes'].append(dict(frames=len(timings), mean_ms=float(np.mean(warm)),
                        p50_ms=float(np.median(warm)), p95_ms=float(np.percentile(warm, 95)),
                        requests_per_second=1000 / float(np.mean(warm)), frames_with_masks=sum(v > 0 for v in visible),
                        peak_gib=torch.cuda.max_memory_allocated() / 2**30))
            manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['total_seconds'] = time.perf_counter() - started
        (run / 'environment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['passes'], indent=2))
    print('Masks/IDs need visual evaluation; speed and persistent slot IDs do not prove correct tracking.')


if __name__ == '__main__':
    main()
