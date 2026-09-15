"""Replay a short clip (or translated photo) through official SAM 3.1 on one GPU."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_runner import Instance
from viz import draw_instances

SOURCE_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
CHECKPOINT_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"


def prepare_frames(source, directory, limit):
    """Save bounded JPEG input so each repeat sees the same frames."""
    directory.mkdir()
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        with Image.open(source) as photo:
            photo = photo.convert("RGB")
            photo.thumbnail((288, 432))
            pixels = np.asarray(photo)
        span = 640 - pixels.shape[1] - 32
        for i in range(limit):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            phase = (i * 8) % (2 * span)
            left = 16 + min(phase, 2 * span - phase)
            frame[24:24 + pixels.shape[0], left:left + pixels.shape[1]] = pixels
            Image.fromarray(frame).save(directory / f"{i:05d}.jpg", quality=95)
        return limit, 10.0, True

    capture = cv2.VideoCapture(str(source))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        while count < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(directory / f"{count:05d}.jpg"), frame):
                raise OSError("Could not write decoded frame")
            count += 1
    finally:
        capture.release()
    if count == 0:
        raise ValueError("The video contains no decodable frames")
    return count, fps, False


def to_instances(output, height, width):
    ids = np.asarray(output["out_obj_ids"])
    masks = np.asarray(output["out_binary_masks"])
    boxes = np.asarray(output["out_boxes_xywh"])
    scores = np.asarray(output["out_probs"])
    if (ids.ndim != 1 or masks.shape != (len(ids), height, width)
            or boxes.shape != (len(ids), 4) or scores.shape != (len(ids),)):
        raise ValueError("SAM 3.1 returned inconsistent mask/box/ID dimensions")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("SAM 3.1 returned non-finite boxes or scores")
    instances = []
    for oid, mask, box, score in zip(ids, masks, boxes, scores):
        x, y, w, h = box * np.array([width, height, width, height])
        instances.append(Instance(mask.astype(bool), (x, y, x + w, y + h), float(score), int(oid)))
    return instances


def run_pass(model, frames_dir, count, prompt, warmup, output_dir, render, fps):
    import torch

    output_dir.mkdir()
    records, times = [], []
    state, iterator, writer = None, None, None
    with Image.open(frames_dir / "00000.jpg") as first:
        width, height = first.size
    try:
        torch.cuda.reset_peak_memory_stats()
        # The pinned generic session wrapper passes offload_state_to_cpu, which
        # multiplex.init_state does not accept. Use the model's public methods.
        state = model.init_state(str(frames_dir), offload_video_to_cpu=True, async_loading_frames=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.add_prompt(state, frame_idx=0, text_str=prompt)
        torch.cuda.synchronize()
        seed_ms = (time.perf_counter() - started) * 1000
        iterator = model.propagate_in_video(state, start_frame_idx=0, reverse=False)
        for expected_index in range(count):
            torch.cuda.synchronize()
            started = time.perf_counter()
            index, output = next(iterator)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - started) * 1000
            if index != expected_index:
                raise RuntimeError(f"Expected frame {expected_index}, received {index}")
            instances = to_instances(output, height, width)
            ids = [i.obj_id for i in instances]
            records.append({"frame": index, "ms": ms, "ids": ids,
                            "boxes": [i.box_xyxy for i in instances],
                            "scores": [i.score for i in instances]})
            times.append(ms)
            if render or index in {0, count - 1}:
                with Image.open(frames_dir / f"{index:05d}.jpg") as frame:
                    overlay = draw_instances(frame, instances)
                if render:
                    if writer is None:
                        # Reuse imageio's ffmpeg writer, already used by the video UI.
                        from video_runner import open_writer
                        writer = open_writer(output_dir / "annotated.mp4", fps)
                    writer.append_data(np.asarray(overlay))
                if index in {0, count - 1}:
                    overlay.save(output_dir / f"frame-{index:05d}.png")
            print(f"Frame {index:03d}: {len(ids)} objects, {ms:.0f} ms, IDs {ids}", flush=True)
        # Upstream prefetches the next detector frame. The final frame reuses
        # that cache without doing another detection, so exclude it from steady speed.
        tail = times[warmup:-1]
        summary = {
            "frames": count, "warmup_frames": warmup, "steady_frames": len(tail), "prompt_ms": seed_ms,
            "propagation_ms_per_frame": float(np.mean(times)),
            "steady_ms_per_frame": float(np.mean(tail)),
            "steady_fps": 1000 / float(np.mean(tail)),
            "steady_p95_ms": float(np.percentile(tail, 95)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "objects_min": min(len(r["ids"]) for r in records[warmup:]),
            "objects_max": max(len(r["ids"]) for r in records[warmup:]),
            "last_ids": records[-1]["ids"],
            "persistent_tail_ids": sorted(set.intersection(*(set(r["ids"]) for r in records[warmup:]))),
        }
        (output_dir / "tracks.json").write_text(json.dumps(records, indent=2))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        if iterator is not None:
            iterator.close()
        if writer is not None:
            writer.close()
        if state is not None:
            state.clear()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/local/runs"))
    args = parser.parse_args()
    if not args.input.is_file() or not args.prompt.strip():
        parser.error("Provide an existing image/video and a non-empty prompt")
    if (args.warmup < 0 or not args.warmup + 2 <= args.frames <= 120
            or not 1 <= args.max_objects <= 16 or args.repeats < 1):
        parser.error("Require warmup >= 0, warmup + 2 <= frames <= 120, 1 <= max-objects <= 16, repeats >= 1")

    import torch
    import sam3
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("Docker cannot see the GPU")
    source = Path(sam3.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError("SAM 3.1 source revision differs from the tested revision")
    checkpoint = hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt",
                                 revision=CHECKPOINT_REVISION, local_files_only=True)
    result_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    result_dir.mkdir(parents=True)
    frames_dir = result_dir / "input"
    count, fps, synthetic = prepare_frames(args.input, frames_dir, args.frames)
    if count <= args.warmup + 1:
        raise ValueError("Clip is shorter than the warm-up period; use a longer clip or lower --warmup")
    environment = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "source_revision": revision, "checkpoint_revision": CHECKPOINT_REVISION,
        "input": str(args.input), "prompt": args.prompt, "synthetic": synthetic,
        "compile": args.compile, "precision": "bf16 autocast", "use_fa3": False,
        "max_objects": args.max_objects, "multiplex_count": 16, "source_fps": fps,
        "frame_batch_size": 1, "hotstart_delay": 0, "detector_prefetch_frames": 1,
        "timing": "GPU-synchronized propagation including upstream postprocessing; excludes decode, prompt, rendering and transfer to Windows",
    }
    (result_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    print(f"Results: {result_dir}\nLoading SAM 3.1 ({environment['gpu']})...", flush=True)
    with (result_dir / "model-load.log").open("w") as log, redirect_stdout(log):
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint, use_fa3=False, compile=args.compile,
            warm_up=False, max_num_objects=args.max_objects, async_loading_frames=False,
        )
    model = predictor.model
    # Process one frame at a time for a live-oriented replay on 12 GB. The
    # official offline demo batches 16 frames and delays output by 15 frames.
    model.use_batched_grounding = False
    model.postprocess_batch_size = 1
    model.hotstart_delay = 0
    summaries = []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            print(f"Pass {repeat + 1}/{args.repeats} (fresh tracking state)", flush=True)
            summaries.append(run_pass(model, frames_dir, count, args.prompt.strip(), args.warmup,
                                      result_dir / f"pass-{repeat + 1}", args.render, fps))
    if synthetic and not all(s["persistent_tail_ids"] for s in summaries):
        raise RuntimeError("Synthetic smoke failed: no ID persisted through the settled frames")
    if synthetic and len({min(s["last_ids"]) for s in summaries}) != 1:
        raise RuntimeError("Synthetic smoke failed: IDs did not restart consistently")
    print(f"PASS: SAM 3.1 replay completed. Results: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
