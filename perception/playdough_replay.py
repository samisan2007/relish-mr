"""Replay the frozen Quest play-dough windows through an existing backend."""

import argparse
import csv
import hashlib
import json
import subprocess
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dartf_runner import DartfVideoTracker
from keyframe_hybrid import KeyframeHybridTracker
from sam31_runner import Sam31Runner
from sam3_runner import Sam3Runner
from video_runner import Sam3VideoTracker, open_writer
from viz import draw_instances
from yoloe_runner import HybridVideoTracker


WINDOWS = {
    "setup": (1, 0, 172),
    "handling": (2, 900, 1260),
    "return": (3, 540, 1260),
    "pieces": (4, 720, 1080),
}
BACKENDS = ("hybrid", "yoloe", "sam3video", "dartf", "sam31", "sam31c")


def tracker_for(name):
    if name == "hybrid":
        return KeyframeHybridTracker(Sam3Runner(dtype=torch.float16), keyframe_every=10)
    if name == "yoloe":
        return HybridVideoTracker(Sam3VideoTracker(dtype=torch.float16),
                                  model_id=str(Path(__file__).with_name("yoloe-11l-seg.pt")))
    if name == "sam3video":
        return Sam3VideoTracker(dtype=torch.float16)
    if name == "dartf":
        return DartfVideoTracker()
    return Sam31Runner(compile_model=name == "sam31c")


def gpu_used_mb():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5).splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("window", choices=WINDOWS)
    p.add_argument("backend", choices=BACKENDS)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "playdough-5070-20260925")
    args = p.parse_args()
    clip, start, stop = WINDOWS[args.window]
    video = Path(__file__).resolve().parents[2] / "Media" / f"play-dough_{clip:02d}.mp4"
    if not video.is_file():
        p.error(f"Missing {video}")
    out = args.out / args.window / args.backend / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True)
    with video.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    metadata = {
        "status": "running", "backend": args.backend, "window": args.window,
        "prompt": "red dough", "threshold": 0.4, "keyframe_every": 10,
        "video": str(video), "video_sha256": digest,
        "source_frames": [start, stop, 3], "gpu": torch.cuda.get_device_name(0),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True).strip(),
        "timing": "Sequential replay; request times exclude decode, drawing, encoding and startup.",
    }
    (out / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    times = []
    peak_mb = gpu_used_mb()
    started = time.perf_counter()
    try:
        with ExitStack() as cleanup:
            cleanup.callback(cap.release)
            tracker = tracker_for(args.backend)
            session = tracker.start_stream("red dough")
            if hasattr(session, "reset_inference_session"):
                cleanup.callback(session.reset_inference_session)
            writer = open_writer(out / "overlay.mp4", (cap.get(cv2.CAP_PROP_FPS) or 30) / 3)
            cleanup.callback(writer.close)
            with (out / "frames.csv").open("w", newline="", encoding="utf-8") as f:
                rows = csv.writer(f)
                rows.writerow(["request", "source_frame", "request_ms", "ids", "masks"])
                for n, source_frame in enumerate(range(start, stop, 3)):
                    ok, bgr = cap.read()
                    if not ok:
                        raise RuntimeError(f"Video ended before source frame {source_frame}")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    instances, elapsed_ms = tracker.track_frame(session, rgb)
                    times.append(elapsed_ms)
                    ids = [i.obj_id if i.obj_id is not None else -1 for i in instances]
                    masks = (np.stack([i.mask for i in instances]).astype(np.uint8) if instances
                             else np.empty((0, *rgb.shape[:2]), np.uint8))
                    np.savez_compressed(out / f"mask-{source_frame:06d}.npz",
                                        ids=np.asarray(ids, np.int32), masks=np.packbits(masks, axis=-1))
                    rows.writerow([n, source_frame, f"{elapsed_ms:.1f}", " ".join(map(str, ids)), len(ids)])
                    f.flush()
                    writer.append_data(np.asarray(draw_instances(Image.fromarray(rgb), instances)))
                    if n % 10 == 0:
                        used = gpu_used_mb()
                        peak_mb = max(peak_mb or 0, used or 0)
                    if args.backend == "sam3video" and (elapsed_ms > 10000 or (peak_mb or 0) > 11700):
                        raise RuntimeError(f"SAM3 video saturation: request {elapsed_ms:.0f} ms, device memory {peak_mb} MB")
                    if n % 25 == 0:
                        print(f"{args.window}/{args.backend}: {n + 1}/{len(range(start, stop, 3))}, {elapsed_ms:.0f} ms, {len(ids)} masks", flush=True)
                    for _ in range(2):
                        cap.grab()
        metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "aborted" if "SAM3 video saturation" in str(error) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        warm = times[10:] or times
        metadata.update(frames=len(times), total_seconds=time.perf_counter() - started,
                        peak_device_memory_mb=peak_mb)
        if warm:
            metadata["warm"] = {"mean_ms": float(np.mean(warm)), "p50_ms": float(np.median(warm)),
                                "p95_ms": float(np.percentile(warm, 95)), "requests_per_second": 1000 / float(np.mean(warm))}
        (out / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"{metadata['status']}: {out}", flush=True)


if __name__ == "__main__":
    assert WINDOWS["handling"] == (2, 900, 1260) and len(range(900, 1260, 3)) == 120
    main()
