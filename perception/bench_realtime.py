"""Benchmark real-time webcam tracking options against the exact same frames.

Captures one clip from the webcam, then replays it through several
Sam3VideoTracker configurations (precision, dispatch mode, internal
processing resolution, torch.compile) so the comparison isn't muddied by
webcam/hand-motion variance between runs.

    python bench_realtime.py --prompt "phone" --capture-seconds 8

Capture once, reuse for every config. Move the object during the countdown
so the clip has real motion to track.
"""

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from video_runner import Sam3VideoTracker


@dataclass
class Config:
    name: str
    dtype: "torch.dtype | None" = None
    processor_size: int | None = None
    use_device_map: bool = True
    compile_model: bool = False


# processor_size overrides (504px/288px) are deliberately absent: the video session's
# memory-bank/position-embedding buffers are hardcoded to the default 72x72 (1008px/14)
# token grid, so a real detection throws a tensor-size mismatch once a track is created.
# Sam3VideoTracker still accepts processor_size for when that's fixed upstream, but it's
# not a usable lever today — see the constructor docstring in video_runner.py.
CONFIGS = [
    Config("baseline (fp32, 1008px, device_map)"),
    Config("fp16 autocast, 1008px", dtype=torch.float16),
    Config("bf16 autocast, 1008px", dtype=torch.bfloat16),
    Config("bf16 autocast, 1008px, no device_map", dtype=torch.bfloat16, use_device_map=False),
    Config("bf16 autocast, 1008px, compiled", dtype=torch.bfloat16, compile_model=True),
]


def capture_clip(camera_index: int, seconds: float, width: int = 640, height: int = 480) -> list[np.ndarray]:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")

    print(f"Capturing ~{seconds:.0f}s from camera {camera_index} — move the object now.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    frames = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()

    if not frames:
        raise RuntimeError("No frames captured.")
    print(f"Captured {len(frames)} frames.")
    return frames


def run_config(cfg: Config, frames: list[np.ndarray], prompt: str, warmup: int) -> dict:
    result = {"name": cfg.name}
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        tracker = Sam3VideoTracker(
            dtype=cfg.dtype,
            processor_size=cfg.processor_size,
            use_device_map=cfg.use_device_map,
            compile_model=cfg.compile_model,
        )
        session = tracker.start_stream(prompt)

        times_ms = []
        counts = []
        seen_ids: set[int] = set()
        for i, frame in enumerate(frames):
            instances, ms = tracker.track_frame(session, frame)
            if i >= warmup:  # let compile/cudnn autotune settle before timing
                times_ms.append(ms)
                counts.append(len(instances))
                seen_ids.update(inst.obj_id for inst in instances)

        result["load_s"] = tracker.load_seconds
        result["avg_ms"] = sum(times_ms) / len(times_ms)
        result["median_ms"] = sorted(times_ms)[len(times_ms) // 2]
        result["fps"] = 1000 / result["avg_ms"]
        result["peak_vram_mb"] = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )
        result["frames_with_hit"] = sum(1 for c in counts if c > 0)
        result["total_frames"] = len(times_ms)
        result["distinct_ids"] = len(seen_ids)

        del tracker, session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def print_report(results: list[dict]):
    print()
    header = f"{'config':<32} {'fps':>6} {'avg ms':>8} {'median ms':>10} {'VRAM MB':>9} {'hits':>10} {'ids':>4}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['name']:<32} FAILED: {r['error']}")
            continue
        hits = f"{r['frames_with_hit']}/{r['total_frames']}"
        print(
            f"{r['name']:<32} {r['fps']:>6.1f} {r['avg_ms']:>8.0f} {r['median_ms']:>10.0f} "
            f"{r['peak_vram_mb']:>9.0f} {hits:>10} {r['distinct_ids']:>4}"
        )
    print()
    print("hits = frames where the prompt matched something (quality sanity check, not just speed).")
    print("ids  = distinct track IDs seen (streaming mode over-counts vs. the offline path — expected).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Noun phrase to track, e.g. 'phone'")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--capture-seconds", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=3, help="Frames to discard before timing")
    parser.add_argument("--configs", nargs="*", help="Subset of config names to run (substring match)")
    args = parser.parse_args()

    frames = capture_clip(args.camera_index, args.capture_seconds)

    configs = CONFIGS
    if args.configs:
        configs = [c for c in CONFIGS if any(s.lower() in c.name.lower() for s in args.configs)]

    results = []
    for cfg in configs:
        print(f"\nRunning: {cfg.name} ...")
        r = run_config(cfg, frames, args.prompt, args.warmup)
        results.append(r)
        if "error" in r:
            print(f"  FAILED: {r['error']}")
        else:
            print(f"  {r['fps']:.1f} fps ({r['avg_ms']:.0f}ms avg)")

    print_report(results)


if __name__ == "__main__":
    main()
