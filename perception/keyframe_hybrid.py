"""Keyframe hybrid: SAM3 image mode finds objects every K frames, EdgeTAM carries
their masks through the frames in between.

SAM3 video is already this shape internally (detector + SAM2-style tracker), but its
tracker costs ~70 ms per object. Here the detector only runs on keyframes and a much
lighter memory tracker (EdgeTAM, a distilled SAM 2) propagates masks every frame.

Unlike the YOLOE hybrid, a keyframe does not restart the IDs: SAM3's masks are
matched to the live tracks by mask overlap. A confident match re-seeds that track
with SAM3's mask under the same ID, a weak one only keeps it alive; an unmatched
confident detection starts a new track; a track SAM3 gives no support on
`retire_after` consecutive keyframes is retired (hidden, but still matchable).

Same start_stream/track_frame shape as the other backends, so the webcam UI can
swap it in. Run as a script to replay a recorded clip through it (or through the
existing SAM3 backends, for a like-for-like comparison):

    python keyframe_hybrid.py ../../Media/pen_test_vid.mp4 pen
    python keyframe_hybrid.py ../../Media/pen_test_vid.mp4 pen --backend sam3video

Writes video, ID CSV, timeline and run metadata to a fresh directory under runs/.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
import time
from pathlib import Path

import numpy as np
import torch

from dartf_runner import track_worker_frame
from sam3_runner import Instance, Sam3Runner
from sam31_runner import Sam31Runner, Sam31Session

EDGETAM_ID = "yonigozlan/EdgeTAM-hf"

# EdgeTAM attends to the last 7 memory frames and up to 16 object pointers, so
# older non-conditioning outputs are dead weight; keep 16. Keep the 2 newest
# keyframe seeds per object instead of every seed since the stream began.
KEEP_NON_COND = 16
KEEP_COND = 2
# Rebuild the EdgeTAM session once this many dead or retired tracks are in it.
COMPACT_AFTER = 4


class _Stream(dict):
    """Per-stream state. A dict for the UI's run log, plus reset_inference_session()
    so the webcam tab's Stop cleanup releases the EdgeTAM memory as for SAM3, and
    closes a SAM 3.1 detector's Docker worker."""

    def reset_inference_session(self):
        self["edgetam"].reset_inference_session()
        if self["detector"] is not None:
            self["detector"].reset_inference_session()


def mask_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller mask, not IoU. Between keyframes EdgeTAM often
    keeps only part of a rotating pen while SAM3 returns all of it; IoU scores that
    pair low and mints a new ID, containment scores it as the same object."""
    smaller = min(a.sum(), b.sum())
    return float(np.logical_and(a, b).sum() / smaller) if smaller else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Plain IoU. Unlike mask_overlap it tells one pen apart from a mask covering
    two pens, which contains it fully but is twice its size."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def dedupe(masks: list[np.ndarray], rank: list, max_overlap: float, overlap=mask_overlap) -> list[int]:
    """Indices of masks to keep, highest rank first; a mask overlapping a kept one
    by `max_overlap` or more is dropped. With mask_overlap, a fragment inside a
    whole object counts: SAM3 image mode often returns both, and each would
    otherwise seed its own track."""
    kept: list[int] = []
    for i in sorted(range(len(masks)), key=rank.__getitem__, reverse=True):
        if all(overlap(masks[i], masks[k]) < max_overlap for k in kept):
            kept.append(i)
    return kept


def match_masks(
    detections: list[np.ndarray], tracks: dict[int, np.ndarray], min_iou: float
) -> tuple[dict[int, int], list[int]]:
    """Greedy matching. Returns ({det_idx: track_id}, unmatched det_idx).

    A pair qualifies on mask_overlap >= min_iou, so a partial EdgeTAM mask still
    matches SAM3's whole object; qualifying pairs are taken highest IoU first. When
    one track has spread over two touching pens, both detections sit fully inside
    it, and IoU lets the other pen's own track claim its detection first.

    ponytail: greedy, not Hungarian; identical for a handful of well-separated objects,
    switch to scipy's linear_sum_assignment if crowded scenes start swapping IDs.
    """
    pairs = sorted(
        ((mask_iou(d, t), di, tid) for di, d in enumerate(detections) for tid, t in tracks.items()
         if mask_overlap(d, t) >= min_iou),
        reverse=True,
    )
    matched: dict[int, int] = {}
    used_tracks: set[int] = set()
    for _, di, tid in pairs:
        if di in matched or tid in used_tracks:
            continue
        matched[di] = tid
        used_tracks.add(tid)
    return matched, [di for di in range(len(detections)) if di not in matched]


def _fix_mask_seed_scores(model) -> None:
    """Work around a transformers EdgeTAM bug (5.16): a mask-seeded object's score
    comes out 1-D while a propagated object's is 2-D, and forward() torch.cat's
    them, so any frame that re-seeds some objects but not others crashes. Give the
    mask path the decoder path's shape. Drop once fixed upstream."""
    use_mask = model._use_mask_as_output

    def patched(*args, **kwargs):
        out = use_mask(*args, **kwargs)
        out.object_score_logits = out.object_score_logits[..., None]
        return out

    model._use_mask_as_output = patched


def _box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


class KeyframeHybridTracker:
    def __init__(
        self,
        sam3: Sam3Runner | Sam31Runner,
        keyframe_every: int = 10,
        threshold: float = 0.4,
        match_iou: float = 0.3,
        retire_after: int = 1,
        keep_threshold: float = 0.2,
        fp16: bool = True,
    ):
        """sam3: the keyframe detector. A Sam31Runner runs SAM 3.1 image mode in its
        Docker worker, one per stream; the thresholds were tuned on SAM3's scores.
        keyframe_every: run SAM3 on every Nth frame (10 at 25 fps is 2.5 Hz).
        threshold: SAM3 detection score floor for seeding or creating a track.
        match_iou: minimum overlap (over the smaller mask) between a detection and a track.
        retire_after: consecutive keyframes a track may go unconfirmed by SAM3.
        keep_threshold: lower floor at which a detection still confirms an existing
        track. EdgeTAM can slide a track onto pen-like background (a shelf edge) at
        high confidence; SAM3 gives such a ghost no support at all, while a real pen
        it misses at `threshold` usually still scores above this."""
        from transformers import EdgeTamVideoModel, Sam2VideoProcessor

        t0 = time.perf_counter()
        self.sam3 = sam3
        self.model = EdgeTamVideoModel.from_pretrained(EDGETAM_ID).to(
            "cuda" if torch.cuda.is_available() else "cpu").eval()
        _fix_mask_seed_scores(self.model)
        self.processor = Sam2VideoProcessor.from_pretrained(EDGETAM_ID)
        self.keyframe_every = keyframe_every
        self.threshold = threshold
        self.match_iou = match_iou
        self.retire_after = retire_after
        self.keep_threshold = keep_threshold
        self.autocast_dtype = torch.float16 if fp16 else None
        self.load_seconds = time.perf_counter() - t0

    @property
    def device(self) -> str:
        return str(self.model.device)

    def _autocast(self):
        return torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        )

    def start_stream(self, text: str, on_started=None) -> "_Stream":
        """Fresh EdgeTAM session per stream; IDs restart at 1. A SAM 3.1 detector
        boots its worker here; on_started lets Stop cancel that boot, as for SAM 3.1."""
        detector = None
        if isinstance(self.sam3, Sam31Runner):
            detector = Sam31Session(text, self.sam3.compile_model, image_only=True,
                                    threshold=self.keep_threshold, on_started=on_started)
        session = self.processor.init_video_session(
            inference_device=self.model.device, video_storage_device="cpu")
        return _Stream(prompt=text, edgetam=session, detector=detector, frame_idx=0, next_id=1,
                       last_masks={}, misses={}, retired=set(), last_keyframe_ms=0.0)

    def track_frame(self, s: dict, frame: np.ndarray) -> tuple[list[Instance], float]:
        """Track one RGB frame. On keyframes SAM3 runs first and its masks are
        installed before EdgeTAM steps, so the frame's output already reflects them.

        ponytail: SAM3 runs inline, so keyframes stall the stream by its latency.
        Live use should run it on a worker and fast-forward its masks to now.
        """
        from PIL import Image

        t0 = time.perf_counter()
        i = s["frame_idx"]
        sess = s["edgetam"]
        inputs = self.processor(images=frame, device=self.model.device, return_tensors="pt")

        if i % self.keyframe_every == 0:
            if s["detector"] is not None:
                dets, _ = track_worker_frame(s["detector"], frame)
                dets = [d for d in dets if d.score >= self.keep_threshold]
            else:
                with self._autocast():
                    dets = self.sam3.segment(Image.fromarray(frame), s["prompt"], self.keep_threshold).instances
            dets = [dets[k] for k in dedupe([d.mask for d in dets], [d.score for d in dets], 0.6)]
            # Match against every mask EdgeTAM still follows, retired ones included:
            # SAM3 misses thin objects on many frames, so a retirement is often wrong,
            # and a re-detection should revive the old ID rather than mint a new one.
            matched, unmatched = match_masks([d.mask for d in dets], s["last_masks"], self.match_iou)

            seeds = {}  # track id -> mask installed on this frame
            for di, tid in matched.items():
                s["misses"][tid] = 0
                s["retired"].discard(tid)
                if dets[di].score < self.threshold:
                    continue  # a weak detection vouches for the track; its mask is often poor
                if mask_iou(s["last_masks"][tid], dets[di].mask) < 0.5:
                    # EdgeTAM's mask disagrees with SAM3's, e.g. it spread onto a touching
                    # pen. Drop its recent memory, or it re-spreads from that history
                    # within a few frames of the new seed.
                    sess.output_dict_per_obj[sess.obj_id_to_idx(tid)]["non_cond_frame_outputs"].clear()
                seeds[tid] = dets[di].mask
            for di in unmatched:
                if dets[di].score < self.threshold:
                    continue
                seeds[s["next_id"]] = dets[di].mask
                s["misses"][s["next_id"]] = 0
                s["next_id"] += 1
            for tid in set(sess.obj_ids) - set(matched.values()):
                s["misses"][tid] = s["misses"].get(tid, 0) + 1
                if s["misses"][tid] >= self.retire_after:
                    s["retired"].add(tid)

            # Retired and lost tracks keep costing EdgeTAM a pass per frame and the
            # session can't drop single objects, so once enough pile up, restart the
            # session with only the live tracks, under the same IDs.
            if len(set(sess.obj_ids) - set(matched.values())) >= COMPACT_AFTER:
                for tid, m in s["last_masks"].items():
                    if tid not in s["retired"]:
                        seeds.setdefault(tid, m)
                sess.reset_tracking_data()
                s["misses"] = {tid: s["misses"].get(tid, 0) for tid in seeds}
                s["retired"] = set()
            if seeds:
                self.processor.add_inputs_to_inference_session(
                    sess, frame_idx=i, obj_ids=list(seeds), input_masks=list(seeds.values()),
                    original_size=inputs.original_sizes[0])
            s["last_keyframe_ms"] = (time.perf_counter() - t0) * 1000

        instances: list[Instance] = []
        if sess.obj_ids:  # nothing seeded yet -> nothing to propagate
            # Explicit frame_idx: without it the session numbers frames by
            # len(processed_frames), which _prune keeps at ~1.
            with torch.no_grad(), self._autocast():
                out = self.model(inference_session=sess, frame=inputs.pixel_values[0], frame_idx=i)
            masks = self.processor.post_process_masks(
                [out.pred_masks.float()], original_sizes=inputs.original_sizes, binarize=True)[0]
            scores = torch.sigmoid(out.object_score_logits.float()).flatten().tolist()
            s["last_masks"], score_of = {}, {}
            for tid, m, score in zip(out.object_ids, masks[:, 0].cpu().numpy(), scores):
                if score < 0.5 or not m.any():  # EdgeTAM's own "object absent" signal
                    continue
                s["last_masks"][tid], score_of[tid] = m, score
            # Two tracks can land on one object: a keyframe mints a new ID while the old
            # track is momentarily absent, then EdgeTAM re-finds it. Keyframe matching
            # then alternates between the pair, so neither retires. Keep one per object,
            # live over retired, then the older ID; the other can't be matched again.
            # IoU, not containment: a pen's own track lies wholly inside a track that
            # has spread over two touching pens, and must not be dropped for it.
            # ponytail: pairwise full-res masks, box pre-check if many objects get slow.
            ids = list(s["last_masks"])
            keep = set(dedupe([s["last_masks"][t] for t in ids],
                              [(t not in s["retired"], -t) for t in ids], 0.6, overlap=mask_iou))
            for k, tid in enumerate(ids):
                if k not in keep:
                    del s["last_masks"][tid]
                    s["retired"].add(tid)
                elif tid not in s["retired"]:
                    m = s["last_masks"][tid]
                    instances.append(Instance(mask=m, box_xyxy=_box(m), score=score_of[tid], obj_id=tid))
            self._prune(sess, i)
        else:
            s["last_masks"] = {}

        s["frame_idx"] += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return instances, (time.perf_counter() - t0) * 1000

    @staticmethod
    def _prune(sess, i: int) -> None:
        """Bound session growth: without this every frame's pixels and outputs are kept."""
        for k in [k for k in sess.processed_frames if k < i]:
            del sess.processed_frames[k]
        for out in sess.output_dict_per_obj.values():
            for k in [k for k in out["non_cond_frame_outputs"] if k < i - KEEP_NON_COND]:
                del out["non_cond_frame_outputs"][k]
            for k in sorted(out["cond_frame_outputs"])[:-KEEP_COND]:
                del out["cond_frame_outputs"][k]


# --- replay CLI ---------------------------------------------------------------

def _create_run(args):
    """Keep each experiment identifiable; never overwrite a prior run."""
    video = Path(args.video).resolve()
    created = datetime.now(timezone.utc)
    run_dir = Path(args.out) / created.strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    with video.open("rb") as f:
        clip_hash = hashlib.file_digest(f, "sha256").hexdigest()
    root = Path(__file__).resolve().parent
    metadata = {
        "created_utc": created.isoformat(), "settings": vars(args).copy(),
        "video": str(video), "video_sha256": clip_hash,
        "python": sys.version, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "ultralytics", "timm", "numpy")},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*root.glob("*.py"), root / "sam31" / "worker.py"]},
        "timing": "Sequential replay; frame requests exclude decode, drawing, encoding and startup; not live latency.",
        "status": "running",
    }
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE).strip()
        metadata["git_status"] = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.PIPE).rstrip("\r\n")
    except (OSError, subprocess.CalledProcessError) as error:
        metadata["git_error"] = str(error)
    (run_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir, metadata


def _load_backend(name: str, args):
    if name.startswith("hybrid"):
        sam3 = {"hybrid": lambda: Sam3Runner(dtype=torch.float16),
                "hybrid31": Sam31Runner,
                "hybrid31c": lambda: Sam31Runner(compile_model=True)}[name]()
        return KeyframeHybridTracker(sam3, args.keyframe_every, args.threshold,
                                     args.match_iou, args.retire_after, args.keep_threshold,
                                     fp16=not args.fp32)
    if name == "sam3video":
        from video_runner import Sam3VideoTracker
        return Sam3VideoTracker(dtype=torch.float16)
    if name == "sam3image":
        from sam3_runner import Sam3ImageTracker
        return Sam3ImageTracker(Sam3Runner(dtype=torch.float16), threshold=args.threshold)
    raise ValueError(name)


def _timeline_png(ids_per_frame: list[list[int]], path: Path) -> None:
    """One row per track ID, one column per frame, lit where that ID was output.
    Fragmentation shows as many short rows; a stable track as one long one."""
    from PIL import Image

    from viz import COLORS

    all_ids = sorted({i for ids in ids_per_frame for i in ids})
    if not all_ids:
        return
    row = {tid: r for r, tid in enumerate(all_ids)}
    h, rh = len(all_ids), 12
    img = np.full((h * rh, len(ids_per_frame), 3), 30, np.uint8)
    for x, ids in enumerate(ids_per_frame):
        for tid in ids:
            img[row[tid] * rh:(row[tid] + 1) * rh - 2, x] = COLORS[tid % len(COLORS)]
    Image.fromarray(img).resize((max(len(ids_per_frame), 600), h * rh)).save(path)


def main() -> None:
    import cv2

    from video_runner import open_writer
    from viz import draw_instances
    from PIL import Image

    p = argparse.ArgumentParser(description="Replay a clip through a tracking backend")
    p.add_argument("video")
    p.add_argument("prompt")
    p.add_argument("--backend", default="hybrid",
                   choices=["hybrid", "hybrid31", "hybrid31c", "sam3video", "sam3image"],
                   help="hybrid31/hybrid31c: SAM 3.1 keyframes (eager/compiled) from its Docker worker")
    p.add_argument("--keyframe-every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--retire-after", type=int, default=1)
    p.add_argument("--keep-threshold", type=float, default=0.2,
                   help="hybrid: SAM3 score that still confirms an existing track")
    p.add_argument("--fp32", action="store_true", help="hybrid: disable fp16 autocast")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    p.add_argument("--out", default="runs")
    args = p.parse_args()
    if not Path(args.video).is_file() or not args.prompt.strip():
        p.error("Provide an existing video and a non-empty prompt")
    if args.keyframe_every < 1 or args.retire_after < 1 or args.max_frames < 0:
        p.error("Require keyframe-every >= 1, retire-after >= 1 and max-frames >= 0")
    if not all(0 <= value <= 1 for value in (args.keep_threshold, args.threshold, args.match_iou)):
        p.error("Thresholds and match-iou must be between 0 and 1")
    if args.backend.startswith("hybrid") and args.keep_threshold > args.threshold:
        p.error("Hybrid keep-threshold must not exceed threshold")

    video = Path(args.video)
    hybrid = args.backend.startswith("hybrid")
    tag = f"{args.backend}_k{args.keyframe_every}" if hybrid else args.backend
    out_dir, metadata = _create_run(args)
    stem = out_dir / f"{video.stem}_{tag}"

    print(f"Results: {out_dir}", flush=True)
    ids_per_frame, ms, key_ms = [], [], []
    n = 0
    started = time.perf_counter()
    try:
        with ExitStack() as cleanup:
            cap = cv2.VideoCapture(str(video))
            cleanup.callback(cap.release)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            metadata["source_fps"] = fps
            tracker = _load_backend(args.backend, args)
            session = tracker.start_stream(args.prompt)
            if hasattr(session, "reset_inference_session"):
                cleanup.callback(session.reset_inference_session)
            writer = open_writer(f"{stem}.mp4", fps)
            cleanup.callback(writer.close)
            f = cleanup.enter_context(open(f"{stem}.csv", "w", newline=""))
            log = csv.writer(f)
            log.writerow(["frame", "ms", "ids"])
            while not args.max_frames or n < args.max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                instances, t = tracker.track_frame(session, rgb)
                ids = sorted(i.obj_id for i in instances if i.obj_id is not None)
                ids_per_frame.append(ids)
                ms.append(t)
                if hybrid and n % args.keyframe_every == 0:
                    key_ms.append(session["last_keyframe_ms"])
                log.writerow([n, f"{t:.1f}", " ".join(map(str, ids))])
                f.flush()  # preserve completed requests if the next one stalls
                n += 1
                writer.append_data(np.asarray(draw_instances(Image.fromarray(rgb), instances)))
                if (n - 1) % 50 == 0:
                    print(f"frame {n - 1:5d}  {t:6.1f} ms  ids={ids}")
            if not n:
                raise ValueError("Video contained no decodable frames")
        metadata["status"] = "completed"
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata.update(frames=n, total_seconds=time.perf_counter() - started)
        metadata["total_timing"] = "Includes model/session startup, decoding, drawing, encoding and cleanup; excludes manifest and timeline."
        if ms:
            measured = ms[10:] or ms
            metadata["requests"] = {
                "warmup_frames_excluded": 10 if len(ms) > 10 else 0,
                "mean_ms": float(np.mean(measured)),
                "p50_ms": float(np.median(measured)),
                "p95_ms": float(np.percentile(measured, 95)),
            }
        (out_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _timeline_png(ids_per_frame, Path(f"{stem}_ids.png"))

    steady = ms[10:] or ms
    distinct = sorted({i for ids in ids_per_frame for i in ids})
    hit = sum(1 for ids in ids_per_frame if ids)
    print(f"\n{args.backend}: {n} frames, median {np.median(steady):.1f} ms "
          f"({1000 / np.mean(steady):.1f} frame requests/sec; p95 {np.percentile(steady, 95):.1f} ms)")
    if key_ms:
        gap = [t for k, t in enumerate(ms) if k % args.keyframe_every][10:]
        print(f"  keyframes: median {np.median(key_ms[1:] or key_ms):.0f} ms (SAM3 + seeding)")
        if gap:
            print(f"  gap frames: median {np.median(gap):.1f} ms")
    print(f"  total processing: {metadata['total_seconds']:.1f}s ({n / metadata['total_seconds']:.2f} fps)")
    print(f"  frames with output: {hit}/{n}; distinct IDs: {len(distinct)} {distinct}")
    print(f"  wrote {stem}.mp4, {stem}.csv, {stem}_ids.png")


if __name__ == "__main__":
    main()
