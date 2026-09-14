"""DARTF FP16 TensorRT native tracking worker, one process per webcam stream.

Stdio carries uint32 little-endian lengths followed by NumPy NPZ archives.
Startup emits ``ready=True``; requests contain uint8 RGB ``frame``; replies
contain ``masks``, ``boxes``, ``scores``, ``ids``, and ``inference_ms``.
All diagnostics, including native library output, go to stderr.

Inference follows DART's dartf/demo/run_video.py native mode, using its
importable runtime and tracker. Engines are built locally; there is no fallback.
"""

import argparse
import io
import os
from pathlib import Path
import struct
import sys
import time
import zipfile

import numpy as np


MAX_PACKET_BYTES = 64 * 1024 * 1024


def _read_exact(stream, count, *, allow_eof=False):
    data = bytearray()
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            if allow_eof and not data:
                return None
            raise EOFError("Truncated DARTF packet")
        data.extend(chunk)
    return bytes(data)


def read_packet(stream):
    """Read one NPZ packet; return None only at a clean stream EOF."""
    header = _read_exact(stream, 4, allow_eof=True)
    if header is None:
        return None
    length = struct.unpack("<I", header)[0]
    if not 0 < length <= MAX_PACKET_BYTES:
        raise ValueError(f"Invalid DARTF packet length: {length}")
    payload = _read_exact(stream, length)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (ValueError, OSError, EOFError, zipfile.BadZipFile, TypeError) as error:
        raise ValueError("Invalid DARTF NPZ packet") from error


def write_packet(stream, **arrays):
    """Write and flush one NPZ packet, including on streams with short writes."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"DARTF packet exceeds {MAX_PACKET_BYTES} bytes")
    pending = memoryview(struct.pack("<I", len(payload)) + payload)
    while pending:
        written = stream.write(pending)
        if not written:
            raise OSError("DARTF stream stopped accepting data")
        pending = pending[written:]
    stream.flush()


class DartfTracker:
    """Single-prompt FP16 detector and SAM3 memory propagation on TensorRT."""

    def __init__(self, dart_root, assets, prompt):
        assets = Path(assets)
        required = (
            "vision_fp16.plan", "ground_c1m_fp16.plan",
            "maskhead_q32_phase_fp16.plan", "img_pos_c1.npy",
            "text_c16.onnx", "bpe_simple_vocab_16e6.txt.gz",
            "trk_neck_fp16.plan", "trk_init_fp16.plan",
            "trk_step_v2_fp16.plan", "pe_mem.npy", "tpos_enc.npy",
        )
        missing = [name for name in required if not (assets / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing DARTF assets: " + ", ".join(missing))
        if not prompt.strip():
            raise ValueError("DARTF requires a non-empty text prompt")
        dartf = Path(dart_root) / "dartf"
        sys.path[:0] = [str(dartf / "demo"), str(dartf / "runtime")]
        import torch
        import torch.nn.functional as functional
        import onnxruntime as ort
        from scipy.optimize import linear_sum_assignment
        from trt_util import Runner, load_engine
        from sam3_track import DynRunner, Sam3Tracker
        from tokenize_classes import Sam3Tokenizer, BUCKET, CONTEXT_LENGTH

        self.torch = torch
        self.functional = functional
        self.assign = linear_sum_assignment
        self.frame_idx = 0
        self.next_id = 1
        self.tracks = {}

        tokenizer = Sam3Tokenizer(assets / "bpe_simple_vocab_16e6.txt.gz")
        token_ids = np.zeros((BUCKET, CONTEXT_LENGTH), dtype=np.int32)
        token_ids[:, 0] = tokenizer.start_id
        token_ids[:, 1] = tokenizer.end_id
        encoded = tokenizer.encode(prompt)
        token_ids[0] = 0
        token_ids[0, :len(encoded)] = encoded
        text_model = ort.InferenceSession(str(assets / "text_c16.onnx"),
                                         providers=["CPUExecutionProvider"])
        text_features, text_mask = text_model.run(None, {"token_ids": token_ids})
        self.text_features = np.ascontiguousarray(text_features[:, :1], dtype=np.float16)
        self.text_mask = np.ascontiguousarray(text_mask[:1], dtype=np.float32)
        self.img_pos = np.load(assets / "img_pos_c1.npy", allow_pickle=False)
        del text_model

        self.vision = Runner(load_engine(str(assets / "vision_fp16.plan")))
        expected = {"tracker_trunk": (1, 1024, 72, 72),
                    "fpn_0": (1, 256, 288, 288),
                    "fpn_1": (1, 256, 144, 144), "fpn_2": (1, 256, 72, 72)}
        for name, shape in expected.items():
            if name not in self.vision.bufs or tuple(self.vision.bufs[name][1]) != shape:
                raise ValueError(f"DARTF vision engine must expose {name} with shape {shape}")
        self.ground = Runner(load_engine(str(assets / "ground_c1m_fp16.plan")))
        if self.ground.bufs["img_feat"][1][0] != 1:
            raise ValueError("DARTF worker requires a single-prompt grounding engine")
        self.mask = DynRunner(load_engine(str(assets / "maskhead_q32_phase_fp16.plan")))
        for name in ("fpn_0", "fpn_1"):
            buffer, shape, _ = self.vision.bufs[name]
            self.mask.bind(name, buffer.ptr, shape)
        self.tracker = Sam3Tracker(
            str(assets / "trk_neck_fp16.plan"), str(assets / "trk_init_fp16.plan"),
            str(assets / "trk_step_v2_fp16.plan"), self.vision, "tracker_trunk",
            pe_mem=np.load(assets / "pe_mem.npy", allow_pickle=False),
            tpos_enc=np.load(assets / "tpos_enc.npy", allow_pickle=False),
            prune_r=0,
        )

    def _overlap(self, masks_a, masks_b):
        a = (self.torch.stack(masks_a) > 0).float().flatten(1)
        b = (self.torch.stack(masks_b) > 0).float().flatten(1)
        intersection = a @ b.T
        area_a, area_b = a.sum(1), b.sum(1)
        return intersection, area_a, area_b

    def _detect(self, frame):
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((1008, 1008), Image.Resampling.BILINEAR))
        pixels = resized.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
        vision = self.vision({"images": pixels[None]}, want=["fpn_2"])
        ground = self.ground({
            "img_feat": vision["fpn_2"].astype(np.float16), "img_pos": self.img_pos,
            "text_feats": self.text_features, "text_mask": self.text_mask,
        }, want=["scores", "presence", "hs", "enc"])
        sigmoid = lambda value: 1.0 / (1.0 + np.exp(-np.clip(value.astype(np.float64), -80, 80)))
        scores = sigmoid(ground["scores"][0, :, 0]) * sigmoid(ground["presence"][0, 0])
        selected = np.argsort(-scores)[:32]
        confident = np.flatnonzero(scores[selected] >= 0.5)
        if not len(confident):
            return []
        masks = self.mask({
            "enc": np.ascontiguousarray(ground["enc"][:, :1]),
            "text_feats": self.text_features, "text_mask": self.text_mask,
            "hs_sel": np.ascontiguousarray(ground["hs"][:1, selected], dtype=np.float16),
        }, numpy=False)["masks"][0, confident].float()
        scores = scores[selected[confident]]
        # Match upstream mask suppression: remove groups, duplicates, and nested parts.
        inter, area, _ = self._overlap(list(masks), list(masks))
        iou = (inter / (area[:, None] + area[None, :] - inter + 1e-6)).cpu().numpy()
        containment = (inter / (self.torch.minimum(area[:, None], area[None, :]) + 1e-6)).cpu().numpy()
        inside = (inter / (area[:, None] + 1e-6)).cpu().numpy()
        area = area.cpu().numpy()
        groups = {j for j in range(len(masks)) if sum(
            i != j and inside[i, j] > 0.7 and area[i] < area[j] * 0.5
            for i in range(len(masks))) >= 2}
        kept = []
        for i in range(len(masks)):
            if area[i] == 0 or i in groups:
                continue
            if any(iou[i, j] > 0.6 or containment[i, j] > 0.85 for j in kept):
                continue
            kept.append(i)
        return [(masks[i].clone(), float(scores[i])) for i in kept]

    def _track(self, detections):
        self.tracker.run_neck()
        ids = sorted(self.tracks)
        propagated = self.tracker.propagate(self.frame_idx, ids) if ids else {}
        for oid, (mask, score, _) in propagated.items():
            self.tracks[oid]["mask"] = mask
            self.tracks[oid]["score"] = float(1 / (1 + np.exp(-np.clip(score, -80, 80))))
        pairs = []
        if ids and detections:
            inter, area_a, area_b = self._overlap(
                [self.tracks[oid]["mask"] for oid in ids], [mask for mask, _ in detections])
            iou = (inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)).cpu().numpy()
            rows, cols = self.assign(1 - iou)
            pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] >= 0.5]
        matched_ids = {ids[i] for i, _ in pairs}
        matched_detections = {j for _, j in pairs}
        refresh_ids, refresh_masks = [], []
        for i, j in pairs:
            oid = ids[i]
            self.tracks[oid]["hits"] += 1
            self.tracks[oid]["misses"] = 0
            if self.frame_idx - self.tracker.obj(oid).last_init >= 6:
                refresh_ids.append(oid)
                refresh_masks.append(detections[j][0])
        for oid in ids:
            if oid not in matched_ids:
                self.tracks[oid]["misses"] += 1
                if self.tracks[oid]["misses"] > 8:
                    self.tracker.drop(oid)
                    del self.tracks[oid]
        for j, (mask, score) in enumerate(detections):
            if j not in matched_detections:
                oid = self.next_id
                self.next_id += 1
                self.tracks[oid] = {"mask": mask, "score": score, "hits": 1, "misses": 0}
                refresh_ids.append(oid)
                refresh_masks.append(mask)
        if refresh_ids:
            masks1008 = self.functional.interpolate(
                self.torch.stack(refresh_masks)[:, None], size=(1008, 1008),
                mode="bilinear", align_corners=False)[:, 0] > 0
            self.tracker.refresh(self.frame_idx, refresh_ids, masks1008)
        self.tracker.prune(self.tracks.keys())

    def track_frame(self, frame):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not all(frame.shape[:2]):
            raise ValueError("DARTF frame must be a non-empty uint8 RGB HxWx3 array")
        started = time.perf_counter()
        height, width = frame.shape[:2]
        with self.torch.inference_mode():
            self._track(self._detect(frame))
            shown = [(oid, track) for oid, track in self.tracks.items()
                     if track["hits"] >= 3 and track["score"] > 0.5]
            masks, boxes, scores, ids = [], [], [], []
            if shown:
                resized = self.functional.interpolate(
                    self.torch.stack([track["mask"] for _, track in shown])[:, None],
                    size=(height, width), mode="bilinear", align_corners=False)[:, 0] > 0
                for (oid, track), mask in zip(shown, resized.cpu().numpy()):
                    ys, xs = np.nonzero(mask)
                    if not len(xs):
                        continue
                    masks.append(mask)
                    boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
                    scores.append(track["score"])
                    ids.append(oid)
        self.torch.cuda.synchronize()
        self.frame_idx += 1
        return {
            "masks": np.asarray(masks, dtype=bool).reshape(-1, height, width),
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "scores": np.asarray(scores, dtype=np.float32),
            "ids": np.asarray(ids, dtype=np.int64),
            "inference_ms": np.asarray((time.perf_counter() - started) * 1000, dtype=np.float64),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dart-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    # Reserve the protocol pipe, then redirect fd 1 too: TRT can print from C++.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    with output:
        tracker = DartfTracker(args.dart_root, args.assets, args.prompt)
        write_packet(output, ready=np.asarray(True))
        while (packet := read_packet(sys.stdin.buffer)) is not None:
            if set(packet) != {"frame"}:
                raise ValueError("DARTF request must contain exactly one 'frame' array")
            write_packet(output, **tracker.track_frame(packet["frame"]))


if __name__ == "__main__":
    main()
