"""SAM 3 inference wrapper.

Loads the model once and exposes segment() for text-prompted concept
segmentation. Returns plain numpy/python results so callers don't need
to touch torch or transformers.

Sam3ImageTracker at the bottom turns that stateless per-frame detector into a
tracker by handing the boxes to ByteTrack, as an alternative to Sam3VideoTracker's
memory bank (same call shape, so the webcam UI can swap between them).
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

MODEL_ID = "facebook/sam3"


@dataclass
class Instance:
    mask: np.ndarray  # bool, HxW, original image size
    box_xyxy: tuple[float, float, float, float]
    score: float
    obj_id: int | None = None  # stable track id; set by video tracking or Sam3ImageTracker


@dataclass
class SegmentResult:
    instances: list[Instance]
    inference_ms: float


class Sam3Runner:
    def __init__(self, model_id: str = MODEL_ID, dtype: "torch.dtype | None" = None):
        """dtype casts the weights on load. Safe here because the image model is
        stateless — the video session's fp32 memory-bank hazard doesn't apply — and
        fp16 halves the ~3.4GB footprint, which matters when this sits alongside the
        video model. Defaults to fp32 so the image tester is unaffected."""
        from transformers import Sam3Model, Sam3Processor

        t0 = time.perf_counter()
        kwargs = {"device_map": "auto"}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.model = Sam3Model.from_pretrained(model_id, **kwargs)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.load_seconds = time.perf_counter() - t0

    @property
    def device(self) -> str:
        return str(self.model.device)

    def segment(self, image, text: str, threshold: float = 0.5) -> SegmentResult:
        """Segment every instance of the concept `text` in a PIL image."""
        t0 = time.perf_counter()
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - t0) * 1000

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        instances = [
            Instance(
                mask=mask.cpu().numpy().astype(bool),
                box_xyxy=tuple(float(v) for v in box),
                score=float(score),
            )
            for mask, box, score in zip(results["masks"], results["boxes"], results["scores"])
        ]
        return SegmentResult(instances=instances, inference_ms=inference_ms)


class Sam3ImageTracker:
    """SAM3's image detector every frame + ByteTrack for the IDs.

    Sam3VideoTracker's memory bank costs ~1s/frame; this drops it and re-detects
    instead, associating frame to frame by IoU. Keeps SAM3's vocabulary (the reason
    YOLOE was unusable) and its masks, loses the temporal memory — so expect ID
    swaps between identical adjacent objects and a new ID after a full occlusion.

    Same start_stream/track_frame shape as Sam3VideoTracker, so the webcam UI
    swaps it in without special-casing.
    """

    def __init__(self, sam3: "Sam3Runner", threshold: float = 0.3, fp16: bool = True):
        self.sam3 = sam3  # shared, already-loaded
        self.threshold = threshold
        # The image model is stateless, so autocast is safe here — unlike the video
        # session, whose memory bank allocates fp32 tensors outside the parameter tree.
        self.autocast_dtype = torch.float16 if fp16 else None
        self.load_seconds = 0.0

    @property
    def device(self) -> str:
        return self.sam3.device

    def start_stream(self, text: str):
        """Fresh ByteTracker per stream, so IDs restart at 1 on every run."""
        from ultralytics.trackers import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        return {"prompt": text, "tracker": BYTETracker(cfg)}

    def track_frame(self, session, frame: np.ndarray) -> tuple[list[Instance], float]:
        """Detect in an RGB frame, then assign track ids. Timing covers the whole
        step, not just the forward pass, so it compares fairly against the others."""
        from PIL import Image
        from ultralytics.engine.results import Boxes

        t0 = time.perf_counter()
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            result = self.sam3.segment(Image.fromarray(frame), session["prompt"], self.threshold)
        instances = result.instances

        # ByteTrack reads .xywh/.conf/.cls off a results-like object; Boxes is one.
        # Single class, so cls is all zeros. Empty frames still go through, otherwise
        # lost tracks never age out of the buffer.
        data = np.array(
            [[*inst.box_xyxy, inst.score, 0.0] for inst in instances], dtype=np.float32
        ).reshape(-1, 6)
        tracked = session["tracker"].update(Boxes(data, frame.shape[:2]))

        # Rows are [x1, y1, x2, y2, track_id, score, cls, idx] — idx indexes back into
        # what we passed in, which is how a track id finds its mask again. Detections
        # ByteTrack didn't confirm keep obj_id=None; the UI already tolerates that.
        for row in tracked:
            instances[int(row[7])].obj_id = int(row[4])

        return instances, (time.perf_counter() - t0) * 1000
