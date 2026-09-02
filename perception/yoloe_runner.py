"""YOLOE video tracking: a fast alternative/complement to Sam3VideoTracker.

Two backends, both exposing the same start_stream(text)/track_frame(session, frame)
shape as Sam3VideoTracker so the webcam UI can swap between them without special-casing:

- YoloeVideoTracker: pure text-prompt mode (YOLOE's own open-vocabulary text encoder,
  a lightweight MobileCLIP model). Fast (~100-500ms/frame), but its vocabulary is
  noticeably weaker than SAM3's on specific food nouns — e.g. "meatball" tops out
  around 0.17 confidence even on the largest checkpoint, vs. SAM3 finding it cleanly.
- HybridVideoTracker: uses a Sam3VideoTracker once to ground the prompt (slow but
  reliable), then hands that box to YOLOE as a *visual* exemplar for every later
  frame (YOLOE's visual-prompt mode is excellent — 0.9+ confidence on the same
  objects text-prompting missed) with persist=True for stable track IDs. Matches
  the actual need: understand what "meatball" means once, then re-localize it
  fast as the HMD moves, without re-doing that semantic understanding every frame.
"""

import time

import cv2
import numpy as np
import torch

from sam3_runner import Instance

YOLOE_MODEL_ID = "yoloe-11l-seg.pt"


def _to_instances(result, orig_hw: tuple[int, int]) -> list[Instance]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    height, width = orig_hw
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)

    instances = []
    for i in range(len(boxes)):
        if masks is not None:
            mask = cv2.resize(
                masks[i].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask = np.zeros((height, width), dtype=bool)
        instances.append(Instance(
            mask=mask,
            box_xyxy=tuple(float(v) for v in boxes.xyxy[i].tolist()),
            score=float(boxes.conf[i]),
            obj_id=int(ids[i]) if ids[i] is not None else None,
        ))
    return instances


class YoloeVideoTracker:
    """Text-prompt mode only. Same call shape as Sam3VideoTracker for the webcam UI."""

    def __init__(self, model_id: str = YOLOE_MODEL_ID, conf: float = 0.1):
        from ultralytics import YOLOE

        t0 = time.perf_counter()
        self.model = YOLOE(model_id)
        self.conf = conf
        self.device_arg = 0 if torch.cuda.is_available() else "cpu"
        self.load_seconds = time.perf_counter() - t0

    @property
    def device(self) -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def start_stream(self, text: str):
        self.model.set_classes([text], self.model.get_text_pe([text]))
        return {"prompt": text}

    def track_frame(self, session, frame: np.ndarray) -> tuple[list[Instance], float]:
        t0 = time.perf_counter()
        results = self.model.track(
            frame, persist=True, conf=self.conf, verbose=False, device=self.device_arg
        )
        ms = (time.perf_counter() - t0) * 1000
        return _to_instances(results[0], frame.shape[:2]), ms


class HybridVideoTracker:
    """Ground the prompt once with SAM3 (reliable on niche nouns), then track the
    resulting box with YOLOE's visual-exemplar mode + ByteTrack (fast, stable IDs).
    Re-attempts grounding with SAM3 on every frame until something is found, then
    switches over permanently for the rest of the stream.
    """

    def __init__(self, sam3_tracker, model_id: str = YOLOE_MODEL_ID, conf: float = 0.15):
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

        self.sam3_tracker = sam3_tracker  # shared, already-loaded — grounding is a brief cameo, not a steady-state cost
        self.model = YOLOE(model_id)
        self.vp_predictor = YOLOEVPSegPredictor
        self.conf = conf
        self.device_arg = 0 if torch.cuda.is_available() else "cpu"
        self.load_seconds = 0.0

    @property
    def device(self) -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def start_stream(self, text: str):
        return {
            "prompt": text,
            "sam3_session": self.sam3_tracker.start_stream(text),
            "seeded": False,
            "visual_prompts": None,
        }

    def track_frame(self, session, frame: np.ndarray) -> tuple[list[Instance], float]:
        if not session["seeded"]:
            instances, ms = self.sam3_tracker.track_frame(session["sam3_session"], frame)
            if instances:
                # One exemplar generalizes poorly to the rest of the class (observed: a
                # single box found 1/13 meatballs vs. 10-13/13 with several) — seed with
                # every instance SAM3 found this frame, not just the top-scoring one.
                session["visual_prompts"] = dict(
                    bboxes=np.array([list(i.box_xyxy) for i in instances]),
                    cls=np.zeros(len(instances), dtype=int),
                )
                session["seeded"] = True
            return instances, ms

        t0 = time.perf_counter()
        results = self.model.track(
            frame,
            visual_prompts=session["visual_prompts"],
            predictor=self.vp_predictor,
            persist=True,
            conf=self.conf,
            verbose=False,
            device=self.device_arg,
        )
        ms = (time.perf_counter() - t0) * 1000
        return _to_instances(results[0], frame.shape[:2]), ms
