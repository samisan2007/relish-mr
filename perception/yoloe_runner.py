"""YOLOE video tracking: a fast alternative/complement to Sam3VideoTracker.

Two backends, both exposing the same start_stream(text)/track_frame(session, frame)
shape as Sam3VideoTracker so the webcam UI can swap between them without special-casing:

- YoloeVideoTracker: pure text-prompt mode (YOLOE's own open-vocabulary text encoder,
  a lightweight MobileCLIP model). Earlier runs found low confidence on specific
  food nouns, but those measurements predate the RGB/BGR fix and need retesting.- HybridVideoTracker: uses SAM3 to find every instance in a seed frame, then
  extracts YOLOE visual embeddings from that frame once. Later frames reuse
  those embeddings with persist=True for tracking. Motion quality needs live
  retesting after correcting the original color and reference-frame bugs.
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
        """Track an RGB frame; Ultralytics expects BGR for NumPy inputs."""
        t0 = time.perf_counter()
        results = self.model.track(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            persist=True, conf=self.conf, verbose=False, device=self.device_arg
        )
        ms = (time.perf_counter() - t0) * 1000
        return _to_instances(results[0], frame.shape[:2]), ms


class HybridVideoTracker:
    """Ground the prompt once with SAM3 (reliable on niche nouns), then track the
    resulting exemplars with YOLOE's persistent tracker.
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
            "reference_image": None,
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
                # Keep the exact image these boxes describe. cvtColor also copies
                # it, so a caller reusing its RGB capture buffer cannot alter it.
                session["reference_image"] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                session["seeded"] = True
            return instances, ms

        t0 = time.perf_counter()
        prompt_kwargs = {}
        if session["reference_image"] is not None:
            # refer_image installs persistent class embeddings. Passing the boxes
            # again on later frames would instead sample their old coordinates.
            prompt_kwargs = dict(
                refer_image=session["reference_image"],
                visual_prompts=session["visual_prompts"],
                predictor=self.vp_predictor,
            )
        results = self.model.track(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            persist=True,
            conf=self.conf,
            verbose=False,
            device=self.device_arg,
            **prompt_kwargs,
        )
        # Release only after a successful handoff, so a failed call can retry.
        session["reference_image"] = None
        session["visual_prompts"] = None
        ms = (time.perf_counter() - t0) * 1000
        return _to_instances(results[0], frame.shape[:2]), ms
