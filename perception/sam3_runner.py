"""SAM 3 inference wrapper.

Loads the model once and exposes segment() for text-prompted concept
segmentation. Returns plain numpy/python results so callers don't need
to touch torch or transformers.
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
    obj_id: int | None = None  # stable track id; set by video tracking only


@dataclass
class SegmentResult:
    instances: list[Instance]
    inference_ms: float


class Sam3Runner:
    def __init__(self, model_id: str = MODEL_ID):
        from transformers import Sam3Model, Sam3Processor

        t0 = time.perf_counter()
        self.model = Sam3Model.from_pretrained(model_id, device_map="auto")
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
