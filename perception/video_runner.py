"""SAM 3 video tracking: text prompt -> masks with stable object IDs across frames.

Separate from sam3_runner.py because it's a different model (Sam3VideoModel) with
a stateful inference session, not a per-image forward pass.

Frames are decoded with a stride so a long clip can be sampled cheaply, and
results are yielded per frame so memory stays flat over a long video.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

from sam3_runner import Instance

MODEL_ID = "facebook/sam3"


@dataclass
class FrameResult:
    frame_idx: int          # index into the sampled frames, not the source video
    image: Image.Image
    instances: list[Instance]
    inference_ms: float


def load_frames(video_path: str | Path, max_frames: int = 150, stride: int = 3) -> tuple[list[np.ndarray], float]:
    """Decode every `stride`-th frame, up to `max_frames`. Returns RGB frames + output fps."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    frames, idx = [], 0
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, src_fps / stride


def open_writer(path: str | Path, fps: float):
    """H.264 writer. OpenCV's mp4v output won't play in a browser; ffmpeg's H.264 will."""
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                              macro_block_size=8)


class Sam3VideoTracker:
    def __init__(self, model_id: str = MODEL_ID):
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        t0 = time.perf_counter()
        self.model = Sam3VideoModel.from_pretrained(model_id, device_map="auto")
        self.processor = Sam3VideoProcessor.from_pretrained(model_id)
        self.load_seconds = time.perf_counter() - t0

    @property
    def device(self) -> str:
        return str(self.model.device)

    def track(self, frames: list[np.ndarray], text: str):
        """Yield a FrameResult per frame. Object IDs are stable across the clip.

        The video is kept on CPU and only the frame being processed moves to the
        GPU, so clip length is bounded by RAM rather than VRAM.
        """
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.model.device,
            processing_device="cpu",
            video_storage_device="cpu",
        )
        self.processor.add_text_prompt(session, text)

        for outputs in self.model.propagate_in_video_iterator(inference_session=session):
            t0 = time.perf_counter()
            processed = self.processor.postprocess_outputs(session, outputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - t0) * 1000

            instances = [
                Instance(
                    mask=mask.cpu().numpy().astype(bool),
                    box_xyxy=tuple(float(v) for v in box),
                    score=float(score),
                    obj_id=int(obj_id),
                )
                for mask, box, score, obj_id in zip(
                    processed["masks"], processed["boxes"],
                    processed["scores"], processed["object_ids"],
                )
            ]
            yield FrameResult(
                frame_idx=outputs.frame_idx,
                image=Image.fromarray(frames[outputs.frame_idx]),
                instances=instances,
                inference_ms=inference_ms,
            )
