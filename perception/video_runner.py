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


def _to_instances(processed: dict) -> list[Instance]:
    """Processor output -> Instance list, keeping the track id."""
    return [
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


def open_writer(path: str | Path, fps: float):
    """H.264 writer. OpenCV's mp4v output won't play in a browser; ffmpeg's H.264 will."""
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                              macro_block_size=8)


class Sam3VideoTracker:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        dtype: "torch.dtype | None" = None,
        processor_size: int | None = None,
        use_device_map: bool = True,
        compile_model: bool = False,
    ):
        """
        dtype: autocast precision (e.g. torch.bfloat16). None runs full fp32, no autocast.
            Applied via torch.autocast around each forward pass rather than converting the
            model's stored weights — the video session creates some tensors (memory-bank
            slots, object queries) as plain fp32 tensors outside the parameter tree, so a
            wholesale `.to(dtype)` leaves those mismatched against cast weights the moment a
            track is actually created ("Input type (float) and bias type (struct c10::Half)
            should be the same"). Autocast casts op-by-op instead, so those fresh fp32
            tensors stay compatible.
        processor_size: override the fixed square resize (default 1008) the processor applies
            to every frame before it hits the model. NOTE: currently broken once a track is
            created — the video session's memory-bank/position-embedding buffers are sized
            for the default 72x72 (1008px / 14) token grid regardless of this override, so a
            real detection throws a tensor-size mismatch (1296 or 400 vs 5184). Left in for
            experimentation but treat as unsupported until that's fixed upstream.
        use_device_map: False does a plain `.to("cuda")` instead of `device_map="auto"`,
            skipping accelerate's dispatch hooks (relevant for single-GPU perf testing).
        compile_model: wraps the model in torch.compile. Experimental — the stateful video
            session can trigger recompiles per frame, so this may not help.
        """
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        t0 = time.perf_counter()
        model_kwargs = {}
        if use_device_map:
            model_kwargs["device_map"] = "auto"
        self.model = Sam3VideoModel.from_pretrained(model_id, **model_kwargs)
        if not use_device_map:
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = dtype

        processor_kwargs = {}
        if processor_size is not None:
            processor_kwargs["size"] = {"height": processor_size, "width": processor_size}
        self.processor = Sam3VideoProcessor.from_pretrained(model_id, **processor_kwargs)

        if compile_model:
            self.model = torch.compile(self.model)
        self.load_seconds = time.perf_counter() - t0

    def _autocast(self):
        return torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        )

    @property
    def device(self) -> str:
        return str(self.model.device)

    def start_stream(self, text: str):
        """Open a session for live frames (webcam). Returns the session to feed track_frame().

        Streaming disables the hotstart heuristics that prune duplicate and
        unmatched tracks, since those need future frames — expect more false
        positives than the offline `track()` path.
        """
        session = self.processor.init_video_session(
            inference_device=self.model.device,
            processing_device="cpu",
            video_storage_device="cpu",
        )
        self.processor.add_text_prompt(session, text)
        return session

    def track_frame(self, session, frame: np.ndarray) -> tuple[list[Instance], float]:
        """Track one live RGB frame. Identities carry over from earlier frames."""
        t0 = time.perf_counter()
        inputs = self.processor(images=frame, device=self.model.device, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad(), self._autocast():
            outputs = self.model(inference_session=session, frame=inputs.pixel_values[0])
        processed = self.processor.postprocess_outputs(
            session, outputs, original_sizes=inputs.original_sizes
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return _to_instances(processed), (time.perf_counter() - t0) * 1000

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

        with self._autocast():
            for outputs in self.model.propagate_in_video_iterator(inference_session=session):
                t0 = time.perf_counter()
                processed = self.processor.postprocess_outputs(session, outputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - t0) * 1000

                yield FrameResult(
                    frame_idx=outputs.frame_idx,
                    image=Image.fromarray(frames[outputs.frame_idx]),
                    instances=_to_instances(processed),
                    inference_ms=inference_ms,
                )
