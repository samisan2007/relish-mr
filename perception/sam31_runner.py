"""Official SAM 3.1 in Docker, using the existing frame transport and UI results."""

from pathlib import Path

import numpy as np

from dartf_runner import DartfSession, WorkerTracker, track_worker_frame
from sam3_runner import SegmentResult

MODEL_SAM31 = "SAM 3.1"
MODEL_SAM31_COMPILED = "SAM 3.1 (compiled)"
SAM31_CHOICES = [MODEL_SAM31, MODEL_SAM31_COMPILED]
COMPILE_CACHE_VOLUME = "relish-sam31-compile-cache"


class Sam31Session(DartfSession):
    label = "SAM 3.1"

    def __init__(self, prompt, compile_model=False, image_only=False, threshold=0.5, on_started=None):
        root = Path(__file__).resolve().parent
        local = root / "sam31-local"
        if not (local / "source" / "sam3" / "model_builder.py").is_file():
            raise RuntimeError("SAM 3.1 is not set up. Run perception\\sam31\\run.ps1 -Action Setup.")
        self.frame_timeout = 1200 if compile_model else 120
        args = [
            "--mount", f"type=bind,source={root},target=/app,readonly",
            "--mount", f"type=bind,source={local},target=/local",
            "--mount", f"type=bind,source={local / 'source'},target=/sam31,readonly",
            # Inductor's compile workers write generated kernels and reopen them by path.
            # A Windows bind mount does not make those writes visible to the other
            # processes in time, so torch.compile fails with FileNotFoundError on a file
            # that is present on disk. Keep the compile caches on a Docker volume.
            "--mount", f"type=volume,source={COMPILE_CACHE_VOLUME},target=/compile-cache",
            "--env", "PYTHONPATH=/sam31:/app", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "HF_HOME=/local/hf", "--env", "HF_HUB_OFFLINE=1",
            "--env", "OMP_NUM_THREADS=8", "--env", "TORCHINDUCTOR_CACHE_DIR=/compile-cache/inductor",
            "--env", "TRITON_CACHE_DIR=/compile-cache/triton",
            "--entrypoint", "python", "relish-sam31:local", "/app/sam31/worker.py",
            "--prompt", prompt, "--threshold", str(threshold),
        ]
        if compile_model:
            args.append("--compile")
        if image_only:
            args.append("--image-only")
        super().__init__(local, local, prompt, worker_args=args, name_prefix="relish-sam31-ui-", on_started=on_started)


class Sam31Runner(WorkerTracker):
    def __init__(self, compile_model=False):
        self.compile_model = compile_model

    def start_stream(self, text, on_started=None):
        return Sam31Session(text, self.compile_model, on_started=on_started)

    def track_frame(self, session, frame):
        return track_worker_frame(session, frame)

    def segment(self, image, text, threshold=0.5):
        session = Sam31Session(text, self.compile_model, image_only=True, threshold=threshold)
        try:
            instances, ms = self.track_frame(session, np.asarray(image.convert("RGB")))
            return SegmentResult([i for i in instances if i.score >= threshold], ms)
        finally:
            session.reset_inference_session()
