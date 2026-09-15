"""DARTF in a GPU Docker process; the webcam and its dependencies stay on Windows."""

import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np

from PIL import Image

from dartf_worker import read_packet, write_packet
from sam3_runner import Instance
from video_runner import FrameResult


class WorkerTracker:
    """Adapts a start_stream/track_frame worker into the frame generator the file
    tab and track_video expect. Both Docker backends feed frames one at a time and
    differ only in which session they open, so the loop is shared."""

    def track(self, frames, text):
        session = self.start_stream(text)
        try:
            for idx, frame in enumerate(frames):
                instances, ms = self.track_frame(session, frame)
                yield FrameResult(idx, Image.fromarray(frame), instances, ms)
        finally:
            session.reset_inference_session()


class DartfVideoTracker(WorkerTracker):
    def __init__(self):
        self.root = Path(os.environ.get(
            "DART_ROOT", Path(__file__).resolve().parents[2] / "DART"
        )).resolve()
        self.assets = Path(os.environ.get(
            "DARTF_ASSETS", self.root / "dartf" / "assets"
        )).resolve()
        required = (
            "vision_fp16.plan", "ground_c1m_fp16.plan",
            "maskhead_q32_phase_fp16.plan", "img_pos_c1.npy",
            "text_c16.onnx", "bpe_simple_vocab_16e6.txt.gz",
            "trk_neck_fp16.plan", "trk_init_fp16.plan",
            "trk_step_v2_fp16.plan", "pe_mem.npy", "tpos_enc.npy",
        )
        missing = [name for name in required if not (self.assets / name).is_file()]
        if missing:
            raise RuntimeError(
                f"DARTF engines are not ready in {self.assets}: {', '.join(missing)}. "
                "See perception/dartf/README.md for setup."
            )

    def start_stream(self, text):
        return DartfSession(self.root, self.assets, text)

    def track_frame(self, session, frame):
        return track_worker_frame(session, frame)


def track_worker_frame(session, frame):
    started = time.perf_counter()
    result = session.request(frame)
    masks, boxes, scores, ids = (result[k] for k in ("masks", "boxes", "scores", "ids"))
    if (ids.ndim != 1 or masks.shape != (len(ids), *frame.shape[:2])
            or boxes.shape != (len(ids), 4) or scores.shape != (len(ids),)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
        raise RuntimeError("GPU worker returned inconsistent masks, boxes, scores or IDs")
    instances = [
        Instance(mask, tuple(float(v) for v in box), float(score), int(oid))
        for mask, box, score, oid in zip(masks, boxes, scores, ids)
    ]
    # Include transfer and conversion overhead in the webcam comparison.
    return instances, (time.perf_counter() - started) * 1000


class DartfSession:
    label = "DARTF"
    frame_timeout = 60

    def __init__(self, root, assets, prompt, *, worker_args=None, name_prefix="relish-dartf-", on_started=None):
        self.name = name_prefix + uuid.uuid4().hex[:12]
        self.log = tempfile.TemporaryFile()
        self.process = None
        self.failed = False
        self.results = queue.Queue()
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        worker_args = worker_args if worker_args is not None else [
            "--mount", f"type=bind,source={root},target=/dart,readonly",
            "--mount", f"type=bind,source={assets},target=/assets,readonly",
            "--mount", f"type=bind,source={Path(__file__).resolve().parent},target=/app,readonly",
            "--entrypoint", "python", "relish-dartf:local",
            "/app/dartf_worker.py", "--dart-root", "/dart", "--assets", "/assets",
            "--prompt", prompt,
        ]
        command = ["docker", "run", "--rm", "-i", "--gpus", "all", "--name", self.name, *worker_args]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, bufsize=0, creationflags=self.flags,
            )
            threading.Thread(target=self._read_results, args=(self.process.stdout,), daemon=True).start()
            if on_started is not None:
                on_started(self)
            ready = self.receive(timeout=180)
            if not bool(ready.get("ready", False)):
                raise RuntimeError(f"{self.label} did not finish initialization")
        except Exception:
            self.reset_inference_session()
            raise

    def _read_results(self, stream):
        try:
            while True:
                packet = read_packet(stream)
                self.results.put(packet)
                if packet is None:
                    return
        except Exception as error:
            self.results.put(error)

    def receive(self, timeout):
        try:
            result = self.results.get(timeout=timeout)
        except queue.Empty:
            self.failed = True
            raise RuntimeError(f"{self.label} did not respond within {timeout} seconds") from None
        if result is None or isinstance(result, Exception):
            self.failed = True
            detail = ""
            if not self.log.closed:
                self.log.seek(0, 2)
                self.log.seek(max(0, self.log.tell() - 4000))
                detail = self.log.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{self.label} stopped: {detail or result or 'worker closed its output'}")
        return result

    def request(self, frame):
        def send():
            try:
                write_packet(self.process.stdin, frame=frame)
            except Exception as error:
                self.results.put(error)

        # A hung GPU can stop the worker reading stdin; bound the write as well as the reply.
        threading.Thread(target=send, daemon=True).start()
        return self.receive(timeout=self.frame_timeout)

    def reset_inference_session(self):
        process = self.process
        if process is None:
            self.log.close()
            return
        self.process = None
        try:
            if self.failed:
                raise subprocess.TimeoutExpired(process.args, 0)
            process.stdin.close()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            # Stop only the container this session created, including after a CUDA hang.
            try:
                subprocess.run(
                    ["docker", "rm", "--force", self.name], timeout=15,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=self.flags, check=False,
                )
            finally:
                process.kill()
                process.wait(timeout=5)
        finally:
            process.stdin.close()
            process.stdout.close()
            self.log.close()
