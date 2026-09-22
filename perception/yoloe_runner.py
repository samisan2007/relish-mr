"""YOLOE video tracking: a fast alternative/complement to Sam3VideoTracker.

Two backends, both exposing the same start_stream(text)/track_frame(session, frame)
shape as Sam3VideoTracker so the webcam UI can swap between them without special-casing:

- YoloeVideoTracker: pure text-prompt mode (YOLOE's own open-vocabulary text encoder,
  a lightweight MobileCLIP model). Earlier runs found low confidence on specific
  food nouns, but those measurements predate the RGB/BGR fix and need retesting.

- HybridVideoTracker: uses a text-grounding model (SAM3 in-process, or SAM 3.1 in its
  Docker worker) to find every instance in a seed frame, then extracts YOLOE visual
  embeddings from that frame. Later frames reuse those embeddings with persist=True,
  and the seeder is re-run whenever YOLOE has lost everything, plus optionally every
  `reground_every` frames, so the exemplars follow the object through pose changes.
"""

import time

import cv2
import numpy as np
import torch

from sam3_runner import Instance

YOLOE_MODEL_ID = "yoloe-11l-seg.pt"


def _area(box_xyxy) -> float:
    x1, y1, x2, y2 = box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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


class HybridSession(dict):
    """The hybrid's per-stream state. A dict so every existing session["key"] access
    keeps working, with the teardown hook the UI and WorkerTracker.track already look
    for — a Docker-worker seeder leaves a container running without it."""

    def reset_inference_session(self):
        seed_session = self.pop("seed_session", None)
        if hasattr(seed_session, "reset_inference_session"):
            seed_session.reset_inference_session()


class HybridVideoTracker:
    """Ground the prompt with a slow, reliable text model (SAM3 or SAM 3.1), then track
    the resulting exemplars with YOLOE's persistent tracker.
    Re-attempts grounding on every frame until something is found, then hands over to
    YOLOE, re-grounding when the track is lost — frozen exemplars lose the object on
    pose change — and on a fixed interval if `reground_every` is set.
    """

    def __init__(self, seed_tracker, model_id: str = YOLOE_MODEL_ID, conf: float = 0.15,
                 reground_every: int = 0, drift_scale: float = 8.0):
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

        self.seed_tracker = seed_tracker  # stays open for the stream; re-grounding it beats paying a Docker boot again
        self.model = YOLOE(model_id)
        self.vp_predictor = YOLOEVPSegPredictor
        self.conf = conf
        # Periodic re-grounding, in frames; 0 disables it. Off by default because
        # installing fresh exemplars restarts YOLOE's track IDs — YOLOE.predict drops
        # self.predictor after set_classes, so the tracker is rebuilt (measured: ids
        # [1] -> [2] across one re-ground). Losing every detection costs no IDs, so
        # that trigger below always runs. Set this when drift matters more than IDs.
        self.reground_every = reground_every
        # YOLOE matches the exemplar embedding, not the words, and a thin exemplar (a pen,
        # a watch strap) generalizes badly: observed locking onto a whole torso at 0.29,
        # over the 0.15 floor. Nothing else in the pipeline can tell that box is wrong, so
        # without this a single bad frame is absorbing. Reject anything more than this many
        # times the biggest seeded box in area; the frame then reads as lost and re-grounds.
        # 8x area is ~2.8x linear, so an object approaching the camera still survives.
        # 0 disables the check.
        self.drift_scale = drift_scale
        self.device_arg = 0 if torch.cuda.is_available() else "cpu"
        self.load_seconds = 0.0

    @property
    def device(self) -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def start_stream(self, text: str, on_started=None):
        # Sam3VideoTracker.start_stream has no on_started; only the Docker-worker
        # seeders do, and they need it so Stop can cancel a container mid-boot.
        kwargs = {"on_started": on_started} if on_started is not None else {}
        return HybridSession(
            prompt=text,
            seed_session=self.seed_tracker.start_stream(text, **kwargs),
            seeded=False,
            since_ground=0,
            exemplar_area=0.0,
            lost=False,
            visual_prompts=None,
            reference_image=None,
        )

    def _ground(self, session, frame: np.ndarray) -> tuple[list[Instance], float]:
        """Run the seeder on this frame and, if it finds anything, stage new exemplars."""
        instances, ms = self.seed_tracker.track_frame(session["seed_session"], frame)
        # Count the attempt, not the hit: a seeder that misses one scheduled frame while
        # YOLOE is still tracking fine must not then be called on every frame after it.
        session["since_ground"] = 0
        if instances:
            # One exemplar generalizes poorly to the rest of the class (observed: a
            # single box found 1/13 meatballs vs. 10-13/13 with several) — seed with
            # every instance the seeder found this frame, not just the top-scoring one.
            session["visual_prompts"] = dict(
                bboxes=np.array([list(i.box_xyxy) for i in instances]),
                cls=np.zeros(len(instances), dtype=int),
            )
            # Keep the exact image these boxes describe. cvtColor also copies
            # it, so a caller reusing its RGB capture buffer cannot alter it.
            session["reference_image"] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            session["exemplar_area"] = max(_area(i.box_xyxy) for i in instances)
        return instances, ms

    def _plausible(self, session, instances: list[Instance]) -> list[Instance]:
        """Drop boxes far larger than anything the seeder pointed at."""
        limit = session["exemplar_area"] * self.drift_scale
        if not limit:
            return instances
        return [i for i in instances if _area(i.box_xyxy) <= limit]

    def track_frame(self, session, frame: np.ndarray) -> tuple[list[Instance], float]:
        if not session["seeded"]:
            instances, ms = self._ground(session, frame)
            session["seeded"] = bool(instances)
            return instances, ms

        seed_ms = 0.0
        session["since_ground"] += 1
        # Re-ground on a schedule, and immediately once YOLOE has lost everything —
        # waiting out the interval with nothing tracked is dead time.
        if (self.reground_every and session["since_ground"] >= self.reground_every) or session["lost"]:
            _, seed_ms = self._ground(session, frame)

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
        ms = (time.perf_counter() - t0) * 1000 + seed_ms
        instances = self._plausible(session, _to_instances(results[0], frame.shape[:2]))
        # An emptied frame re-grounds on the next one, so drift corrects itself instead
        # of persisting — that is the point of dropping the box rather than flagging it.
        session["lost"] = not instances
        return instances, ms
