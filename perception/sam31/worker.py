"""Causal SAM 3.1: one RGB request, one result, with persistent object memory."""

import argparse
from copy import copy
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dartf_worker import read_packet, write_packet
from sam31.benchmark import CHECKPOINT_REVISION, SOURCE_REVISION, to_instances


def prune_memory(state, frame_idx, history=32):
    """Keep recent tracking memory and each bucket's first conditioning frame."""
    cutoff = frame_idx - history + 1
    state["cached_frame_outputs"].clear()  # UI never seeks backwards or edits past masks.
    metadata = state["tracker_metadata"]
    for mapping in (metadata.get("obj_id_to_sam2_score_frame_wise", {}),
                    metadata.get("rank0_metadata", {}).get("suppressed_obj_ids", {})):
        for idx in list(mapping):
            if idx < cutoff:
                del mapping[idx]
    # ponytail: bounded 32-frame memory changes long-occlusion behavior; compare
    # longer retention on real clips before treating this as a production tracker.
    for tracker_state in state["sam2_inference_states"]:
        first = tracker_state["first_ann_frame_idx"]
        outputs = [tracker_state["output_dict"],
                   *tracker_state["output_dict_per_obj"].values(),
                   *tracker_state["temp_output_dict_per_obj"].values()]
        for output in outputs:
            for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for idx in list(output[key]):
                    if idx < cutoff and not (key == "cond_frame_outputs" and idx == first):
                        del output[key][idx]
        for key, indices in tracker_state["consolidated_frame_inds"].items():
            indices.intersection_update(tracker_state["output_dict"][key])
        for idx in list(tracker_state["frames_already_tracked"]):
            if idx < cutoff:
                del tracker_state["frames_already_tracked"][idx]
        for key in ("point_inputs_per_obj", "mask_inputs_per_obj"):
            for inputs in tracker_state.get(key, {}).values():
                for idx in list(inputs):
                    if idx < cutoff and idx != first:
                        del inputs[idx]


def compile_detector(model):
    """Compile only the detector, whose shapes the fixed 1008 px input pins down.
    Upstream's _compile_model also compiles tracker and matching functions for
    static shapes that change with the number of objects and memory frames, so
    live each new count recompiled: 21, 16, 86 and 32 s stalls as three pens came
    into view, and again whenever the count changed. That looked like a frozen
    stream, for ~15% over eager once settled."""
    if getattr(model, "_model_is_compiled", False) or not model.compile_model:
        return
    import sam3.model.sam3_video_base as video_base

    tracker = model.tracker
    eager = (tracker.maskmem_backbone.forward, tracker.transformer.encoder.forward,
             tracker.sam_mask_decoder.forward, tracker._suppress_object_pw_area_shrinkage,
             video_base._associate_det_trk_compilable)
    model._compile_model()
    (tracker.maskmem_backbone.forward, tracker.transformer.encoder.forward,
     tracker.sam_mask_decoder.forward, tracker._suppress_object_pw_area_shrinkage,
     video_base._associate_det_trk_compilable) = eager


class Sam31LiveTracker:
    def __init__(self, prompt, compile_model=False, image_only=False, threshold=0.5):
        import torch
        import sam3
        from huggingface_hub import hf_hub_download
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        source = Path(sam3.__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != SOURCE_REVISION:
            raise RuntimeError("SAM 3.1 source differs from the tested revision; see sam31/README.md.")
        if not prompt.strip() or not 0 < threshold < 1:
            raise ValueError("Provide a prompt and a confidence threshold between 0 and 1")
        checkpoint = hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt",
                                     revision=CHECKPOINT_REVISION, local_files_only=True)
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint, use_fa3=False, compile=compile_model,
            warm_up=False, max_num_objects=16, async_loading_frames=False)
        self.model = self.predictor.model
        self.model.use_batched_grounding = False
        self.model.postprocess_batch_size = 1
        self.model.hotstart_delay = 0
        if image_only:
            self.model.score_threshold_detection = min(0.4, threshold)
            self.model.new_det_thresh = threshold
        self.prompt, self.image_only = prompt, image_only
        self.state = None
        self.frame_idx = 0
        self.torch = torch

    def track_frame(self, frame):
        from PIL import Image
        from sam3.model.io_utils import load_resource_as_video_frames

        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not all(frame.shape[:2]):
            raise ValueError("SAM 3.1 frame must be a non-empty uint8 RGB HxWx3 array")
        if self.image_only:
            # Every request is an independent picture (the keyframe hybrid's detector);
            # later frames would otherwise run the video tracker on this state.
            self.state, self.frame_idx = None, 0
        if self.state is not None and frame.shape[:2] != (self.state["orig_height"], self.state["orig_width"]):
            raise ValueError("Frame dimensions changed; stop and restart tracking")
        t0 = time.perf_counter()
        idx, model = self.frame_idx, self.model
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            if self.state is None:
                self.state = model.init_state([Image.fromarray(frame)], offload_video_to_cpu=True)
                self.state["is_image_only"] = self.image_only
                images = self.state["input_batch"].img_batch
                images.tensors = [images.tensors[0]]
                if self.image_only:
                    compile_detector(model)
                _, output = model.add_prompt(self.state, frame_idx=0, text_str=self.prompt)
            else:
                state = self.state
                images, _, _ = load_resource_as_video_frames(
                    [Image.fromarray(frame)], model.image_size, offload_video_to_cpu=True,
                    img_mean=model.image_mean, img_std=model.image_std)
                batch = state["input_batch"]
                batch.img_batch.tensors.append(images[0])
                batch.img_batch.tensors[idx - 1] = None
                stage = copy(batch.find_inputs[0])
                stage.img_ids = stage.img_ids.new_tensor([idx])
                stage.img_ids_np = np.array([idx])
                batch.find_inputs.append(stage)
                state["previous_stages_out"].append(None)
                state["per_frame_geometric_prompt"].append(None)
                state["num_frames"] = idx + 1
                for tracker_state in state["sam2_inference_states"]:
                    tracker_state["num_frames"] = idx + 1
                # Only the current frame exists: detector prefetch cannot read a future frame.
                compile_detector(model)
                out = model._run_single_frame_inference(state, idx, reverse=False)
                output = model._postprocess_output(
                    state, out, removed_obj_ids=out["removed_obj_ids"],
                    suppressed_obj_ids=out["suppressed_obj_ids"],
                    unconfirmed_obj_ids=out["unconfirmed_obj_ids"])
            instances = to_instances(output, *frame.shape[:2])
            prune_memory(self.state, idx)
            self.torch.cuda.synchronize()
        self.frame_idx += 1
        height, width = frame.shape[:2]
        return {
            "masks": np.asarray([i.mask for i in instances], dtype=bool).reshape(-1, height, width),
            "boxes": np.asarray([i.box_xyxy for i in instances], dtype=np.float32).reshape(-1, 4),
            "scores": np.asarray([i.score for i in instances], dtype=np.float32),
            "ids": np.asarray([i.obj_id for i in instances], dtype=np.int64),
            "inference_ms": np.asarray((time.perf_counter() - t0) * 1000),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    with output:
        tracker = Sam31LiveTracker(args.prompt, args.compile, args.image_only, args.threshold)
        write_packet(output, ready=np.asarray(True))
        while (packet := read_packet(sys.stdin.buffer)) is not None:
            if set(packet) != {"frame"}:
                raise ValueError("SAM 3.1 request must contain exactly one 'frame' array")
            write_packet(output, **tracker.track_frame(packet["frame"]))


if __name__ == "__main__":
    main()
