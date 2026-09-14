"""Build the unpruned FP16 reference pipeline inside the DARTF GPU container.

python /app/dartf/build_engines.py --dart /dart --ckpt /dart/weights/sam3.pt --out /assets

This does not build DARTF's calibrated W8A8 variant. All 32 vision blocks, the
segmentation head, and SAM3 memory attention are retained; no LQ plugins or
calibration images are needed. Text runs once per prompt through ONNX Runtime.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


DART_COMMIT = "16fada39054ac5058f6e7c1e8748cb9cc288f90c"
STAGES = ("backbone", "text", "heads", "tracker", "plans")


def tracker_source(source):
    """The pinned exporter omits the learned temporal position buffer."""
    anchor = 'np.save(f"{a.out}/pe_mem.npy", PE_MEM.numpy());'
    if source.count(anchor) != 1 or 'np.save(f"{a.out}/tpos_enc.npy"' in source:
        raise RuntimeError("DART tracker exporter changed; review the temporal-position fix")
    return source.replace(
        anchor,
        anchor + ' np.save(f"{a.out}/tpos_enc.npy", tracker.maskmem_tpos_enc.detach().cpu().numpy());',
    )


def run(*command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def export_backbone(args):
    import onnx
    import torch
    from transformers import Sam3Model

    class Backbone(torch.nn.Module):
        def __init__(self, vision):
            super().__init__()
            self.vision = vision

        def forward(self, images):
            # Sam3VisionModel.forward, with its pre-neck tensor made explicit.
            hidden = self.vision.backbone(images).last_hidden_state
            trunk = hidden.view(1, 72, 72, 1024).permute(0, 3, 1, 2)
            fpn, _ = self.vision.neck(trunk)
            return fpn[0], fpn[1], fpn[2], trunk

    torch.set_num_threads(args.threads)
    model = Sam3Model.from_pretrained(
        "facebook/sam3", attn_implementation="eager", local_files_only=True,
    ).cpu().eval()
    wrapper = Backbone(model.vision_encoder).eval()
    del model  # Release the unused text/detection weights before tracing the backbone.
    with torch.no_grad():
        program = torch.onnx.export(
            wrapper, (torch.zeros(1, 3, 1008, 1008),), dynamo=True,
            opset_version=18,
            input_names=["images"],
            output_names=["fpn_0", "fpn_1", "fpn_2", "tracker_trunk"],
        )
    program.save(str(args.out / "vision_fp16.onnx"))
    onnx.checker.check_model(str(args.out / "vision_fp16.onnx"))


def export_text(args):
    import numpy as np
    import onnxruntime as ort
    import torch
    from sam3.model_builder import _create_text_encoder

    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    text = _create_text_encoder(str(args.out / "bpe_simple_vocab_16e6.txt.gz")).eval()
    state = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=True)
    state = state.get("model", state)
    prefix = "detector.backbone.language_backbone."
    text.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state

    class Text(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.text = encoder

        def forward(self, token_ids):
            _, memory = self.text.encoder(token_ids)
            features = self.text.resizer(memory.transpose(0, 1))
            return features.to(torch.float16), token_ids.eq(0).to(torch.float32)

    wrapper = Text(text).eval()
    ids = torch.zeros(16, 32, dtype=torch.int32)
    ids[:, :2] = torch.tensor([49406, 49407], dtype=torch.int32)
    encoded = text.tokenizer(["pen"], context_length=32)
    ids[0] = encoded[0].to(torch.int32)
    with torch.no_grad():
        expected = wrapper(ids)
        torch.onnx.export(
            wrapper, (ids,), str(args.out / "text_c16.onnx"),
            input_names=["token_ids"], output_names=["text_feats", "text_mask"],
            opset_version=17, dynamo=False,
        )
    session = ort.InferenceSession(str(args.out / "text_c16.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"token_ids": ids.numpy()})
    np.testing.assert_allclose(actual[0], expected[0].numpy(), rtol=0.02, atol=0.005)
    np.testing.assert_array_equal(actual[1], expected[1].numpy())


def build_plans(args):
    dartf = args.dart / "dartf"
    builds = [
        ("vision_fp16", "runtime/build_engine.py", ["--no-int8", "--dart-mode", "norm-only"]),
        ("ground_c1m", "runtime/build_engine.py", ["--no-int8"]),
        ("maskhead_q32_phase", "demo/build_mask_engine.py", ["1", "1", "1"]),
        ("trk_neck", "demo/build_trk.py", ["B=1:1:1"]),
        # Batch two objects at a time to fit the 12 GB GPU; additional objects are chunked.
        ("trk_init", "demo/build_trk.py", ["B=1:2:2"]),
        ("trk_step_v2", "demo/build_trk.py", ["B=1:2:2,Kn=5184:15552:36288,P=1:6:19"]),
    ]
    for name, builder, options in builds:
        filename = name + (".plan" if name == "vision_fp16" else "_fp16.plan")
        target = args.out / filename
        if target.is_file():
            print("Already built:", target, flush=True)
            continue
        temporary = target.with_suffix(".building.plan")
        run(sys.executable, dartf / builder, args.out / (name + ".onnx"), temporary, *options)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dart", type=Path, default=Path("/dart"))
    parser.add_argument("--ckpt", type=Path, default=Path("/dart/weights/sam3.pt"))
    parser.add_argument("--out", type=Path, default=Path("/assets"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args()
    args.dart, args.ckpt, args.out = args.dart.resolve(), args.ckpt.resolve(), args.out.resolve()
    commit = subprocess.check_output(["git", "-C", str(args.dart), "rev-parse", "HEAD"], text=True).strip()
    if commit != DART_COMMIT:
        raise RuntimeError(f"Expected DART {DART_COMMIT}, found {commit}")
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.dart, args.dart / "dartf" / "runtime", args.dart / "dartf" / "export"]
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    os.environ.pop("LQ_PLUGINS", None)
    sys.path[:0] = list(map(str, paths))
    shutil.copyfile(args.dart / "sam3/assets/bpe_simple_vocab_16e6.txt.gz", args.out / "bpe_simple_vocab_16e6.txt.gz")

    if args.stage == "all":
        for stage in STAGES:
            marker = args.out / ("." + stage + ".done")
            if marker.is_file() and marker.read_text().strip() == DART_COMMIT:
                print("Already exported:", stage, flush=True)
                continue
            run(sys.executable, Path(__file__).resolve(), "--dart", args.dart,
                "--ckpt", args.ckpt, "--out", args.out, "--threads", args.threads, "--stage", stage)
            marker.write_text(DART_COMMIT + "\n")
        print("FP16 reference engines ready:", args.out, flush=True)
    elif args.stage == "backbone":
        export_backbone(args)
    elif args.stage == "text":
        export_text(args)
    elif args.stage == "heads":
        run(sys.executable, args.dart / "dartf/demo/export_ground_mask.py", "--dart", args.dart,
            "--ckpt", args.ckpt, "--out", args.out, "--buckets", "1", "--phase-conv", "--threads", args.threads)
    elif args.stage == "tracker":
        source = (args.dart / "dartf/demo/export_tracker.py").read_text(encoding="utf-8")
        patched = args.out / "export_tracker_with_tpos.py"
        patched.write_text(tracker_source(source), encoding="utf-8")
        run(sys.executable, patched, "--dart", args.dart, "--ckpt", args.ckpt,
            "--out", args.out, "--export", "--v2", "--threads", args.threads)
    elif args.stage == "plans":
        build_plans(args)


if __name__ == "__main__":
    main()
