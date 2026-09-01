"""Batch 1 smoke test CLI: SAM 3 text-prompt segmentation on a single image.

Usage:
    python test_sam3.py                          # built-in COCO kitchen image, prompt "dial"
    python test_sam3.py photo.jpg "meatball"     # your own photo + text prompt
    python test_sam3.py photo.jpg "mug" --threshold 0.4

Logic lives in sam3_runner.py (inference), geometry.py (mask measurements),
viz.py (overlay). This file only wires them together.
"""

import argparse
import io
import sys
from pathlib import Path

import requests
import torch
from PIL import Image

from geometry import mask_geometry
from sam3_runner import Sam3Runner
from viz import draw_instances

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000136466.jpg"
DEFAULT_PROMPT = "dial"


def load_image(source: str) -> tuple[Image.Image, Path]:
    if source.startswith("http"):
        img = Image.open(io.BytesIO(requests.get(source, timeout=30).content))
        out_stem = Path("test_image")
    else:
        path = Path(source)
        if not path.exists():
            sys.exit(f"Image not found: {path}")
        img = Image.open(path)
        out_stem = path.with_suffix("")
    return img.convert("RGB"), out_stem


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM 3 text-prompt segmentation smoke test")
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE_URL,
                        help="Path or URL of the image (default: COCO kitchen photo)")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT,
                        help='Text concept to segment, e.g. "meatball" (default: "dial")')
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: running on CPU — expect this to be slow")

    runner = Sam3Runner()
    print(f"Model loaded in {runner.load_seconds:.1f}s on {runner.device}")

    image, out_stem = load_image(args.image)
    print(f'Image: {image.size[0]}x{image.size[1]} | prompt: "{args.prompt}"')

    # First pass includes CUDA warmup; report both cold and warm timings.
    result = runner.segment(image, args.prompt, threshold=args.threshold)
    print(f"inference (cold): {result.inference_ms:.0f}ms")
    result = runner.segment(image, args.prompt, threshold=args.threshold)
    print(f"inference (warm): {result.inference_ms:.0f}ms")

    n = len(result.instances)
    print(f'\nFound {n} instance(s) of "{args.prompt}"')
    if n == 0:
        print("Nothing found — try a lower --threshold or a different phrase.")
        return

    for i, inst in enumerate(result.instances):
        geo = mask_geometry(inst.mask)
        print(f"  #{i}: score={inst.score:.3f} centroid=({geo.centroid_xy[0]:.0f},{geo.centroid_xy[1]:.0f}) "
              f"diameter={geo.diameter_px:.0f}px area={geo.area_px}px²")

    out_path = out_stem.parent / f"{out_stem.name}_sam3.png"
    draw_instances(image, result.instances).save(out_path)
    print(f"\nOverlay saved to: {out_path}")


if __name__ == "__main__":
    main()
