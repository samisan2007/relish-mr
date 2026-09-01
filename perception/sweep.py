"""Batch 1 sweep: run SAM 3 over the Test Data images with food/vessel prompts.

Loads the model once, runs every (image, prompt) pair, prints one result line
per pair, saves overlays to "Test Data/results/".
"""

import time
from pathlib import Path

from PIL import Image

from geometry import mask_geometry
from sam3_runner import Sam3Runner
from viz import draw_instances

DATA_DIR = Path(__file__).parent.parent / "Test Data"
OUT_DIR = DATA_DIR / "results"

# image stem -> prompts to try
PLAN = {
    "test_img_01": ["herbs", "sink"],
    "test_img_02": ["meatball", "pot"],
    "test_img_03": ["meatball", "lemon"],
    "test_img_04": ["meatball"],
    "test_img_05": ["shrimp", "frying pan"],
    "test_img_06": ["leek", "bowl"],
    "test_img_07": ["meatball", "pot"],
    "test_img_08": ["red onion", "lemon", "bowl"],
}


def main() -> None:
    runner = Sam3Runner()
    print(f"Model loaded in {runner.load_seconds:.1f}s on {runner.device}\n")
    OUT_DIR.mkdir(exist_ok=True)

    t_total = time.perf_counter()
    for stem, prompts in PLAN.items():
        image = Image.open(DATA_DIR / f"{stem}.png").convert("RGB")
        for prompt in prompts:
            result = runner.segment(image, prompt)
            sizes = [f"{mask_geometry(i.mask).diameter_px:.0f}px" for i in result.instances]
            print(f"{stem} | {prompt:12s} | {len(result.instances)} found | "
                  f"{result.inference_ms:.0f}ms | diameters: {', '.join(sizes) if sizes else '-'}")
            if result.instances:
                out = OUT_DIR / f"{stem}_{prompt.replace(' ', '_')}.png"
                draw_instances(image, result.instances).save(out)
    print(f"\nTotal sweep: {time.perf_counter() - t_total:.1f}s | overlays in {OUT_DIR}")


if __name__ == "__main__":
    main()
