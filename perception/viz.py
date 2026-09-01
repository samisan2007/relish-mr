"""Overlay rendering for segmentation results — debug/inspection only."""

import numpy as np
from PIL import Image, ImageDraw

from geometry import mask_geometry
from sam3_runner import Instance

COLORS = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
    (175, 82, 222), (255, 204, 0), (90, 200, 250), (255, 45, 85),
]


def draw_instances(
    image: Image.Image,
    instances: list[Instance],
    show_masks: bool = True,
    show_boxes: bool = True,
) -> Image.Image:
    """Tinted masks and/or bounding boxes with scores, plus centroid dots."""
    overlay = image.convert("RGBA")
    for i, inst in enumerate(instances):
        # Key colour to the track id so a tracked object keeps its colour across frames.
        color = COLORS[(inst.obj_id if inst.obj_id is not None else i) % len(COLORS)]

        if show_masks:
            tint = np.zeros((*inst.mask.shape, 4), dtype=np.uint8)
            tint[inst.mask] = (*color, 110)
            overlay = Image.alpha_composite(overlay, Image.fromarray(tint))

        draw = ImageDraw.Draw(overlay, "RGBA")
        if show_boxes:
            x1, y1, x2, y2 = inst.box_xyxy
            draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=3)
            label = f"{inst.score:.2f}" if inst.obj_id is None else f"#{inst.obj_id}  {inst.score:.2f}"
            draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255, 255))

        geo = mask_geometry(inst.mask)
        if geo:
            cx, cy = geo.centroid_xy
            r = 4
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, 255))
    return overlay.convert("RGB")
