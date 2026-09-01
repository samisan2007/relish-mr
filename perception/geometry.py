"""Mask -> geometry extraction (build plan: centroid, diameter, rim, base).

Pure numpy, no ML dependencies, so it can be unit-tested against
synthetic masks. Rim/base extraction (for the vessel fill-line) will be
added in a later batch.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskGeometry:
    centroid_xy: tuple[float, float]  # pixel coords
    diameter_px: float                # longest bbox side
    area_px: int


def mask_geometry(mask: np.ndarray) -> MaskGeometry | None:
    """Centroid, characteristic diameter, and area of a boolean HxW mask.

    Returns None for an empty mask.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return MaskGeometry(
        centroid_xy=(float(xs.mean()), float(ys.mean())),
        diameter_px=float(max(xs.max() - xs.min(), ys.max() - ys.min())),
        area_px=int(xs.size),
    )


def normalized(point_xy: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    """Pixel coords -> [0,1] normalized coords (what the service will return to Unity)."""
    return point_xy[0] / width, point_xy[1] / height
