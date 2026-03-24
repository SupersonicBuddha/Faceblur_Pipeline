"""
Elliptical blur mask generation.

We generate a filled ellipse mask (uint8, 0/255) from a padded bounding box.
The mask is used to composite the blurred region onto the original frame.

Design
------
* Pad the bbox by ``padding_ratio`` on each side.
* Clip to frame boundaries.
* Draw a filled white ellipse inscribed in the padded bbox.

The oval shape is more natural than a hard rectangle — it avoids
blurring ears/hair aggressively while still covering the full face.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from src.tracking.types import BBox


def make_ellipse_mask(
    frame_shape: Tuple[int, int],
    bbox: BBox,
    padding_ratio: float = 0.35,
) -> np.ndarray:
    """
    Generate a single-channel uint8 mask with a filled white ellipse.

    Parameters
    ----------
    frame_shape:
        ``(height, width)`` of the target frame.
    bbox:
        Detected/tracked face bounding box ``(x1, y1, x2, y2)``.
    padding_ratio:
        Fractional expansion applied to each side of the bbox.
        E.g. 0.35 expands a 100×100 box to ~170×170 before clipping.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(H, W)`` — 255 inside the ellipse, 0 outside.
    """
    H, W = frame_shape
    x1, y1, x2, y2 = bbox

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    # Padded bbox, clipped to frame
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(W - 1, x2 + pad_x)
    py2 = min(H - 1, y2 + pad_y)

    center_x = (px1 + px2) // 2
    center_y = (py1 + py2) // 2
    axis_x = max(1, (px2 - px1) // 2)
    axis_y = max(1, (py2 - py1) // 2)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(center_x, center_y),
        axes=(axis_x, axis_y),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,  # filled
    )
    return mask


def local_ellipse_mask(roi_h: int, roi_w: int) -> np.ndarray:
    """
    Generate a filled ellipse mask inscribed in a rectangle of size
    ``(roi_h, roi_w)``.

    Used by the ROI-based blur compositor: the ellipse exactly fits the
    padded bounding-box ROI that was already clipped to frame boundaries.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(roi_h, roi_w)`` — 255 inside, 0 outside.
    """
    mask = np.zeros((max(1, roi_h), max(1, roi_w)), dtype=np.uint8)
    cx = max(1, roi_w // 2)
    cy = max(1, roi_h // 2)
    ax = max(1, roi_w // 2)
    ay = max(1, roi_h // 2)
    cv2.ellipse(
        mask,
        center=(cx, cy),
        axes=(ax, ay),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )
    return mask


def padded_bbox(
    bbox: BBox,
    padding_ratio: float,
    frame_shape: Tuple[int, int],
) -> BBox:
    """
    Return the padded and clipped bounding box (without the ellipse step).

    Useful for other callers that need the region before mask generation.
    """
    H, W = frame_shape
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(W - 1, x2 + pad_x),
        min(H - 1, y2 + pad_y),
    )
