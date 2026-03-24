"""
Composite blurred faces back onto the original frame.

ROI-based approach (V1.1)
--------------------------
Rather than blurring the entire frame and masking, we:

1. For each face bbox, compute the padded ROI (the ellipse bounding box).
2. Extract the ROI from the frame.
3. Blur only the ROI — much cheaper for 1080p+ frames with small faces.
4. Generate a local ellipse mask inscribed in the ROI rectangle.
5. Blend the blurred ROI back using the mask, write into the output array.

This is equivalent to the full-frame approach when no boundary clipping
occurs (the ellipse fills the ROI exactly), and avoids wasted computation
on pixels that will be masked out anyway.

Multiple faces are handled by iterating over bboxes.  If ROIs overlap,
the later face in the list is applied on top, which is correct (both
faces get blurred; the overlap region gets blurred twice, which is fine).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.blur.blur_ops import gaussian_blur_frame
from src.blur.masks import local_ellipse_mask, padded_bbox
from src.config import FaceBlurConfig
from src.tracking.types import BBox


def apply_face_blurs(
    frame: np.ndarray,
    bboxes: List[BBox],
    config: FaceBlurConfig,
) -> np.ndarray:
    """
    Apply Gaussian blur to all face regions in *frame*.

    If *bboxes* is empty, the original frame is returned unchanged.

    Each face is processed independently via ROI extraction to avoid
    blurring the entire frame for a small number of faces.

    Parameters
    ----------
    frame:
        uint8 BGR input frame.
    bboxes:
        List of face bounding boxes ``(x1, y1, x2, y2)``.
    config:
        Pipeline config (kernel size, blur strength, padding ratio).

    Returns
    -------
    np.ndarray
        A new uint8 BGR frame with face regions blurred.
    """
    if not bboxes:
        return frame

    H, W = frame.shape[:2]
    out = frame.copy()

    for bbox in bboxes:
        # ── Padded ROI (clipped to frame) ──────────────────────────────────
        px1, py1, px2, py2 = padded_bbox(bbox, config.blur_padding_ratio, (H, W))
        roi_w = px2 - px1
        roi_h = py2 - py1
        if roi_w <= 0 or roi_h <= 0:
            continue  # degenerate bbox — skip

        # ── Blur only the ROI ──────────────────────────────────────────────
        roi = frame[py1:py2, px1:px2]
        blurred_roi = gaussian_blur_frame(
            roi,
            kernel_size=config.effective_blur_kernel,
            sigma=config.blur_strength,
        )

        # ── Ellipse mask in ROI-local coordinates ──────────────────────────
        mask = local_ellipse_mask(roi_h, roi_w)

        # ── Composite: inside ellipse → blurred; outside → original ────────
        mask3 = mask[:, :, np.newaxis].astype(np.float32) / 255.0
        blended = (
            blurred_roi.astype(np.float32) * mask3
            + roi.astype(np.float32) * (1.0 - mask3)
        ).astype(np.uint8)

        out[py1:py2, px1:px2] = blended

    return out
