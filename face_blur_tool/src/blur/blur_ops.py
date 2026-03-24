"""
Blur operations applied inside a mask region.

We use Gaussian blur as the default — it is smooth, well-understood,
and strong enough to defeat face recognition at the kernel sizes we use.

The blur is applied to the full frame first, then composited using the mask,
so the blur does not "leak" outside the ellipse boundary.
"""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur_frame(
    frame: np.ndarray,
    kernel_size: int,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Apply Gaussian blur to the entire frame.

    Parameters
    ----------
    frame:
        uint8 BGR frame.
    kernel_size:
        Must be odd and ≥ 1.
    sigma:
        Standard deviation (0 = auto from kernel_size).

    Returns
    -------
    np.ndarray
        Blurred copy of the frame (same shape/dtype).
    """
    k = _ensure_odd(max(1, kernel_size))
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)


def pixelate_region(
    frame: np.ndarray,
    block_size: int = 20,
) -> np.ndarray:
    """
    Alternative: pixelate the entire frame (not used by default).

    Included for completeness and testing.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(
        frame,
        (max(1, w // block_size), max(1, h // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1
