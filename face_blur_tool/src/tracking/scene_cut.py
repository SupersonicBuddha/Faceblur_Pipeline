"""
Lightweight scene-cut detector based on histogram difference.

We compare the normalised colour histogram of consecutive frames.
A large delta indicates a hard cut (flash, edit, transition).

This is intentionally simple — we don't need perfect detection.
False negatives (missed cuts) just mean the tracker keeps a stale track
for a few frames, which is safe.  False positives reset the tracker,
causing a brief re-detection lag, which is also acceptable.

The histogram is computed over a downscaled version of the frame for speed.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.logging_utils import get_logger

logger = get_logger(__name__)

# Internal resolution used for histogram computation
_HIST_RESIZE = (160, 90)  # width × height


class SceneCutDetector:
    """
    Stateful per-video scene cut detector.

    Call :meth:`is_cut` for each consecutive frame pair.

    Parameters
    ----------
    threshold:
        Histogram delta above which a scene cut is declared.
        Range [0, 1]; 0.40 is a reasonable default.
    """

    def __init__(self, threshold: float = 0.40) -> None:
        self._threshold = threshold
        self._prev_hist: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Discard the stored previous-frame histogram."""
        self._prev_hist = None

    def is_cut(self, frame: np.ndarray) -> bool:
        """
        Compare *frame* to the previous frame.

        Returns True if a scene cut is detected.
        Always returns False for the first frame (no previous frame).
        """
        hist = _compute_histogram(frame)
        if self._prev_hist is None:
            self._prev_hist = hist
            return False

        delta = _histogram_delta(self._prev_hist, hist)
        self._prev_hist = hist

        if delta > self._threshold:
            logger.debug(f"Scene cut detected (delta={delta:.3f})")
            return True
        return False


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    """
    Compute a normalised 3-channel (B, G, R) histogram from a downscaled frame.

    Returns a 1-D array of length 3 × 256 = 768.
    """
    small = cv2.resize(frame, _HIST_RESIZE, interpolation=cv2.INTER_AREA)
    hists = []
    for ch in range(3):
        h = cv2.calcHist([small], [ch], None, [256], [0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten())
    return np.concatenate(hists)


def _histogram_delta(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Return the L1 distance between two normalised histograms, scaled to [0, 1].
    """
    diff = np.abs(h1 - h2).sum()
    # Maximum possible L1 distance for 3 normalised channels is 6.0
    # (each channel sums to 1, max diff = 2, 3 channels = 6)
    return float(diff / 6.0)
