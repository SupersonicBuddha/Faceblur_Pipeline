"""
Stream frames from a local video file via OpenCV.

We do NOT load the full video into memory; frames are yielded one at a time.
"""

from __future__ import annotations

from typing import Generator, Tuple

import cv2
import numpy as np

from src.logging_utils import get_logger

logger = get_logger(__name__)


class FrameReadError(RuntimeError):
    pass


def iter_frames(
    local_path: str,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield ``(frame_index, bgr_frame)`` for every frame in *local_path*.

    Parameters
    ----------
    local_path:
        Path to a local video file (.mp4, .mov, …).

    Yields
    ------
    (frame_index, frame)
        ``frame_index`` is 0-based.
        ``frame`` is a uint8 BGR numpy array of shape (H, W, 3).

    Raises
    ------
    FrameReadError
        If the file cannot be opened.
    """
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        raise FrameReadError(f"Cannot open video: {local_path!r}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()

    logger.debug(f"Read {frame_idx} frames from {local_path!r}")


def frame_count(local_path: str) -> int:
    """Return the total number of frames (may be 0 for some containers)."""
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count
