"""
Temporal smoothing of bounding box coordinates.

Smoothing reduces jitter in blur regions so the output looks stable.
We use a simple causal rolling mean (no future frames required), which
is appropriate for the streaming frame-by-frame processing model.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from src.tracking.types import BBox


class BBoxSmoother:
    """
    Per-track rolling-average smoother for bounding boxes.

    Parameters
    ----------
    window:
        Number of recent frames to average over.
    """

    def __init__(self, window: int = 7) -> None:
        if window < 1:
            raise ValueError(f"window must be ≥ 1, got {window}")
        self._window = window
        # track_id → deque of recent bboxes
        self._history: Dict[int, Deque[BBox]] = {}

    def smooth(self, track_id: int, bbox: BBox) -> BBox:
        """
        Add *bbox* to the history for *track_id* and return the smoothed bbox.
        """
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self._window)
        q = self._history[track_id]
        q.append(bbox)
        return _mean_bbox(list(q))

    def reset_track(self, track_id: int) -> None:
        """Discard history for a track (e.g. after scene cut)."""
        self._history.pop(track_id, None)

    def reset_all(self) -> None:
        """Discard all history."""
        self._history.clear()

    def remove_stale_tracks(self, active_ids: List[int]) -> None:
        """Remove history entries for track IDs no longer active."""
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]


def _mean_bbox(bboxes: List[BBox]) -> BBox:
    """Return the component-wise integer mean of a list of bboxes."""
    if not bboxes:
        raise ValueError("Cannot average empty bbox list")
    n = len(bboxes)
    x1 = round(sum(b[0] for b in bboxes) / n)
    y1 = round(sum(b[1] for b in bboxes) / n)
    x2 = round(sum(b[2] for b in bboxes) / n)
    y2 = round(sum(b[3] for b in bboxes) / n)
    return (x1, y1, x2, y2)
