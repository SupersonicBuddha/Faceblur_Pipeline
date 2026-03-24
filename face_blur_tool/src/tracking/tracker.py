"""
IoU-based multi-object face tracker with conservative persistence.

Design
------
* Hungarian algorithm (scipy.optimize.linear_sum_assignment) assigns
  new detections to existing tracks using Intersection-over-Union as cost.
* Tracks persist for ``persistence_frames`` consecutive missed frames
  before being dropped — this is deliberately conservative to reduce misses.
* A new track is spawned for every unmatched detection.
* Scene cuts immediately reset all active tracks (see scene_cut.py).

Tradeoffs
---------
We chose IoU matching over a full Kalman filter because:
  * Simpler, fewer tuneable parameters, easier to debug.
  * For offline batch with detection every 4 frames, a lightweight
    motion model (linear extrapolation) is sufficient.
  * A Kalman filter is a natural upgrade path if needed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import]

from src.detectors.base import Detection
from src.tracking.types import BBox, Track, TrackedFrame
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _iou(b1: BBox, b2: BBox) -> float:
    """Compute IoU between two bboxes (x1,y1,x2,y2)."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def _extrapolate_bbox(track: Track) -> BBox:
    """
    Simple linear extrapolation from the last two bbox positions.

    Falls back to the last known bbox if history is too short.
    """
    history = track.bbox_history
    if len(history) < 2:
        return track.bbox
    b1 = history[-2]
    b2 = history[-1]
    dx = b2[0] - b1[0]
    dy = b2[1] - b1[1]
    return (
        b2[0] + dx,
        b2[1] + dy,
        b2[2] + dx,
        b2[3] + dy,
    )


class FaceTracker:
    """
    Stateful multi-face tracker.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU to consider a detection-to-track match valid.
    persistence_frames:
        Maximum number of consecutive missed frames before a track is dropped.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        persistence_frames: int = 12,
    ) -> None:
        self._iou_thresh = iou_threshold
        self._persistence = persistence_frames
        self._tracks: Dict[int, Track] = {}
        # Instance-local ID counter; deterministic per tracker lifecycle
        self._next_track_id: int = 1

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop all active tracks (call on scene cut)."""
        self._tracks.clear()
        logger.debug("Tracker reset (scene cut)")

    def update(
        self,
        frame_idx: int,
        detections: List[Detection],
    ) -> TrackedFrame:
        """
        Update tracker with new detections for *frame_idx*.

        Returns :class:`TrackedFrame` containing all currently active tracks.
        """
        if not self._tracks:
            # Bootstrap: start a new track for every detection
            for det in detections:
                self._new_track(det, frame_idx)
            return self._current_tracked_frame(frame_idx)

        # Predict next positions for all active tracks
        track_ids = list(self._tracks.keys())
        predicted: List[BBox] = [
            _extrapolate_bbox(self._tracks[tid]) for tid in track_ids
        ]
        det_bboxes: List[BBox] = [d.bbox for d in detections]

        # Build IoU cost matrix
        if predicted and det_bboxes:
            cost = np.zeros((len(predicted), len(det_bboxes)), dtype=np.float32)
            for i, pb in enumerate(predicted):
                for j, db in enumerate(det_bboxes):
                    cost[i, j] = 1.0 - _iou(pb, db)

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_tracks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= (1.0 - self._iou_thresh):
                    tid = track_ids[r]
                    self._tracks[tid].update(detections[c].bbox, frame_idx)
                    matched_tracks.add(tid)
                    matched_dets.add(c)

            # Unmatched tracks → miss
            for i, tid in enumerate(track_ids):
                if tid not in matched_tracks:
                    self._tracks[tid].mark_missed()

            # Unmatched detections → new tracks
            for j, det in enumerate(detections):
                if j not in matched_dets:
                    self._new_track(det, frame_idx)
        else:
            # No detections → all existing tracks miss a frame
            for tid in track_ids:
                self._tracks[tid].mark_missed()
            for det in detections:
                self._new_track(det, frame_idx)

        # Prune dead tracks
        dead = [
            tid
            for tid, t in self._tracks.items()
            if t.missed_frames > self._persistence
        ]
        for tid in dead:
            del self._tracks[tid]

        return self._current_tracked_frame(frame_idx)

    # ── Private helpers ────────────────────────────────────────────────────

    def _new_track(self, detection: Detection, frame_idx: int) -> None:
        tid = self._next_track_id
        self._next_track_id += 1
        t = Track(
            track_id=tid,
            bbox=detection.bbox,
            last_seen_frame=frame_idx,
        )
        t.bbox_history.append(detection.bbox)
        self._tracks[tid] = t

    def _current_tracked_frame(self, frame_idx: int) -> TrackedFrame:
        return TrackedFrame(
            frame_idx=frame_idx,
            tracks=list(self._tracks.values()),
        )
