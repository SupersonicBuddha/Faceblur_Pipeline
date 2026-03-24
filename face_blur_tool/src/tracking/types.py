"""
Shared types for the tracking subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# (x1, y1, x2, y2)
BBox = Tuple[int, int, int, int]


@dataclass
class Track:
    """
    A face track: a sequence of bounding boxes across frames.

    Attributes
    ----------
    track_id:
        Unique integer ID for this track.
    bbox:
        Current (or last known) bounding box.
    last_seen_frame:
        Frame index at which the track was last updated by a detection.
    missed_frames:
        Consecutive frames since the last confirmed detection.
    confirmed:
        True once the track has been active for at least one frame.
    bbox_history:
        Recent bounding boxes for smoothing (newest last).
    """

    track_id: int
    bbox: BBox
    last_seen_frame: int
    missed_frames: int = 0
    confirmed: bool = True
    bbox_history: List[BBox] = field(default_factory=list)

    def update(self, bbox: BBox, frame_idx: int) -> None:
        self.bbox = bbox
        self.last_seen_frame = frame_idx
        self.missed_frames = 0
        self.bbox_history.append(bbox)

    def mark_missed(self) -> None:
        self.missed_frames += 1
        # Carry forward the last known bbox (for conservative continuation)
        self.bbox_history.append(self.bbox)

    @property
    def is_lost(self) -> bool:
        """Track has been lost for too many consecutive frames."""
        return self.missed_frames > 0  # caller checks against persistence threshold


@dataclass
class TrackedFrame:
    """All active tracks for one output frame."""

    frame_idx: int
    tracks: List[Track] = field(default_factory=list)
