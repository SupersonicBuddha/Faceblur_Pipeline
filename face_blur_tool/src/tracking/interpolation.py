"""
Interpolate bounding boxes across short detection gaps.

When detection runs every N frames and a track is active, intermediate
frames carry the last known bbox.  This module back-fills intermediate
frames with linearly interpolated bboxes so the blur region moves
smoothly rather than jumping on detection frames.

Usage pattern
-------------
The interpolator is used as a *post-processing pass* over the per-frame
track list produced by the tracker.  After the full video has been
decoded and tracked, we walk the per-frame track data and fill gaps.

Because we do it as a post-pass we have both "before" and "after" anchors
for each gap, enabling true linear interpolation rather than just
extrapolation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.tracking.types import BBox


def lerp_bbox(b0: BBox, b1: BBox, t: float) -> BBox:
    """
    Linearly interpolate between two bboxes.

    Parameters
    ----------
    b0, b1:
        Start and end bboxes.
    t:
        Interpolation factor [0, 1].  0 → b0, 1 → b1.
    """
    def lerp(a: int, b: int) -> int:
        return round(a + (b - a) * t)

    return (
        lerp(b0[0], b1[0]),
        lerp(b0[1], b1[1]),
        lerp(b0[2], b1[2]),
        lerp(b0[3], b1[3]),
    )


def fill_gaps(
    frame_bboxes: Dict[int, BBox],
    max_gap: int,
) -> Dict[int, BBox]:
    """
    Fill short gaps in a per-frame bbox dict by linear interpolation.

    Parameters
    ----------
    frame_bboxes:
        Mapping from frame_index → bbox for a single track.
        May have missing keys where the track was not detected.
    max_gap:
        Maximum gap size (frames) to fill.  Larger gaps are left empty.

    Returns
    -------
    Dict[int, BBox]
        A copy of *frame_bboxes* with gaps ≤ *max_gap* filled in.
    """
    if not frame_bboxes:
        return {}

    result: Dict[int, BBox] = dict(frame_bboxes)
    sorted_frames = sorted(frame_bboxes.keys())

    for i in range(len(sorted_frames) - 1):
        f0 = sorted_frames[i]
        f1 = sorted_frames[i + 1]
        gap = f1 - f0 - 1  # number of missing frames
        if gap <= 0 or gap > max_gap:
            continue
        b0 = frame_bboxes[f0]
        b1 = frame_bboxes[f1]
        for step in range(1, gap + 1):
            t = step / (gap + 1)
            result[f0 + step] = lerp_bbox(b0, b1, t)

    return result
