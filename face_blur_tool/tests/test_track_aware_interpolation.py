"""
Tests for track-aware (identity-safe) interpolation behaviour.

These tests verify that:
1. Per-track interpolation does not cross face identities.
2. Multi-face scenes with different face counts on each side of a gap do not
   silently lose eligible blur regions.
3. Tracks present on only one side of a gap are preserved in their known
   frames and are not extended across the gap.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from src.tracking.interpolation import fill_gaps, lerp_bbox
from src.tracking.types import BBox


# ── Helpers ────────────────────────────────────────────────────────────────


def _compose_per_frame(
    per_track_frames: Dict[int, Dict[int, BBox]],
    max_gap: int,
) -> Dict[int, List[BBox]]:
    """
    Mirror the logic in process_video._run_detection_and_blur's
    interpolation post-pass.  Useful for testing the end-to-end
    composition without spinning up a full video.
    """
    from collections import defaultdict

    per_frame: Dict[int, List[BBox]] = defaultdict(list)
    for tid, frames in per_track_frames.items():
        filled = fill_gaps(frames, max_gap=max_gap)
        for frame_idx, bbox in filled.items():
            per_frame[frame_idx].append(bbox)
    return dict(per_frame)


# ── Identity-safe interpolation ────────────────────────────────────────────


class TestTrackAwareInterpolation:
    def test_single_track_gap_is_filled(self):
        """A gap of 1 frame within a single track should be interpolated."""
        per_track = {
            1: {0: (0, 0, 100, 100), 2: (20, 20, 120, 120)},
        }
        result = _compose_per_frame(per_track, max_gap=3)

        assert 1 in result
        assert len(result[1]) == 1
        expected = lerp_bbox((0, 0, 100, 100), (20, 20, 120, 120), 0.5)
        assert result[1][0] == expected

    def test_two_tracks_no_crossing(self):
        """
        Two tracks with non-overlapping positions.  Interpolated frame must
        contain both bboxes from their respective tracks — not any cross-
        track mixture.
        """
        # track 1: left side of frame
        # track 2: right side of frame
        per_track = {
            1: {0: (0, 0, 100, 100), 2: (10, 10, 110, 110)},
            2: {0: (400, 300, 500, 400), 2: (390, 290, 490, 390)},
        }
        result = _compose_per_frame(per_track, max_gap=3)

        assert 1 in result
        bboxes = result[1]
        assert len(bboxes) == 2, "Both tracks should produce a bbox at frame 1"

        # track 1 interpolated midpoint
        t1_expected = lerp_bbox((0, 0, 100, 100), (10, 10, 110, 110), 0.5)
        # track 2 interpolated midpoint
        t2_expected = lerp_bbox((400, 300, 500, 400), (390, 290, 490, 390), 0.5)

        assert t1_expected in bboxes
        assert t2_expected in bboxes

    def test_reordered_faces_no_cross_identity(self):
        """
        Two tracks where at frame 2 the bboxes have swapped left-right
        positions relative to frame 0.  Each track should still be
        interpolated along its own trajectory.
        """
        # track 1 starts left, moves right
        # track 2 starts right, moves left
        per_track = {
            1: {0: (0, 100, 100, 200), 4: (300, 100, 400, 200)},
            2: {0: (300, 100, 400, 200), 4: (0, 100, 100, 200)},
        }
        result = _compose_per_frame(per_track, max_gap=5)

        # Both tracks should contribute to intermediate frames
        assert 2 in result
        assert len(result[2]) == 2

        # track 1 at frame 2 should be between (0,100,100,200) and (300,100,400,200)
        t1_mid = lerp_bbox((0, 100, 100, 200), (300, 100, 400, 200), 0.5)
        # track 2 at frame 2 should be between (300,100,400,200) and (0,100,100,200)
        t2_mid = lerp_bbox((300, 100, 400, 200), (0, 100, 100, 200), 0.5)

        assert t1_mid in result[2]
        assert t2_mid in result[2]

    def test_unmatched_track_appears_only_in_known_frames(self):
        """
        Track 2 disappears before the gap and track 3 appears after the gap.
        Neither should be extended across the gap — only the frames where
        each was detected should contain that track's bbox.
        """
        per_track = {
            1: {0: (10, 10, 90, 90), 4: (20, 20, 100, 100)},  # persists across gap
            2: {0: (200, 200, 300, 300)},                       # only at frame 0
            3: {4: (400, 400, 500, 500)},                       # only at frame 4
        }
        result = _compose_per_frame(per_track, max_gap=5)

        # Frame 0: track 1 + track 2 = 2 bboxes
        assert len(result[0]) == 2
        assert (200, 200, 300, 300) in result[0]

        # Frame 4: track 1 + track 3 = 2 bboxes
        assert len(result[4]) == 2
        assert (400, 400, 500, 500) in result[4]

        # Frames 1–3: only track 1 should be present (gap-filled)
        for f in [1, 2, 3]:
            assert f in result, f"Frame {f} should have track 1's interpolated bbox"
            assert len(result[f]) == 1, (
                f"Frame {f} should contain only track 1, "
                f"but got {len(result[f])} bboxes"
            )

    def test_different_face_counts_both_sides_no_truncation(self):
        """
        Frame 0 has 2 faces; frame 4 has 1 face.
        The face present only on frame 0 should NOT be extended into the gap
        (no carry-forward beyond its known frames), but it should still appear
        at frame 0.  The single track present on both sides should be
        interpolated across the gap.
        """
        per_track = {
            1: {0: (10, 10, 90, 90), 4: (20, 20, 100, 100)},   # both sides
            2: {0: (200, 200, 300, 300)},                        # only frame 0
        }
        result = _compose_per_frame(per_track, max_gap=5)

        # Frame 0 has both tracks
        assert len(result[0]) == 2

        # Gap frames 1–3: only track 1 should be present
        for f in [1, 2, 3]:
            assert f in result
            assert len(result[f]) == 1

        # Track 2 bbox must NOT appear in gap frames
        for f in [1, 2, 3]:
            assert (200, 200, 300, 300) not in result[f]

    def test_gap_too_large_not_filled(self):
        """Gaps larger than max_gap must not be filled for any track."""
        per_track = {
            1: {0: (0, 0, 100, 100), 20: (50, 50, 150, 150)},
        }
        result = _compose_per_frame(per_track, max_gap=5)

        for f in range(1, 20):
            assert f not in result, f"Frame {f} should not be filled (gap too large)"

    def test_empty_tracks_produces_empty_result(self):
        result = _compose_per_frame({}, max_gap=5)
        assert result == {}

    def test_no_gap_tracks_unchanged(self):
        """Consecutive frames with no gaps should pass through as-is."""
        per_track = {
            1: {0: (0, 0, 50, 50), 1: (5, 5, 55, 55), 2: (10, 10, 60, 60)},
        }
        result = _compose_per_frame(per_track, max_gap=3)
        assert result[0] == [(0, 0, 50, 50)]
        assert result[1] == [(5, 5, 55, 55)]
        assert result[2] == [(10, 10, 60, 60)]
