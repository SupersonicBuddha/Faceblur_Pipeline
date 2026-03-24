"""
Tests for FaceTracker instance-scoped track IDs.

Verifies that track IDs are deterministic per tracker instance,
and that two independent instances do not share state.
"""

from __future__ import annotations

import pytest

from src.detectors.base import Detection
from src.tracking.tracker import FaceTracker


def _det(x1: int, y1: int, x2: int, y2: int, conf: float = 0.9) -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf)


class TestTrackerInstanceIds:
    def test_first_track_id_is_one(self):
        tracker = FaceTracker()
        frame = tracker.update(0, [_det(0, 0, 100, 100)])
        assert frame.tracks[0].track_id == 1

    def test_ids_increment_within_instance(self):
        tracker = FaceTracker()
        # Two simultaneous detections → two tracks
        frame = tracker.update(0, [_det(0, 0, 50, 50), _det(200, 200, 300, 300)])
        ids = sorted(t.track_id for t in frame.tracks)
        assert ids == [1, 2]

    def test_two_instances_start_from_one_independently(self):
        """Each tracker instance gets its own ID counter starting at 1."""
        t1 = FaceTracker()
        t2 = FaceTracker()

        f1 = t1.update(0, [_det(0, 0, 100, 100)])
        f2 = t2.update(0, [_det(0, 0, 100, 100)])

        # Both should start at ID 1 — no shared global counter
        assert f1.tracks[0].track_id == 1
        assert f2.tracks[0].track_id == 1

    def test_reset_does_not_reset_id_counter(self):
        """
        After a scene-cut reset, new tracks get fresh IDs continuing
        from where the counter left off.  IDs are never reused.
        """
        tracker = FaceTracker()
        tracker.update(0, [_det(0, 0, 100, 100)])   # track_id=1
        tracker.reset()
        frame = tracker.update(1, [_det(0, 0, 100, 100)])  # new track after reset
        # Should be 2, not 1 (never reuse IDs within an instance)
        assert frame.tracks[0].track_id == 2

    def test_no_module_global_side_effect(self):
        """Creating and destroying many trackers must not affect a fresh instance."""
        # Create and use several trackers
        for _ in range(10):
            t = FaceTracker()
            for i in range(5):
                t.update(i, [_det(i * 10, i * 10, i * 10 + 50, i * 10 + 50)])

        # A brand-new tracker should still start at 1
        fresh = FaceTracker()
        frame = fresh.update(0, [_det(0, 0, 100, 100)])
        assert frame.tracks[0].track_id == 1
