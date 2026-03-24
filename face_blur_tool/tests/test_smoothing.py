"""Tests for bounding box temporal smoothing."""

from __future__ import annotations

import pytest
from src.tracking.smoothing import BBoxSmoother, _mean_bbox


class TestMeanBBox:
    def test_single_bbox(self):
        result = _mean_bbox([(10, 20, 30, 40)])
        assert result == (10, 20, 30, 40)

    def test_two_identical(self):
        result = _mean_bbox([(10, 10, 50, 50), (10, 10, 50, 50)])
        assert result == (10, 10, 50, 50)

    def test_average_of_two(self):
        result = _mean_bbox([(0, 0, 100, 100), (100, 100, 200, 200)])
        assert result == (50, 50, 150, 150)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _mean_bbox([])


class TestBBoxSmoother:
    def test_single_observation_passthrough(self):
        s = BBoxSmoother(window=5)
        bbox = (10, 20, 100, 200)
        result = s.smooth(track_id=1, bbox=bbox)
        assert result == bbox

    def test_smoothed_over_window(self):
        s = BBoxSmoother(window=3)
        bboxes = [(0, 0, 100, 100), (10, 10, 110, 110), (20, 20, 120, 120)]
        for i, bb in enumerate(bboxes):
            last = s.smooth(track_id=1, bbox=bb)
        # After 3 observations, should be average of all 3
        expected = _mean_bbox(bboxes)
        assert last == expected

    def test_window_is_a_rolling_mean(self):
        s = BBoxSmoother(window=2)
        s.smooth(1, (0, 0, 100, 100))
        s.smooth(1, (0, 0, 100, 100))
        result = s.smooth(1, (100, 100, 200, 200))
        # Window = 2: only last 2 observations matter
        expected = _mean_bbox([(0, 0, 100, 100), (100, 100, 200, 200)])
        assert result == expected

    def test_independent_tracks(self):
        s = BBoxSmoother(window=5)
        r1 = s.smooth(track_id=1, bbox=(0, 0, 50, 50))
        r2 = s.smooth(track_id=2, bbox=(100, 100, 200, 200))
        assert r1 == (0, 0, 50, 50)
        assert r2 == (100, 100, 200, 200)

    def test_reset_track(self):
        s = BBoxSmoother(window=5)
        for _ in range(5):
            s.smooth(1, (0, 0, 100, 100))
        s.reset_track(1)
        # After reset, new observation should not be averaged with history
        result = s.smooth(1, (200, 200, 400, 400))
        assert result == (200, 200, 400, 400)

    def test_reset_all(self):
        s = BBoxSmoother(window=5)
        s.smooth(1, (0, 0, 100, 100))
        s.smooth(2, (50, 50, 150, 150))
        s.reset_all()
        r = s.smooth(1, (10, 10, 20, 20))
        assert r == (10, 10, 20, 20)

    def test_remove_stale_tracks(self):
        s = BBoxSmoother(window=5)
        s.smooth(1, (0, 0, 10, 10))
        s.smooth(2, (20, 20, 30, 30))
        s.remove_stale_tracks([1])  # remove 2
        # Track 2 should be gone; track 1 still has its history
        r1 = s.smooth(1, (0, 0, 10, 10))
        assert r1 != (20, 20, 30, 30)

    def test_window_one_no_smoothing(self):
        s = BBoxSmoother(window=1)
        s.smooth(1, (0, 0, 100, 100))
        result = s.smooth(1, (50, 50, 60, 60))
        assert result == (50, 50, 60, 60)
