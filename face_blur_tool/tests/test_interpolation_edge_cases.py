"""
Edge-case tests for bbox interpolation (fill_gaps / lerp_bbox).

Covers gaps not hit by test_interpolation.py:
- max_gap=0 must fill nothing (the "disable interpolation" knob)
- Negative-coordinate bboxes are interpolated correctly (no sign flip)
- Frame indices starting far from 0 (e.g. frame 1000–1010)
- Identical start/end boxes produce constant interpolated output
- lerp_bbox with t values at the exact boundary 0.0 and 1.0
- fill_gaps preserves the original anchor frames unchanged
- gap of exactly 1 frame is filled when max_gap >= 1
"""

from __future__ import annotations

import pytest
from src.tracking.interpolation import fill_gaps, lerp_bbox


class TestFillGapsEdgeCases:

    def test_max_gap_zero_fills_nothing(self):
        """
        max_gap=0 means "do not interpolate any gaps".
        This is the configuration value when interpolation is disabled.
        A gap of 1 (the minimum possible gap) must NOT be filled.
        """
        data = {0: (0, 0, 10, 10), 2: (20, 20, 30, 30)}  # gap = 1
        result = fill_gaps(data, max_gap=0)
        assert 1 not in result
        assert set(result.keys()) == {0, 2}

    def test_gap_of_one_filled_when_max_gap_is_one(self):
        """A single missing frame is filled when max_gap=1."""
        data = {0: (0, 0, 10, 10), 2: (20, 20, 30, 30)}
        result = fill_gaps(data, max_gap=1)
        assert 1 in result

    def test_large_frame_indices_filled_correctly(self):
        """
        Interpolation must work regardless of how large the frame indices are.
        A gap between frames 1000 and 1002 should fill frame 1001.
        """
        data = {1000: (0, 0, 10, 10), 1002: (20, 20, 30, 30)}
        result = fill_gaps(data, max_gap=5)
        assert 1001 in result
        assert result[1001] == lerp_bbox((0, 0, 10, 10), (20, 20, 30, 30), 0.5)

    def test_negative_coordinate_bboxes_interpolate_correctly(self):
        """
        Bbox coordinates can be negative (e.g. partially off-screen faces in
        wide-angle footage, or before clipping in tracker output).
        Interpolation must not change the sign or magnitude unexpectedly.
        """
        b0 = (-10, -5, 50, 60)
        b1 = (10, 5, 70, 80)
        data = {0: b0, 2: b1}
        result = fill_gaps(data, max_gap=3)
        assert 1 in result
        expected = lerp_bbox(b0, b1, 0.5)
        assert result[1] == expected

    def test_identical_start_end_bboxes_give_constant_fill(self):
        """
        If a face is stationary across a gap, all interpolated frames
        should be identical to the anchor bboxes.
        """
        bbox = (100, 100, 200, 200)
        data = {0: bbox, 5: bbox}
        result = fill_gaps(data, max_gap=10)
        for i in range(1, 5):
            assert result[i] == bbox, f"Frame {i} should equal anchor bbox"

    def test_anchor_frames_not_changed_by_fill(self):
        """
        fill_gaps must never overwrite or alter the original anchor frames.
        """
        b0 = (0, 0, 10, 10)
        b1 = (100, 100, 200, 200)
        data = {0: b0, 4: b1}
        result = fill_gaps(data, max_gap=5)
        assert result[0] == b0
        assert result[4] == b1

    def test_gap_exactly_at_max_gap_boundary_is_filled(self):
        """A gap of exactly max_gap frames must be filled (boundary inclusive)."""
        data = {0: (0, 0, 10, 10), 4: (40, 40, 50, 50)}  # gap = 3
        result = fill_gaps(data, max_gap=3)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_gap_one_over_max_gap_is_not_filled(self):
        """A gap of max_gap+1 must not be filled."""
        data = {0: (0, 0, 10, 10), 5: (50, 50, 60, 60)}  # gap = 4
        result = fill_gaps(data, max_gap=3)
        for i in range(1, 5):
            assert i not in result

    def test_multiple_consecutive_gaps_with_max_gap_zero(self):
        """
        Even with many gaps adjacent to each other, max_gap=0 prevents
        all filling.
        """
        data = {i * 2: (i * 10, 0, i * 10 + 10, 10) for i in range(5)}
        result = fill_gaps(data, max_gap=0)
        assert set(result.keys()) == set(data.keys())

    def test_unsorted_input_dict_still_fills_correctly(self):
        """
        fill_gaps sorts the keys internally.  Input in non-sorted order
        must still produce correct interpolation.
        """
        data = {4: (40, 40, 50, 50), 0: (0, 0, 10, 10), 2: (20, 20, 30, 30)}
        result = fill_gaps(data, max_gap=3)
        assert 1 in result
        assert 3 in result


class TestLerpBboxEdgeCases:

    def test_t_zero_is_exactly_b0(self):
        b0, b1 = (10, 20, 30, 40), (100, 200, 300, 400)
        assert lerp_bbox(b0, b1, 0.0) == b0

    def test_t_one_is_exactly_b1(self):
        b0, b1 = (10, 20, 30, 40), (100, 200, 300, 400)
        assert lerp_bbox(b0, b1, 1.0) == b1

    def test_negative_coordinates_preserved(self):
        """Negative coords must not be silently zeroed."""
        b0 = (-20, -10, 0, 0)
        b1 = (0, 0, 20, 10)
        mid = lerp_bbox(b0, b1, 0.5)
        assert mid[0] == -10  # round((-20+0)/2)
        assert mid[1] == -5   # round((-10+0)/2)

    def test_output_is_tuple_of_ints(self):
        result = lerp_bbox((0, 0, 100, 100), (50, 50, 150, 150), 0.3)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_identical_bboxes_return_same_regardless_of_t(self):
        bbox = (50, 60, 150, 160)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert lerp_bbox(bbox, bbox, t) == bbox
