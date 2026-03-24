"""Tests for bbox interpolation across short detection gaps."""

from __future__ import annotations

import pytest
from src.tracking.interpolation import fill_gaps, lerp_bbox


class TestLerpBbox:
    def test_t0_returns_b0(self):
        assert lerp_bbox((0, 0, 100, 100), (200, 200, 400, 400), 0.0) == (0, 0, 100, 100)

    def test_t1_returns_b1(self):
        assert lerp_bbox((0, 0, 100, 100), (200, 200, 400, 400), 1.0) == (200, 200, 400, 400)

    def test_midpoint(self):
        result = lerp_bbox((0, 0, 100, 100), (100, 100, 200, 200), 0.5)
        assert result == (50, 50, 150, 150)

    def test_rounding(self):
        # Should round to nearest int
        result = lerp_bbox((0, 0, 10, 10), (0, 0, 11, 11), 0.5)
        assert all(isinstance(v, int) for v in result)


class TestFillGaps:
    def test_no_gaps_unchanged(self):
        data = {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (2, 2, 12, 12)}
        result = fill_gaps(data, max_gap=5)
        assert result[0] == data[0]
        assert result[1] == data[1]
        assert result[2] == data[2]

    def test_fills_single_gap(self):
        data = {0: (0, 0, 100, 100), 2: (100, 100, 200, 200)}
        result = fill_gaps(data, max_gap=3)
        assert 1 in result
        # Frame 1 should be midpoint
        assert result[1] == lerp_bbox((0, 0, 100, 100), (100, 100, 200, 200), 0.5)

    def test_gap_too_large_not_filled(self):
        data = {0: (0, 0, 10, 10), 10: (100, 100, 200, 200)}
        result = fill_gaps(data, max_gap=5)
        # Gap of 9 > max_gap of 5 — should not be filled
        for i in range(1, 10):
            assert i not in result

    def test_fills_gap_of_exact_max_gap(self):
        data = {0: (0, 0, 10, 10), 4: (40, 40, 50, 50)}  # gap = 3
        result = fill_gaps(data, max_gap=3)
        for i in range(1, 4):
            assert i in result

    def test_empty_input(self):
        assert fill_gaps({}, max_gap=5) == {}

    def test_single_frame_unchanged(self):
        data = {5: (10, 10, 20, 20)}
        result = fill_gaps(data, max_gap=5)
        assert result == data

    def test_multiple_gaps_all_filled(self):
        data = {0: (0, 0, 10, 10), 2: (20, 20, 30, 30), 4: (40, 40, 50, 50)}
        result = fill_gaps(data, max_gap=3)
        assert 1 in result
        assert 3 in result

    def test_original_not_mutated(self):
        data = {0: (0, 0, 10, 10), 2: (20, 20, 30, 30)}
        original_keys = set(data.keys())
        fill_gaps(data, max_gap=5)
        assert set(data.keys()) == original_keys
