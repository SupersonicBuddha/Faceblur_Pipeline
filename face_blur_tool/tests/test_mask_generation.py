"""Tests for ellipse mask generation and clipping."""

from __future__ import annotations

import numpy as np
import pytest

from src.blur.masks import make_ellipse_mask, padded_bbox


class TestMakeEllipseMask:
    def test_shape_matches_frame(self):
        mask = make_ellipse_mask((480, 640), (100, 100, 300, 300))
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_interior_is_255(self):
        mask = make_ellipse_mask((480, 640), (100, 100, 300, 300))
        # Center of bbox should be inside the ellipse
        cx, cy = 200, 200
        assert mask[cy, cx] == 255

    def test_corner_is_0(self):
        mask = make_ellipse_mask((480, 640), (100, 100, 300, 300), padding_ratio=0.0)
        # Top-left corner should be outside
        assert mask[0, 0] == 0

    def test_all_values_are_0_or_255(self):
        mask = make_ellipse_mask((480, 640), (100, 100, 300, 300))
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})

    def test_face_at_frame_edge_does_not_raise(self):
        # BBox partially outside frame — should clip cleanly
        mask = make_ellipse_mask((480, 640), (-50, -50, 100, 100))
        assert mask.shape == (480, 640)

    def test_full_frame_bbox(self):
        mask = make_ellipse_mask((100, 100), (0, 0, 100, 100))
        assert mask.shape == (100, 100)
        # At least the center should be inside
        assert mask[50, 50] == 255

    def test_zero_padding_smaller_than_padded(self):
        m_no_pad = make_ellipse_mask((480, 640), (100, 100, 200, 200), padding_ratio=0.0)
        m_padded = make_ellipse_mask((480, 640), (100, 100, 200, 200), padding_ratio=0.35)
        # Padded mask should cover more pixels
        assert m_padded.sum() >= m_no_pad.sum()


class TestPaddedBbox:
    def test_basic_expansion(self):
        bbox = (100, 100, 200, 200)
        padded = padded_bbox(bbox, padding_ratio=0.1, frame_shape=(480, 640))
        x1, y1, x2, y2 = padded
        assert x1 < 100
        assert y1 < 100
        assert x2 > 200
        assert y2 > 200

    def test_clips_to_frame(self):
        # BBox at top-left edge
        bbox = (0, 0, 50, 50)
        padded = padded_bbox(bbox, padding_ratio=0.5, frame_shape=(480, 640))
        assert padded[0] >= 0
        assert padded[1] >= 0

    def test_clips_to_frame_bottom_right(self):
        bbox = (590, 430, 640, 480)
        padded = padded_bbox(bbox, padding_ratio=0.5, frame_shape=(480, 640))
        assert padded[2] <= 639
        assert padded[3] <= 479
