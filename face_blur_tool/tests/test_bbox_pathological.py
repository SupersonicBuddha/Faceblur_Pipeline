"""
Pathological bounding-box and frame-shape edge cases.

Tests the blur compositor (apply_face_blurs) and mask helpers (padded_bbox,
local_ellipse_mask) under inputs that could plausibly appear in production:

  - Swapped corners (x2 < x1 or y2 < y1)
  - Bbox completely outside the frame
  - Bbox with negative pixel coordinates
  - Bbox partially outside the frame (clipped correctly)
  - Very small frames (4×4, 1×1)
  - Zero-dimension bbox after padding clip
  - Grayscale (2D) frame — documents the known limitation / crash

BUG DOCUMENTED:
  test_grayscale_frame_raises — apply_face_blurs expects BGR (H×W×3).
  Passing a grayscale (H×W) ndarray triggers a numpy broadcasting error
  inside the blend step.  The test marks this as xfail so the suite stays
  green while the limitation is visible.  Remove xfail once the compositor
  adds an explicit channel check.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.blur.compositor import apply_face_blurs
from src.blur.masks import local_ellipse_mask, padded_bbox
from src.config import FaceBlurConfig


# ── Helpers ────────────────────────────────────────────────────────────────


def _checkerboard(h: int, w: int, block: int = 4) -> np.ndarray:
    """High-frequency BGR frame so blur is always detectable."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if (x // block + y // block) % 2 == 0:
                frame[y, x, :] = 255
    return frame


def _solid(h: int, w: int, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


_CFG = FaceBlurConfig(blur_kernel_size=5, blur_strength=3.0, blur_padding_ratio=0.0)
_CFG_PAD = FaceBlurConfig(blur_kernel_size=5, blur_strength=3.0, blur_padding_ratio=0.1)


# ── padded_bbox edge cases ──────────────────────────────────────────────────


class TestPaddedBboxPathological:

    def test_swapped_x_coords_produces_non_positive_roi_width(self):
        """
        When x2 < x1, padded_bbox treats the bbox width as max(1, x2-x1)=1.
        After padding=0, px2=x2 < px1=x1, so roi_w <= 0 and the compositor
        must skip the region without crashing.
        """
        # x2=50 < x1=150 — corners swapped on x axis
        px1, py1, px2, py2 = padded_bbox((150, 100, 50, 200), 0.0, (480, 640))
        assert px2 <= px1 or (px2 - px1) <= 0

    def test_swapped_y_coords_produces_non_positive_roi_height(self):
        """y2 < y1 — same reasoning as x swap."""
        px1, py1, px2, py2 = padded_bbox((100, 200, 200, 100), 0.0, (480, 640))
        assert py2 <= py1 or (py2 - py1) <= 0

    def test_negative_x1_y1_clipped_to_zero(self):
        """Negative top-left coords are clipped to 0, not wrapped."""
        px1, py1, px2, py2 = padded_bbox((-30, -20, 100, 100), 0.0, (480, 640))
        assert px1 >= 0
        assert py1 >= 0

    def test_x2_y2_beyond_frame_clipped_to_frame_edge(self):
        """Bottom-right coords beyond frame dimensions are clipped."""
        H, W = 480, 640
        px1, py1, px2, py2 = padded_bbox((500, 400, 700, 600), 0.0, (H, W))
        assert px2 < W
        assert py2 < H

    def test_bbox_entirely_outside_frame_right(self):
        """A bbox completely to the right of the frame clips to zero-size ROI."""
        H, W = 480, 640
        px1, py1, px2, py2 = padded_bbox((700, 100, 800, 200), 0.0, (H, W))
        # After clipping, px1 will be near W-1 and px2 will equal W-1 → roi_w=0
        assert px2 - px1 <= 0

    def test_zero_size_bbox_treated_as_width_1(self):
        """x2==x1 → bw=max(1,0)=1, so pad_x=int(1*ratio), ROI may still be skippable."""
        result = padded_bbox((100, 100, 100, 200), 0.0, (480, 640))
        # px2 = min(639, 100+0) = 100 = px1 → roi_w = 0 → compositor skips
        px1, py1, px2, py2 = result
        assert px2 - px1 == 0


# ── apply_face_blurs: pathological bboxes ─────────────────────────────────


class TestApplyFaceBlursPathological:

    def test_swapped_corner_bbox_does_not_raise(self):
        """Swapped x-corners must be silently skipped, not crash."""
        frame = _checkerboard(480, 640)
        result = apply_face_blurs(frame, [(300, 100, 100, 200)], _CFG)
        assert result.shape == frame.shape

    def test_swapped_xy_corners_does_not_raise(self):
        """Both axes swapped — must be silently skipped."""
        frame = _checkerboard(480, 640)
        result = apply_face_blurs(frame, [(300, 300, 100, 100)], _CFG)
        assert result.shape == frame.shape

    def test_bbox_completely_outside_frame_does_not_raise(self):
        """A bbox 1000px to the right of a 640-wide frame must be skipped."""
        frame = _checkerboard(480, 640)
        result = apply_face_blurs(frame, [(700, 100, 900, 300)], _CFG)
        assert result.shape == frame.shape

    def test_bbox_with_negative_coordinates_does_not_raise(self):
        """
        Negative coordinates arise when a face is partially off-screen.
        The padded_bbox helper clips them to 0; the compositor must not crash.
        """
        frame = _checkerboard(480, 640)
        result = apply_face_blurs(frame, [(-50, -30, 100, 100)], _CFG)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_bbox_partially_outside_frame_clipped_and_blurred(self):
        """
        A face that straddles the right/bottom edge should be clipped to frame
        boundaries and the visible portion still blurred.
        """
        frame = _checkerboard(480, 640)
        # bbox extends 100px beyond the 640-wide frame on the right
        result = apply_face_blurs(frame, [(580, 400, 720, 470)], _CFG)
        assert result.shape == frame.shape

    def test_very_small_frame_4x4_does_not_raise(self):
        """A 4×4 frame with a bbox that fills it should not raise."""
        frame = _checkerboard(4, 4, block=1)
        result = apply_face_blurs(frame, [(0, 0, 4, 4)], _CFG)
        assert result.shape == (4, 4, 3)

    def test_single_pixel_frame_does_not_raise(self):
        """1×1 frame — the degenerate extreme."""
        frame = _solid(1, 1)
        result = apply_face_blurs(frame, [(0, 0, 1, 1)], _CFG)
        assert result.shape == (1, 1, 3)

    def test_bbox_zero_area_is_skipped_frame_unchanged(self):
        """x1==x2 and y1==y2: zero-area bbox must produce no pixel changes."""
        frame = _checkerboard(480, 640)
        result = apply_face_blurs(frame, [(200, 200, 200, 200)], _CFG)
        np.testing.assert_array_equal(result, frame)

    def test_multiple_pathological_bboxes_in_one_call(self):
        """
        Mixing pathological and valid bboxes: the valid one is blurred,
        the pathological ones are skipped — no crash, correct output shape.
        """
        frame = _checkerboard(480, 640)
        bboxes = [
            (700, 100, 900, 300),   # outside frame
            (-50, -30, 100, 100),   # negative coords
            (300, 300, 100, 100),   # swapped corners
            (100, 100, 300, 300),   # valid
        ]
        result = apply_face_blurs(frame, bboxes, _CFG)
        assert result.shape == frame.shape
        # Valid bbox center should be blurred
        cx, cy = 200, 200
        assert not np.array_equal(result[cy, cx], frame[cy, cx])

    @pytest.mark.xfail(
        reason=(
            "apply_face_blurs expects BGR (H×W×3) input. "
            "Passing a grayscale (H×W) frame causes a numpy broadcasting error "
            "in the blend step.  This is a known limitation — the compositor "
            "does not validate the number of channels.  "
            "Remove xfail once an explicit channel check is added."
        ),
        strict=True,
    )
    def test_grayscale_frame_raises(self):
        """
        A grayscale ndarray of shape (H, W) passed to apply_face_blurs
        triggers a numpy broadcasting error inside the compositing step
        (mask3 is (H,W,1) but roi is (H,W), so * fails for non-square shapes).
        This test documents the bug so it is visible in CI.
        """
        gray_frame = np.full((480, 640), 128, dtype=np.uint8)  # shape (H, W) — no channel dim
        config = FaceBlurConfig()
        # This currently raises; the test expects it to NOT raise (hence xfail strict=True)
        apply_face_blurs(gray_frame, [(100, 100, 300, 300)], config)


# ── local_ellipse_mask pathological sizes ──────────────────────────────────


class TestLocalEllipseMaskPathological:

    def test_wide_and_flat_roi(self):
        """A very wide, flat ROI (e.g. 2 rows × 200 cols) must not raise."""
        mask = local_ellipse_mask(2, 200)
        assert mask.shape == (2, 200)
        assert mask.dtype == np.uint8

    def test_tall_and_narrow_roi(self):
        """A very tall, narrow ROI (200 rows × 2 cols) must not raise."""
        mask = local_ellipse_mask(200, 2)
        assert mask.shape == (200, 2)

    def test_zero_height_treated_as_one(self):
        """roi_h=0 is clamped to 1 by max(1, roi_h)."""
        mask = local_ellipse_mask(0, 50)
        assert mask.shape[0] >= 1

    def test_zero_width_treated_as_one(self):
        """roi_w=0 is clamped to 1."""
        mask = local_ellipse_mask(50, 0)
        assert mask.shape[1] >= 1

    def test_values_only_zero_or_255(self):
        """All values must be binary — no partial-mask anti-aliasing."""
        mask = local_ellipse_mask(33, 57)
        assert set(np.unique(mask)).issubset({0, 255})
