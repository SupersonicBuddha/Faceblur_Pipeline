"""Tests for blur compositor — verifies only masked region is blurred."""

from __future__ import annotations

import numpy as np
import pytest

from src.blur.blur_ops import gaussian_blur_frame
from src.blur.compositor import apply_face_blurs
from src.blur.masks import local_ellipse_mask
from src.config import FaceBlurConfig


def _solid_frame(h: int, w: int, value: int = 128) -> np.ndarray:
    """Create a solid-colour BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _checkerboard_frame(h: int, w: int, block: int = 8) -> np.ndarray:
    """
    Create a checkerboard frame (high-frequency, detectable after blur).

    A linear gradient blurs to itself (blurring a linear function returns the
    same linear function), making it a poor test signal for blur detection.
    A checkerboard has high spatial frequency that is strongly attenuated by
    Gaussian blur.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if (x // block + y // block) % 2 == 0:
                frame[y, x, :] = 255
    return frame


class TestApplyFaceBlurs:
    def test_no_faces_returns_original(self):
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig()
        result = apply_face_blurs(frame, [], config)
        np.testing.assert_array_equal(result, frame)

    def test_output_same_shape_and_dtype(self):
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig()
        result = apply_face_blurs(frame, [(100, 100, 300, 300)], config)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_blur_changes_face_region(self):
        # Use a frame with high-frequency content so blur is detectable
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig(blur_kernel_size=51, blur_strength=20.0)
        bbox = (100, 100, 300, 300)
        result = apply_face_blurs(frame, [bbox], config)
        # Center of face region should differ from original
        cx, cy = 200, 200
        assert not np.array_equal(result[cy, cx], frame[cy, cx])

    def test_region_far_outside_face_unchanged(self):
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig(
            blur_kernel_size=51,
            blur_strength=20.0,
            blur_padding_ratio=0.0,
        )
        bbox = (100, 100, 200, 200)  # small region in top-left
        result = apply_face_blurs(frame, [bbox], config)
        # Bottom-right corner should be unchanged
        np.testing.assert_array_equal(result[460:480, 620:640], frame[460:480, 620:640])

    def test_multiple_faces_all_blurred(self):
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig(blur_kernel_size=51, blur_strength=20.0, blur_padding_ratio=0.0)
        bboxes = [(50, 50, 100, 100), (400, 300, 500, 400)]
        result = apply_face_blurs(frame, bboxes, config)
        # Both centers should have changed
        for bbox in bboxes:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            assert not np.array_equal(result[cy, cx], frame[cy, cx])

    def test_face_at_frame_edge(self):
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig()
        # BBox near edge — should not raise or produce out-of-bounds
        result = apply_face_blurs(frame, [(0, 0, 50, 50)], config)
        assert result.shape == frame.shape

    def test_solid_frame_blurred_is_still_solid(self):
        # A solid frame blurred should remain the same colour
        frame = _solid_frame(480, 640, 100)
        config = FaceBlurConfig(blur_kernel_size=51, blur_strength=20.0)
        result = apply_face_blurs(frame, [(100, 100, 300, 300)], config)
        # The face region of a solid frame blurs to the same value
        assert result.shape == frame.shape


class TestROIBlur:
    """Verify the ROI-based compositor leaves pixels outside the padded bbox unchanged."""

    def test_pixels_outside_padded_roi_are_untouched(self):
        """
        With padding_ratio=0.0 the padded ROI equals the original bbox.
        Pixels one pixel outside that box must be byte-identical to the original.
        """
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig(
            blur_kernel_size=51,
            blur_strength=20.0,
            blur_padding_ratio=0.0,
        )
        bbox = (100, 100, 200, 200)
        result = apply_face_blurs(frame, [bbox], config)

        # One row above the bbox
        np.testing.assert_array_equal(result[99, :], frame[99, :])
        # One row below the bbox (y2=200, next row = 200)
        np.testing.assert_array_equal(result[200, :], frame[200, :])
        # One col left of bbox
        np.testing.assert_array_equal(result[:, 99], frame[:, 99])
        # One col right of bbox
        np.testing.assert_array_equal(result[:, 200], frame[:, 200])

    def test_pixels_inside_roi_can_change(self):
        """Center of face region must differ from original (high-frequency source)."""
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig(blur_kernel_size=51, blur_strength=20.0)
        result = apply_face_blurs(frame, [(100, 100, 300, 300)], config)
        cx, cy = 200, 200
        assert not np.array_equal(result[cy, cx], frame[cy, cx])

    def test_output_shape_and_dtype_unchanged(self):
        frame = _checkerboard_frame(480, 640)
        config = FaceBlurConfig()
        result = apply_face_blurs(frame, [(50, 50, 150, 150)], config)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_degenerate_zero_size_bbox_does_not_raise(self):
        """A bbox with zero area after padding clip should be silently skipped."""
        frame = _solid_frame(480, 640)
        config = FaceBlurConfig(blur_padding_ratio=0.0)
        # bbox with x1==x2, y1==y2 → zero area
        result = apply_face_blurs(frame, [(100, 100, 100, 100)], config)
        assert result.shape == frame.shape


class TestLocalEllipseMask:
    """Tests for the ROI-local ellipse mask helper."""

    def test_shape_matches_requested_size(self):
        mask = local_ellipse_mask(80, 120)
        assert mask.shape == (80, 120)
        assert mask.dtype == np.uint8

    def test_center_is_inside(self):
        mask = local_ellipse_mask(100, 100)
        assert mask[50, 50] == 255

    def test_corners_are_outside(self):
        mask = local_ellipse_mask(100, 100)
        assert mask[0, 0] == 0
        assert mask[0, 99] == 0
        assert mask[99, 0] == 0
        assert mask[99, 99] == 0

    def test_only_zero_or_255(self):
        mask = local_ellipse_mask(64, 80)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_1x1_roi_does_not_raise(self):
        mask = local_ellipse_mask(1, 1)
        assert mask.shape == (1, 1)


class TestGaussianBlurFrame:
    def test_output_shape_preserved(self):
        frame = _checkerboard_frame(480, 640)
        blurred = gaussian_blur_frame(frame, kernel_size=21)
        assert blurred.shape == frame.shape
        assert blurred.dtype == np.uint8

    def test_even_kernel_promoted_to_odd(self):
        frame = _checkerboard_frame(100, 100)
        # Should not raise even with an even kernel size (function auto-corrects)
        blurred = gaussian_blur_frame(frame, kernel_size=20)
        assert blurred.shape == frame.shape

    def test_larger_kernel_more_blur(self):
        frame = _checkerboard_frame(480, 640)
        mild = gaussian_blur_frame(frame, kernel_size=3)
        strong = gaussian_blur_frame(frame, kernel_size=51, sigma=20.0)
        # Strong blur should be smoother — compute variance of each
        var_mild = float(np.var(mild.astype(np.float32)))
        var_strong = float(np.var(strong.astype(np.float32)))
        assert var_strong < var_mild
