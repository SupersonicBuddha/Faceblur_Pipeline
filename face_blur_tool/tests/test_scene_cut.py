"""Tests for scene cut detection and tracker reset on cut."""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.scene_cut import SceneCutDetector, _histogram_delta, _compute_histogram


class TestSceneCutDetector:
    def test_first_frame_never_cut(self):
        det = SceneCutDetector(threshold=0.1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert det.is_cut(frame) is False

    def test_identical_frames_no_cut(self):
        det = SceneCutDetector(threshold=0.40)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        det.is_cut(frame)  # prime
        assert det.is_cut(frame.copy()) is False

    def test_completely_different_frames_is_cut(self):
        det = SceneCutDetector(threshold=0.40)
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        det.is_cut(black)  # prime
        assert det.is_cut(white) is True

    def test_reset_clears_history(self):
        det = SceneCutDetector(threshold=0.40)
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        det.is_cut(black)  # prime with black
        det.reset()
        # After reset, first frame call should never return cut
        assert det.is_cut(white) is False

    def test_threshold_respected(self):
        det_tight = SceneCutDetector(threshold=0.01)
        det_loose = SceneCutDetector(threshold=0.99)

        frame_a = np.random.randint(0, 128, (480, 640, 3), dtype=np.uint8)
        frame_b = np.random.randint(128, 256, (480, 640, 3), dtype=np.uint8)

        det_tight.is_cut(frame_a)
        det_loose.is_cut(frame_a)

        cut_tight = det_tight.is_cut(frame_b)
        cut_loose = det_loose.is_cut(frame_b)

        # Tight threshold detects more cuts
        assert cut_tight >= cut_loose


class TestHistogramDelta:
    def test_identical_histograms_zero_delta(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h = _compute_histogram(frame)
        assert _histogram_delta(h, h) == pytest.approx(0.0)

    def test_black_vs_white_large_delta(self):
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        h1 = _compute_histogram(black)
        h2 = _compute_histogram(white)
        delta = _histogram_delta(h1, h2)
        assert delta > 0.5

    def test_delta_range_zero_to_one(self):
        frame_a = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        h1 = _compute_histogram(frame_a)
        h2 = _compute_histogram(frame_b)
        delta = _histogram_delta(h1, h2)
        assert 0.0 <= delta <= 1.0
