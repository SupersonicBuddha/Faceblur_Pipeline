"""Tests for config loading, defaults, and derived properties."""

import os
import pytest
from src.config import FaceBlurConfig, _parse_gcs_prefix


class TestParseGcsPrefix:
    def test_basic(self):
        bucket, prefix = _parse_gcs_prefix("gs://my-bucket/some/prefix")
        assert bucket == "my-bucket"
        assert prefix == "some/prefix"

    def test_no_prefix(self):
        bucket, prefix = _parse_gcs_prefix("gs://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            _parse_gcs_prefix("s3://bucket/prefix")


class TestFaceBlurConfigDefaults:
    def test_default_input_prefix(self):
        cfg = FaceBlurConfig()
        assert cfg.input_prefix == "gs://tbd-qa/for_fb"

    def test_default_output_prefix(self):
        cfg = FaceBlurConfig()
        assert cfg.output_prefix == "gs://tbd-qa/post_fb"

    def test_default_manifest_path(self):
        cfg = FaceBlurConfig()
        assert cfg.manifest_path == "gs://tbd-qa/post_fb/processed_manifest.jsonl"

    def test_derived_input_bucket(self):
        cfg = FaceBlurConfig()
        assert cfg.input_bucket == "tbd-qa"

    def test_derived_input_blob_prefix(self):
        cfg = FaceBlurConfig()
        assert cfg.input_blob_prefix == "for_fb"

    def test_derived_output_bucket(self):
        cfg = FaceBlurConfig()
        assert cfg.output_bucket == "tbd-qa"

    def test_derived_manifest_bucket(self):
        cfg = FaceBlurConfig()
        assert cfg.manifest_bucket == "tbd-qa"

    def test_derived_manifest_blob(self):
        cfg = FaceBlurConfig()
        assert cfg.manifest_blob == "post_fb/processed_manifest.jsonl"

    def test_effective_blur_kernel_auto(self):
        cfg = FaceBlurConfig(blur_kernel_size=0, blur_strength=25.0)
        k = cfg.effective_blur_kernel
        assert k % 2 == 1  # must be odd
        assert k >= 3

    def test_effective_blur_kernel_explicit(self):
        cfg = FaceBlurConfig(blur_kernel_size=51)
        assert cfg.effective_blur_kernel == 51

    def test_effective_blur_kernel_even_becomes_odd(self):
        cfg = FaceBlurConfig(blur_kernel_size=50)
        assert cfg.effective_blur_kernel == 51


class TestFaceBlurConfigValidation:
    def test_invalid_input_prefix_raises(self):
        with pytest.raises(ValueError, match="input_prefix"):
            FaceBlurConfig(input_prefix="not-a-gcs-uri")

    def test_invalid_output_prefix_raises(self):
        with pytest.raises(ValueError, match="output_prefix"):
            FaceBlurConfig(output_prefix="s3://wrong-scheme/prefix")

    def test_invalid_manifest_path_raises(self):
        with pytest.raises(ValueError, match="manifest_path"):
            FaceBlurConfig(manifest_path="/local/path.jsonl")

    def test_detection_interval_zero_raises(self):
        with pytest.raises(ValueError, match="detection_interval"):
            FaceBlurConfig(detection_interval=0)

    def test_detection_interval_negative_raises(self):
        with pytest.raises(ValueError, match="detection_interval"):
            FaceBlurConfig(detection_interval=-1)

    def test_confidence_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="detector_confidence_threshold"):
            FaceBlurConfig(detector_confidence_threshold=1.1)

    def test_confidence_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="detector_confidence_threshold"):
            FaceBlurConfig(detector_confidence_threshold=-0.1)

    def test_blur_padding_negative_raises(self):
        with pytest.raises(ValueError, match="blur_padding_ratio"):
            FaceBlurConfig(blur_padding_ratio=-0.1)

    def test_tracker_iou_above_one_raises(self):
        with pytest.raises(ValueError, match="tracker_iou_threshold"):
            FaceBlurConfig(tracker_iou_threshold=1.5)

    def test_mux_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="ffmpeg_mux_timeout_sec"):
            FaceBlurConfig(ffmpeg_mux_timeout_sec=0)

    def test_retry_count_negative_raises(self):
        with pytest.raises(ValueError, match="retry_count"):
            FaceBlurConfig(retry_count=-1)

    def test_smoothing_window_zero_raises(self):
        with pytest.raises(ValueError, match="smoothing_window"):
            FaceBlurConfig(smoothing_window=0)

    def test_interpolation_max_gap_negative_raises(self):
        with pytest.raises(ValueError, match="interpolation_max_gap"):
            FaceBlurConfig(interpolation_max_gap=-1)

    def test_multiple_errors_reported_together(self):
        """All invalid fields must be reported in one ValueError, not just the first."""
        with pytest.raises(ValueError) as exc_info:
            FaceBlurConfig(detection_interval=0, retry_count=-1)
        msg = str(exc_info.value)
        assert "detection_interval" in msg
        assert "retry_count" in msg

    def test_valid_defaults_do_not_raise(self):
        """Default config must always be valid."""
        FaceBlurConfig()  # must not raise

    def test_boundary_values_are_valid(self):
        """Boundary-inclusive values must be accepted."""
        cfg = FaceBlurConfig(
            detection_interval=1,
            detector_confidence_threshold=0.0,
            blur_padding_ratio=0.0,
            tracker_iou_threshold=0.0,
            ffmpeg_mux_timeout_sec=1,
            retry_count=0,
            smoothing_window=1,
            interpolation_max_gap=0,
        )
        assert cfg.detection_interval == 1


class TestFaceBlurConfigEnvVars:
    def test_env_override_input_prefix(self, monkeypatch):
        monkeypatch.setenv("FB_INPUT_PREFIX", "gs://other-bucket/vids")
        cfg = FaceBlurConfig()
        assert cfg.input_prefix == "gs://other-bucket/vids"

    def test_env_override_detector_type(self, monkeypatch):
        monkeypatch.setenv("FB_DETECTOR_TYPE", "mediapipe")
        cfg = FaceBlurConfig()
        assert cfg.detector_type == "mediapipe"
