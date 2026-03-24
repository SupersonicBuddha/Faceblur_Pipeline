"""Tests for output GCS path derivation."""

import pytest
from src.utils.paths import build_output_gcs_path, gcs_uri_to_bucket_and_blob


class TestBuildOutputGcsPath:
    def test_basic_mp4(self):
        out = build_output_gcs_path(
            "gs://tbd-qa/for_fb/foo/bar/video1.mp4",
            "gs://tbd-qa/for_fb",
            "gs://tbd-qa/post_fb",
        )
        assert out == "gs://tbd-qa/post_fb/foo/bar/video1_blurred.mp4"

    def test_basic_mov(self):
        out = build_output_gcs_path(
            "gs://tbd-qa/for_fb/clip.mov",
            "gs://tbd-qa/for_fb",
            "gs://tbd-qa/post_fb",
        )
        assert out == "gs://tbd-qa/post_fb/clip_blurred.mov"

    def test_nested_path(self):
        out = build_output_gcs_path(
            "gs://tbd-qa/for_fb/2024/05/session1/raw.mp4",
            "gs://tbd-qa/for_fb",
            "gs://tbd-qa/post_fb",
        )
        assert out == "gs://tbd-qa/post_fb/2024/05/session1/raw_blurred.mp4"

    def test_preserves_extension_case(self):
        # Extension is part of the original name; we preserve it as-is
        out = build_output_gcs_path(
            "gs://tbd-qa/for_fb/Video.MP4",
            "gs://tbd-qa/for_fb",
            "gs://tbd-qa/post_fb",
        )
        assert out.endswith("_blurred.MP4")

    def test_trailing_slash_on_prefix_ignored(self):
        out = build_output_gcs_path(
            "gs://tbd-qa/for_fb/v.mp4",
            "gs://tbd-qa/for_fb/",
            "gs://tbd-qa/post_fb/",
        )
        assert out == "gs://tbd-qa/post_fb/v_blurred.mp4"

    def test_source_not_under_prefix_raises(self):
        with pytest.raises(ValueError):
            build_output_gcs_path(
                "gs://other-bucket/for_fb/v.mp4",
                "gs://tbd-qa/for_fb",
                "gs://tbd-qa/post_fb",
            )

    def test_different_buckets_input_output(self):
        out = build_output_gcs_path(
            "gs://input-bucket/raw/v.mp4",
            "gs://input-bucket/raw",
            "gs://output-bucket/processed",
        )
        assert out == "gs://output-bucket/processed/v_blurred.mp4"


class TestGcsUriParse:
    def test_standard(self):
        bucket, blob = gcs_uri_to_bucket_and_blob("gs://my-bucket/path/to/blob.mp4")
        assert bucket == "my-bucket"
        assert blob == "path/to/blob.mp4"

    def test_root(self):
        bucket, blob = gcs_uri_to_bucket_and_blob("gs://my-bucket")
        assert bucket == "my-bucket"
        assert blob == ""

    def test_invalid(self):
        with pytest.raises(ValueError):
            gcs_uri_to_bucket_and_blob("https://bucket/blob")
