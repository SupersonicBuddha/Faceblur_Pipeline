"""
Edge-case tests for URI parsing and output-path derivation.

Covers gaps not hit by test_output_naming.py:
- Empty string and scheme-only URIs passed to gcs_uri_to_bucket_and_blob
- gs:// with no bucket name
- Plain path (no scheme) passed to both helpers
- build_output_gcs_path with extension-less source file
- build_output_gcs_path where source exactly equals the prefix (no filename component)
- build_output_gcs_path with a single-component relative path (file at prefix root)
"""

from __future__ import annotations

import pytest

from src.utils.paths import build_output_gcs_path, gcs_uri_to_bucket_and_blob


class TestGcsUriEdgeCases:

    def test_empty_string_raises(self):
        """An empty string is not a valid GCS URI."""
        with pytest.raises(ValueError):
            gcs_uri_to_bucket_and_blob("")

    def test_plain_path_no_scheme_raises(self):
        """A bare filesystem path is not a GCS URI."""
        with pytest.raises(ValueError):
            gcs_uri_to_bucket_and_blob("/data/bucket/video.mp4")

    def test_wrong_scheme_raises(self):
        """s3:// and https:// must both raise."""
        with pytest.raises(ValueError):
            gcs_uri_to_bucket_and_blob("s3://bucket/blob.mp4")
        with pytest.raises(ValueError):
            gcs_uri_to_bucket_and_blob("https://storage.googleapis.com/bucket/blob")

    def test_scheme_only_uri_returns_empty_bucket_and_blob(self):
        """
        'gs://' with nothing after it is technically parseable but
        produces an empty bucket name and empty blob.

        This is not a valid GCS path — callers should treat an empty
        bucket name as an error.  The test documents current behaviour
        so any future stricter validation is a visible, intentional change.
        """
        bucket, blob = gcs_uri_to_bucket_and_blob("gs://")
        assert bucket == ""
        assert blob == ""

    def test_bucket_only_uri_gives_empty_blob(self):
        """gs://bucket (no trailing slash, no blob) → blob is empty string."""
        bucket, blob = gcs_uri_to_bucket_and_blob("gs://my-bucket")
        assert bucket == "my-bucket"
        assert blob == ""

    def test_trailing_slash_preserved_in_blob(self):
        """
        A trailing slash on the blob component is passed through unchanged.
        This matches a 'directory marker' object name in GCS.
        """
        bucket, blob = gcs_uri_to_bucket_and_blob("gs://bucket/some/prefix/")
        assert bucket == "bucket"
        assert blob == "some/prefix/"

    def test_deep_nested_blob_parsed(self):
        bucket, blob = gcs_uri_to_bucket_and_blob("gs://b/a/b/c/d/e/f.mp4")
        assert bucket == "b"
        assert blob == "a/b/c/d/e/f.mp4"

    def test_blob_with_spaces_parsed(self):
        """Blob names may contain spaces — the parser must not strip them."""
        bucket, blob = gcs_uri_to_bucket_and_blob("gs://bucket/path with spaces/v.mp4")
        assert bucket == "bucket"
        assert blob == "path with spaces/v.mp4"


class TestBuildOutputGcsPathEdgeCases:

    def test_extensionless_source_file(self):
        """
        A source file with no extension should get '_blurred' appended and
        no dot-separated suffix added.
        """
        out = build_output_gcs_path(
            "gs://bucket/input/raw_footage",
            "gs://bucket/input",
            "gs://bucket/output",
        )
        assert out == "gs://bucket/output/raw_footage_blurred"

    def test_source_with_multiple_dots_in_name(self):
        """
        Only the last extension is treated as the extension;
        earlier dots are part of the stem.
        """
        out = build_output_gcs_path(
            "gs://bucket/in/my.video.2024.mp4",
            "gs://bucket/in",
            "gs://bucket/out",
        )
        assert out == "gs://bucket/out/my.video.2024_blurred.mp4"

    def test_source_directly_under_prefix_root(self):
        """File sits immediately at the prefix root (no subdirectory)."""
        out = build_output_gcs_path(
            "gs://bucket/input/clip.mp4",
            "gs://bucket/input",
            "gs://bucket/output",
        )
        assert out == "gs://bucket/output/clip_blurred.mp4"

    def test_source_equals_prefix_path_raises(self):
        """
        If source_uri == input_prefix (no trailing slash, no filename),
        it does NOT start with input_prefix + '/' so ValueError is raised.
        """
        with pytest.raises(ValueError):
            build_output_gcs_path(
                "gs://bucket/input",
                "gs://bucket/input",
                "gs://bucket/output",
            )

    def test_partial_prefix_match_raises(self):
        """
        'gs://bucket/input_extra/v.mp4' must not match prefix 'gs://bucket/input'.
        The trailing '/' guard prevents false prefix matches on directory names.
        """
        with pytest.raises(ValueError):
            build_output_gcs_path(
                "gs://bucket/input_extra/v.mp4",
                "gs://bucket/input",
                "gs://bucket/output",
            )

    def test_output_prefix_with_trailing_slash_normalised(self):
        """Trailing slashes on either prefix should not produce double slashes."""
        out = build_output_gcs_path(
            "gs://bucket/in/v.mp4",
            "gs://bucket/in/",
            "gs://bucket/out/",
        )
        assert "//" not in out.replace("gs://", "")
        assert out == "gs://bucket/out/v_blurred.mp4"

    def test_different_bucket_names_in_input_and_output(self):
        out = build_output_gcs_path(
            "gs://src-bucket/raw/2024/clip.mov",
            "gs://src-bucket/raw",
            "gs://dst-bucket/processed",
        )
        assert out == "gs://dst-bucket/processed/2024/clip_blurred.mov"
