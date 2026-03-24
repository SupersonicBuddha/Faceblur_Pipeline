"""
Pure path-manipulation helpers — no GCS or filesystem I/O.
"""

from __future__ import annotations

import os


def build_output_gcs_path(
    source_uri: str,
    input_prefix: str,
    output_prefix: str,
) -> str:
    """
    Compute the output GCS URI for a source video.

    Rules
    -----
    * Preserve the relative folder structure that sits beneath *input_prefix*.
    * Insert ``_blurred`` before the file extension.
    * Preserve the original extension.

    Examples
    --------
    >>> build_output_gcs_path(
    ...     "gs://tbd-qa/for_fb/foo/bar/video1.mp4",
    ...     "gs://tbd-qa/for_fb",
    ...     "gs://tbd-qa/post_fb",
    ... )
    'gs://tbd-qa/post_fb/foo/bar/video1_blurred.mp4'
    """
    # Strip trailing slash from both prefixes for consistent comparison
    clean_src_prefix = input_prefix.rstrip("/")
    clean_out_prefix = output_prefix.rstrip("/")

    if not source_uri.startswith(clean_src_prefix + "/"):
        raise ValueError(
            f"source_uri {source_uri!r} does not start with "
            f"input_prefix {input_prefix!r}"
        )

    # Relative path from the input prefix root
    relative_path = source_uri[len(clean_src_prefix) + 1:]  # e.g. "foo/bar/video1.mp4"

    # Inject _blurred before extension
    stem, ext = os.path.splitext(relative_path)
    blurred_relative = f"{stem}_blurred{ext}"

    return f"{clean_out_prefix}/{blurred_relative}"


def gcs_uri_to_bucket_and_blob(uri: str) -> tuple[str, str]:
    """
    Parse 'gs://bucket/path/to/blob' → ('bucket', 'path/to/blob').
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri!r}")
    without_scheme = uri[len("gs://"):]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob


def local_temp_path(temp_dir: str, gcs_uri: str, suffix: str = "") -> str:
    """
    Derive a stable local temp path from a GCS URI.

    Replaces 'gs://' and '/' with safe filesystem chars so the path
    stays human-readable during debugging.
    """
    safe = gcs_uri.replace("gs://", "").replace("/", "_")
    if suffix and not safe.endswith(suffix):
        safe += suffix
    return os.path.join(temp_dir, safe)
