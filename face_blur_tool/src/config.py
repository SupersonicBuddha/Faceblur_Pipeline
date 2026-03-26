"""
Configuration dataclass for the face blur pipeline.

All settings have sane defaults for the tbd-qa project.
Override via environment variables or direct instantiation in the Colab notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FaceBlurConfig:
    # ── GCS paths ──────────────────────────────────────────────────────────
    input_prefix: str = "gs://tbd-qa/for_fb"
    output_prefix: str = "gs://tbd-qa/post_fb"
    manifest_path: str = "gs://tbd-qa/post_fb/processed_manifest.jsonl"

    # ── Input filtering ────────────────────────────────────────────────────
    supported_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".mov"]
    )

    # ── Detector ───────────────────────────────────────────────────────────
    # "retinaface" | "mediapipe"
    detector_type: str = "retinaface"

    # Run detection every N frames; track between detection frames
    detection_interval: int = 4

    # Minimum face bounding-box dimension in pixels (filter tiny faces)
    min_face_size: int = 30

    # Confidence threshold — kept low to bias toward recall
    detector_confidence_threshold: float = 0.35

    # ── Tracking ───────────────────────────────────────────────────────────
    # IoU threshold for associating a detection to an existing track
    tracker_iou_threshold: float = 0.25

    # How many consecutive missed-detection frames a track stays alive
    track_persistence_frames: int = 12

    # ── Scene cut ──────────────────────────────────────────────────────────
    # Normalised histogram delta [0–1]; above this → scene cut
    scene_cut_threshold: float = 0.40

    # ── Smoothing / interpolation ──────────────────────────────────────────
    # Rolling-window size for bbox coordinate smoothing
    smoothing_window: int = 7

    # Max gap (frames) across which bbox is interpolated
    interpolation_max_gap: int = 8

    # ── Blur mask ──────────────────────────────────────────────────────────
    # Fractional padding added to each side of the detected bbox
    blur_padding_ratio: float = 0.35

    # ── Blur transform ─────────────────────────────────────────────────────
    # Gaussian kernel size (must be odd; 0 → auto from strength)
    blur_kernel_size: int = 0  # 0 = auto

    # Blur strength (sigma for Gaussian; higher = stronger)
    blur_strength: float = 25.0

    # ── Tiled detection ────────────────────────────────────────────────────
    # Split each frame into overlapping tiles before detection.
    # Greatly improves recall for small/background faces.
    use_tiled_detection: bool = True

    # Size of each square tile in pixels
    tile_size: int = 640

    # Fractional overlap between adjacent tiles [0, 1)
    tile_overlap: float = 0.25

    # IoU threshold for suppressing duplicate detections across tile boundaries
    tile_nms_threshold: float = 0.4

    # ── Retry ──────────────────────────────────────────────────────────────
    retry_count: int = 1

    # ── ffmpeg ─────────────────────────────────────────────────────────────
    # Timeout (seconds) for the ffmpeg audio-mux subprocess.
    # 300 s is safe for typical jobs (≤ 2 h video on a slow machine).
    # Raise this for very long videos or slow storage.
    ffmpeg_mux_timeout_sec: int = 300

    # ── Local temp dir ─────────────────────────────────────────────────────
    local_temp_dir: str = "/tmp/face_blur"

    # ── Audio ──────────────────────────────────────────────────────────────
    preserve_audio: bool = True

    # ── GCP project (optional; needed for some credential paths) ───────────
    gcp_project: Optional[str] = None

    # ── Model version tag embedded in manifest records ─────────────────────
    model_version: str = "v1.0.0"

    def __post_init__(self) -> None:
        # Allow env-var overrides for the most common knobs
        if ev := os.environ.get("FB_INPUT_PREFIX"):
            self.input_prefix = ev
        if ev := os.environ.get("FB_OUTPUT_PREFIX"):
            self.output_prefix = ev
        if ev := os.environ.get("FB_MANIFEST_PATH"):
            self.manifest_path = ev
        if ev := os.environ.get("FB_DETECTOR_TYPE"):
            self.detector_type = ev
        if ev := os.environ.get("FB_LOCAL_TEMP_DIR"):
            self.local_temp_dir = ev
        if ev := os.environ.get("GCP_PROJECT"):
            self.gcp_project = ev
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError for any invalid configuration value."""
        errors: List[str] = []

        # GCS URIs must start with gs://
        for attr in ("input_prefix", "output_prefix", "manifest_path"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val.startswith("gs://"):
                errors.append(f"{attr} must be a GCS URI (gs://...), got {val!r}")

        if self.detection_interval < 1:
            errors.append(
                f"detection_interval must be >= 1, got {self.detection_interval}"
            )
        if not (0.0 <= self.detector_confidence_threshold <= 1.0):
            errors.append(
                f"detector_confidence_threshold must be in [0.0, 1.0], "
                f"got {self.detector_confidence_threshold}"
            )
        if self.blur_padding_ratio < 0.0:
            errors.append(
                f"blur_padding_ratio must be >= 0.0, got {self.blur_padding_ratio}"
            )
        if not (0.0 <= self.tracker_iou_threshold <= 1.0):
            errors.append(
                f"tracker_iou_threshold must be in [0.0, 1.0], "
                f"got {self.tracker_iou_threshold}"
            )
        if self.ffmpeg_mux_timeout_sec < 1:
            errors.append(
                f"ffmpeg_mux_timeout_sec must be >= 1, "
                f"got {self.ffmpeg_mux_timeout_sec}"
            )
        if self.retry_count < 0:
            errors.append(f"retry_count must be >= 0, got {self.retry_count}")
        if self.smoothing_window < 1:
            errors.append(
                f"smoothing_window must be >= 1, got {self.smoothing_window}"
            )
        if self.interpolation_max_gap < 0:
            errors.append(
                f"interpolation_max_gap must be >= 0, got {self.interpolation_max_gap}"
            )

        if errors:
            raise ValueError(
                "Invalid FaceBlurConfig:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    # ── Derived helpers ────────────────────────────────────────────────────

    @property
    def input_bucket(self) -> str:
        """Return the GCS bucket name from input_prefix."""
        return _parse_gcs_prefix(self.input_prefix)[0]

    @property
    def input_blob_prefix(self) -> str:
        """Return the blob prefix (without bucket) from input_prefix."""
        return _parse_gcs_prefix(self.input_prefix)[1]

    @property
    def output_bucket(self) -> str:
        return _parse_gcs_prefix(self.output_prefix)[0]

    @property
    def output_blob_prefix(self) -> str:
        return _parse_gcs_prefix(self.output_prefix)[1]

    @property
    def manifest_bucket(self) -> str:
        return _parse_gcs_prefix(self.manifest_path)[0]

    @property
    def manifest_blob(self) -> str:
        return _parse_gcs_prefix(self.manifest_path)[1]

    @property
    def effective_blur_kernel(self) -> int:
        """Return an odd kernel size derived from blur_strength when blur_kernel_size==0."""
        if self.blur_kernel_size > 0:
            k = self.blur_kernel_size
        else:
            k = max(3, int(self.blur_strength) * 2 + 1)
        # Ensure odd
        return k if k % 2 == 1 else k + 1


def _parse_gcs_prefix(uri: str) -> tuple[str, str]:
    """Split 'gs://bucket/some/prefix' → ('bucket', 'some/prefix')."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected GCS URI starting with gs://, got: {uri!r}")
    without_scheme = uri[len("gs://"):]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix
