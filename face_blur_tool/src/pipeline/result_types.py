"""
Result types returned by the video processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoResult:
    """Outcome of processing a single video."""

    source_gcs_path: str
    output_gcs_path: str
    status: str  # "success" | "failed" | "skipped"
    retry_count: int = 0
    error_message: Optional[str] = None
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frames_processed: int = 0
    faces_detected_total: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"
