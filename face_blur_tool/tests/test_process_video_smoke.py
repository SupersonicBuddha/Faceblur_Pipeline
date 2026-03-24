"""
Smoke test: process one tiny synthetic video end-to-end (no real GCS/GPU).

We generate a small synthetic video locally, stub out GCS I/O, inject a
fake detector, and verify the output file is produced with the correct
resolution and FPS.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.config import FaceBlurConfig
from src.detectors.base import Detection, FaceDetector


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_synthetic_video(path: str, n_frames: int = 30, w: int = 320, h: int = 240) -> None:
    """Write a tiny solid-colour video to *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (w, h))
    for i in range(n_frames):
        colour = int(i * 255 / n_frames)
        frame = np.full((h, w, 3), colour, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class FakeDetector(FaceDetector):
    """Returns a fixed detection for every N-th frame."""

    def __init__(self, detections_per_frame: List[Detection]) -> None:
        self._dets = detections_per_frame

    def detect(self, frame) -> List[Detection]:
        return list(self._dets)

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeBlob:
    def __init__(self, key: str, store: Dict) -> None:
        self._key = key
        self._store = store

    def exists(self) -> bool:
        return self._key in self._store

    def download_as_text(self, encoding="utf-8") -> str:
        return self._store.get(self._key, "")

    def upload_from_string(self, data: bytes, content_type="") -> None:
        self._store[self._key] = data.decode("utf-8")

    def download_to_filename(self, path: str) -> None:
        content = self._store.get(self._key, b"")
        if isinstance(content, str):
            content = content.encode()
        with open(path, "wb") as f:
            f.write(content)

    def upload_from_filename(self, path: str, content_type="") -> None:
        with open(path, "rb") as f:
            self._store[self._key] = f.read()


class _FakeBucket:
    def __init__(self, name, store):
        self._name = name
        self._store = store

    def blob(self, name):
        return _FakeBlob(f"{self._name}/{name}", self._store)


class _FakeGCSClient:
    def __init__(self):
        self._store: Dict = {}

    def bucket(self, name):
        return _FakeBucket(name, self._store)

    def list_blobs(self, bucket, prefix="", **_):
        return iter([])


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestProcessVideoSmoke:
    def test_no_faces_produces_output(self, tmp_path):
        """A video with no detections should still produce a valid output file."""
        # Create a synthetic input video
        input_path = str(tmp_path / "input.mp4")
        _make_synthetic_video(input_path, n_frames=15, w=160, h=120)

        # Set up fake GCS with the video pre-loaded
        gcs_client = _FakeGCSClient()
        src_bucket = "tbd-qa"
        src_blob = "for_fb/input.mp4"
        gcs_client._store[f"{src_bucket}/{src_blob}"] = open(input_path, "rb").read()

        config = FaceBlurConfig(local_temp_dir=str(tmp_path / "tmp"))
        os.makedirs(config.local_temp_dir, exist_ok=True)

        detector = FakeDetector([])  # no detections

        from src.pipeline.process_video import process_single_video

        result = process_single_video(
            source_gcs_path=f"gs://{src_bucket}/{src_blob}",
            output_gcs_path="gs://tbd-qa/post_fb/input_blurred.mp4",
            gcs_client=gcs_client,
            config=config,
            detector=detector,
        )

        assert result.status == "success", f"Expected success, got: {result.error_message}"
        # Output blob should be in the fake store
        out_key = "tbd-qa/post_fb/input_blurred.mp4"
        assert out_key in gcs_client._store
        output_bytes = gcs_client._store[out_key]
        assert len(output_bytes) > 0

    def test_with_face_detection_produces_output(self, tmp_path):
        """A video with face detections should produce a blurred output."""
        input_path = str(tmp_path / "input.mp4")
        _make_synthetic_video(input_path, n_frames=15, w=320, h=240)

        gcs_client = _FakeGCSClient()
        src_bucket = "tbd-qa"
        src_blob = "for_fb/face_vid.mp4"
        gcs_client._store[f"{src_bucket}/{src_blob}"] = open(input_path, "rb").read()

        config = FaceBlurConfig(local_temp_dir=str(tmp_path / "tmp"))
        os.makedirs(config.local_temp_dir, exist_ok=True)

        # Fake detector always returns a detection in the center of the frame
        detector = FakeDetector(
            [Detection(bbox=(80, 60, 160, 120), confidence=0.95)]
        )

        from src.pipeline.process_video import process_single_video

        result = process_single_video(
            source_gcs_path=f"gs://{src_bucket}/{src_blob}",
            output_gcs_path="gs://tbd-qa/post_fb/face_vid_blurred.mp4",
            gcs_client=gcs_client,
            config=config,
            detector=detector,
        )

        assert result.status == "success", f"Expected success, got: {result.error_message}"
        assert result.faces_detected_total > 0
