"""
Tests for exception classification in process_single_video.

Verifies that:
- Expected operational failures (probe error, mux error, I/O error) produce
  VideoResult(status="failed") without propagating exceptions.
- The error_message includes the exception type for diagnostics.
- Unexpected errors are also captured (not propagated) and flagged as
  "Unexpected:" in the message.
"""

from __future__ import annotations

import os
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.config import FaceBlurConfig
from src.frame_writer import FrameWriteError
from src.pipeline.process_video import process_single_video
from src.video_probe import VideoProbeError


# ── Minimal fake GCS client ────────────────────────────────────────────────


class _FakeBlob:
    def __init__(self, key, store):
        self._key = key
        self._store = store

    def exists(self):
        return self._key in self._store

    def download_as_text(self, encoding="utf-8"):
        val = self._store.get(self._key, "")
        return val if isinstance(val, str) else val.decode()

    def upload_from_string(self, data, content_type=""):
        self._store[self._key] = data.decode() if isinstance(data, bytes) else data

    def download_to_filename(self, path):
        content = self._store.get(self._key, b"")
        if isinstance(content, str):
            content = content.encode()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)

    def upload_from_filename(self, path, content_type=""):
        with open(path, "rb") as f:
            self._store[self._key] = f.read()


class _FakeBucket:
    def __init__(self, name, store):
        self._name = name
        self._store = store

    def blob(self, name):
        return _FakeBlob(f"{self._name}/{name}", self._store)


class FakeGCSClient:
    def __init__(self, video_bytes: bytes = b"\x00" * 16):
        self._store: Dict = {
            "tbd-qa/for_fb/v.mp4": video_bytes,
        }

    def bucket(self, name):
        return _FakeBucket(name, self._store)

    def list_blobs(self, *a, **kw):
        return iter([])


def _config(tmp_path) -> FaceBlurConfig:
    return FaceBlurConfig(local_temp_dir=str(tmp_path / "tmp"))


# ── Tests ──────────────────────────────────────────────────────────────────


class TestExceptionTaxonomy:

    def _run(self, tmp_path, **patches):
        """Helper: patch probe_video and run process_single_video."""
        config = _config(tmp_path)
        client = FakeGCSClient()
        detector = MagicMock()
        detector.detect.return_value = []

        kwargs = dict(
            source_gcs_path="gs://tbd-qa/for_fb/v.mp4",
            output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
            gcs_client=client,
            config=config,
            detector=detector,
        )

        ctx = {}
        for target, side_effect in patches.items():
            ctx[target] = patch(target, side_effect=side_effect)

        active = [c.__enter__() for c in ctx.values()]
        result = process_single_video(**kwargs)
        for c in ctx.values():
            c.__exit__(None, None, None)
        return result

    def test_probe_error_produces_failed_result(self, tmp_path):
        """VideoProbeError must be caught and recorded — not re-raised."""
        result = self._run(
            tmp_path,
            **{"src.pipeline.process_video.probe_video": VideoProbeError("corrupt file")}
        )
        assert result.status == "failed"
        assert "VideoProbeError" in result.error_message
        assert "corrupt file" in result.error_message

    def test_mux_error_produces_failed_result(self, tmp_path):
        """FrameWriteError from ffmpeg must be caught and recorded."""
        with patch("src.pipeline.process_video.probe_video") as mock_probe, \
             patch("src.pipeline.process_video.mux_audio",
                   side_effect=FrameWriteError("ffmpeg exited 1")), \
             patch("src.pipeline.process_video._run_detection_and_blur",
                   return_value=str(tmp_path / "dummy.mp4")):
            # probe returns a minimal VideoInfo
            from src.video_probe import VideoInfo
            mock_probe.return_value = VideoInfo(
                width=160, height=120, fps=25.0,
                duration_sec=1.0, codec="h264",
                has_audio=False, nb_frames=25,
            )
            config = _config(tmp_path)
            client = FakeGCSClient()
            detector = MagicMock()
            detector.detect.return_value = []

            result = process_single_video(
                source_gcs_path="gs://tbd-qa/for_fb/v.mp4",
                output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
                gcs_client=client,
                config=config,
                detector=detector,
            )

        assert result.status == "failed"
        assert "FrameWriteError" in result.error_message

    def test_os_error_on_download_produces_failed_result(self, tmp_path):
        """OSError (e.g. disk full) must be caught and recorded."""
        with patch(
            "src.gcs_io.download_blob_to_file",
            side_effect=OSError("No space left on device"),
        ):
            config = _config(tmp_path)
            client = FakeGCSClient()
            detector = MagicMock()
            result = process_single_video(
                source_gcs_path="gs://tbd-qa/for_fb/v.mp4",
                output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
                gcs_client=client,
                config=config,
                detector=detector,
            )
        assert result.status == "failed"
        assert "OSError" in result.error_message

    def test_unexpected_error_is_captured_with_unexpected_prefix(self, tmp_path):
        """Bugs that produce unexpected RuntimeErrors must be captured too."""
        with patch(
            "src.pipeline.process_video.probe_video",
            side_effect=RuntimeError("internal bug"),
        ):
            config = _config(tmp_path)
            client = FakeGCSClient()
            detector = MagicMock()
            result = process_single_video(
                source_gcs_path="gs://tbd-qa/for_fb/v.mp4",
                output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
                gcs_client=client,
                config=config,
                detector=detector,
            )
        assert result.status == "failed"
        assert "Unexpected" in result.error_message

    def test_process_never_raises(self, tmp_path):
        """process_single_video must NEVER propagate exceptions."""
        with patch(
            "src.pipeline.process_video.probe_video",
            side_effect=SystemError("very bad"),
        ):
            config = _config(tmp_path)
            client = FakeGCSClient()
            detector = MagicMock()
            # Must not raise
            result = process_single_video(
                source_gcs_path="gs://tbd-qa/for_fb/v.mp4",
                output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
                gcs_client=client,
                config=config,
                detector=detector,
            )
        assert result.status == "failed"
