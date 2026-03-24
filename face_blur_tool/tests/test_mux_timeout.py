"""
Unit tests for configurable ffmpeg mux timeout.

All subprocess calls are mocked; no real ffmpeg is invoked.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.config import FaceBlurConfig
from src.frame_writer import FrameWriteError, mux_audio


class TestMuxTimeout:
    def test_timeout_raises_frame_write_error(self):
        """TimeoutExpired from subprocess must be converted to FrameWriteError."""
        with patch(
            "src.frame_writer.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1),
        ):
            with pytest.raises(FrameWriteError, match="timed out"):
                mux_audio(
                    video_only_path="v.mp4",
                    original_video_path="src.mp4",
                    output_path="out.mp4",
                    has_audio=True,
                    timeout_sec=1,
                )

    def test_timeout_error_message_includes_seconds(self):
        """Error message must mention the configured timeout duration."""
        with patch(
            "src.frame_writer.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=42),
        ):
            with pytest.raises(FrameWriteError, match="42s"):
                mux_audio(
                    video_only_path="v.mp4",
                    original_video_path="src.mp4",
                    output_path="out.mp4",
                    has_audio=True,
                    timeout_sec=42,
                )

    def test_default_timeout_is_300(self):
        """Default timeout must be 300 s (safe for typical jobs)."""
        # Capture the timeout kwarg passed to subprocess.run
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("src.frame_writer.subprocess.run", side_effect=_fake_run):
            mux_audio(
                video_only_path="v.mp4",
                original_video_path="src.mp4",
                output_path="out.mp4",
                has_audio=True,
                # no timeout_sec → should default to 300
            )
        assert captured["timeout"] == 300

    def test_custom_timeout_passed_to_subprocess(self):
        """A custom timeout_sec value must reach subprocess.run."""
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("src.frame_writer.subprocess.run", side_effect=_fake_run):
            mux_audio(
                video_only_path="v.mp4",
                original_video_path="src.mp4",
                output_path="out.mp4",
                has_audio=True,
                timeout_sec=600,
            )
        assert captured["timeout"] == 600

    def test_config_timeout_field_default(self):
        """FaceBlurConfig.ffmpeg_mux_timeout_sec must default to 300."""
        cfg = FaceBlurConfig()
        assert cfg.ffmpeg_mux_timeout_sec == 300

    def test_config_timeout_field_overridable(self):
        """FaceBlurConfig.ffmpeg_mux_timeout_sec must be overridable."""
        cfg = FaceBlurConfig(ffmpeg_mux_timeout_sec=600)
        assert cfg.ffmpeg_mux_timeout_sec == 600

    def test_no_audio_skips_subprocess(self):
        """When has_audio=False, ffmpeg must not be called at all."""
        import tempfile, os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            src = f.name
            f.write(b"\x00" * 8)  # dummy bytes

        try:
            with patch("src.frame_writer.subprocess.run") as mock_run, \
                 patch("shutil.copy2") as mock_copy:
                mux_audio(
                    video_only_path=src,
                    original_video_path=src,
                    output_path="/tmp/out_test.mp4",
                    has_audio=False,
                    timeout_sec=1,
                )
            mock_run.assert_not_called()
            mock_copy.assert_called_once()
        finally:
            os.unlink(src)

    def test_ffmpeg_nonzero_returncode_raises(self):
        """A non-zero return code from ffmpeg must raise FrameWriteError."""
        result = MagicMock()
        result.returncode = 1
        result.stderr = b"some ffmpeg error"

        with patch("src.frame_writer.subprocess.run", return_value=result):
            with pytest.raises(FrameWriteError, match="ffmpeg mux failed"):
                mux_audio(
                    video_only_path="v.mp4",
                    original_video_path="src.mp4",
                    output_path="out.mp4",
                    has_audio=True,
                    timeout_sec=300,
                )
