"""
Write processed frames to a temporary video file, then mux with audio.

We write a raw video-only file with OpenCV VideoWriter, then use ffmpeg
to mux the original audio track back in (if requested and available).
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from typing import Generator, Optional

import cv2
import numpy as np

from src.logging_utils import get_logger

logger = get_logger(__name__)


class FrameWriteError(RuntimeError):
    pass


class FrameSink:
    """
    Incremental frame sink backed by cv2.VideoWriter.

    Usage
    -----
    ::

        sink = FrameSink(path, fps, width, height)
        for frame in frames:
            sink.write(frame)
        sink.close()
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        self._path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise FrameWriteError(
                f"cv2.VideoWriter could not open {output_path!r} "
                f"(fps={fps}, {width}x{height}, codec={codec})"
            )

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "FrameSink":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def mux_audio(
    video_only_path: str,
    original_video_path: str,
    output_path: str,
    has_audio: bool,
    timeout_sec: int = 300,
) -> None:
    """
    Use ffmpeg to combine a processed video-only file with the original audio.

    If *has_audio* is False or the source has no audio stream, the video-only
    file is simply copied to *output_path*.

    Parameters
    ----------
    video_only_path:
        Path to the processed frames file (no audio).
    original_video_path:
        Path to the original source video (audio source).
    output_path:
        Destination for the final muxed file.
    has_audio:
        Whether the original video has an audio track.
    timeout_sec:
        Maximum seconds to wait for ffmpeg.  Raise this for very long videos.
        Defaults to 300 s.  Configurable via ``FaceBlurConfig.ffmpeg_mux_timeout_sec``.
    """
    if not has_audio:
        logger.info("No audio — copying video-only output directly.")
        import shutil
        shutil.copy2(video_only_path, output_path)
        return

    cmd = [
        "ffmpeg",
        "-y",                          # overwrite output
        "-i", video_only_path,         # video stream
        "-i", original_video_path,     # audio stream source
        "-c:v", "copy",                # copy video stream unchanged
        "-c:a", "aac",                 # re-encode audio to AAC for compatibility
        "-map", "0:v:0",               # video from first input
        "-map", "1:a:0?",              # audio from second input (optional)
        "-shortest",                   # end when shortest stream ends
        output_path,
    ]
    logger.info(f"Muxing audio: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        raise FrameWriteError(
            f"ffmpeg mux timed out after {timeout_sec}s. "
            f"Increase ffmpeg_mux_timeout_sec in config for long videos."
        )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise FrameWriteError(f"ffmpeg mux failed (rc={result.returncode}): {stderr}")
