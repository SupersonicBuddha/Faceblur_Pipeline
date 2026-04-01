"""
Write processed frames to a temporary video file, then mux with audio.

We pipe raw BGR frames directly into an ffmpeg subprocess (libx264), then use
a second ffmpeg call to mux the original audio track back in.

Using ffmpeg instead of cv2.VideoWriter avoids a class of silent failures where
VideoWriter reports isOpened()=True but writes zero frames (e.g. when the mp4v
codec is unavailable or frame dimensions disagree with the container header due
to rotation metadata).  libx264 is universally available on Linux/Colab and
raises real errors on failure.
"""

from __future__ import annotations

import subprocess
import tempfile
from typing import Optional

import numpy as np

from src.logging_utils import get_logger

logger = get_logger(__name__)


class FrameWriteError(RuntimeError):
    pass


class FrameSink:
    """
    Incremental frame sink that pipes raw BGR frames to ffmpeg (libx264).

    Usage
    -----
    ::

        sink = FrameSink(path, fps, width, height)
        for frame in frames:
            sink.write(frame)
        sink.close()

    Or as a context manager (recommended — guarantees close on error)::

        with FrameSink(path, fps, width, height) as sink:
            for frame in frames:
                sink.write(frame)
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",  # kept for API compatibility; ignored (libx264 is used)
    ) -> None:
        self._path = output_path
        self._frame_count = 0
        # Stderr is redirected to a temp file so the pipe never blocks while
        # we are streaming stdin frames.  We read it back on close() for error
        # reporting.
        self._stderr_buf: tempfile.SpooledTemporaryFile = tempfile.SpooledTemporaryFile(
            max_size=1 << 20  # 1 MB in memory, spill to disk beyond that
        )

        cmd = [
            "ffmpeg", "-y",
            # ── Input: raw BGR frames from stdin ──────────────────────────
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            # ── Output: H.264 in an MP4 container ─────────────────────────
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",  # required for broad player compatibility
            output_path,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_buf,
            )
        except FileNotFoundError:
            raise FrameWriteError("ffmpeg not found; install ffmpeg.")

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # ffmpeg exited early — read stderr for the reason
            self._stderr_buf.seek(0)
            stderr = self._stderr_buf.read().decode("utf-8", errors="replace")
            raise FrameWriteError(
                f"ffmpeg pipe closed unexpectedly while writing frames: {stderr}"
            )
        self._frame_count += 1

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        self._proc.wait()

        self._stderr_buf.seek(0)
        stderr_text = self._stderr_buf.read().decode("utf-8", errors="replace")
        self._stderr_buf.close()

        if self._proc.returncode != 0:
            raise FrameWriteError(
                f"ffmpeg video write failed (rc={self._proc.returncode}): {stderr_text}"
            )
        if self._frame_count == 0:
            raise FrameWriteError(
                f"FrameSink closed with no frames written to {self._path!r}"
            )

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
