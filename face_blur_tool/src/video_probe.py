"""
Probe a local video file with ffprobe to extract metadata.

We run ffprobe before spending GPU time on a video so that corrupt or
unreadable files are caught early.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_sec: float
    codec: str
    has_audio: bool
    nb_frames: Optional[int]  # may be None for some containers


class VideoProbeError(RuntimeError):
    """Raised when ffprobe cannot parse the file."""


def probe_video(local_path: str) -> VideoInfo:
    """
    Run ffprobe on *local_path* and return :class:`VideoInfo`.

    Raises
    ------
    VideoProbeError
        If ffprobe fails or the file has no valid video stream.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        local_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except FileNotFoundError:
        raise VideoProbeError("ffprobe not found; install ffmpeg.")
    except subprocess.TimeoutExpired:
        raise VideoProbeError(f"ffprobe timed out on {local_path!r}")

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise VideoProbeError(
            f"ffprobe returned code {result.returncode} for {local_path!r}: {stderr}"
        )

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"Could not parse ffprobe output: {exc}")

    video_stream = None
    has_audio = False
    for stream in info.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio":
            has_audio = True

    if video_stream is None:
        raise VideoProbeError(f"No video stream found in {local_path!r}")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if width == 0 or height == 0:
        raise VideoProbeError(f"Invalid dimensions {width}x{height} in {local_path!r}")

    fps = _parse_fps(video_stream.get("r_frame_rate", "25/1"))

    # Duration: prefer stream value, fall back to format
    duration_sec = float(
        video_stream.get("duration")
        or info.get("format", {}).get("duration", 0)
        or 0
    )

    codec = video_stream.get("codec_name", "unknown")

    nb_frames_raw = video_stream.get("nb_frames")
    nb_frames = int(nb_frames_raw) if nb_frames_raw else None

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
        codec=codec,
        has_audio=has_audio,
        nb_frames=nb_frames,
    )


def _parse_fps(rate_str: str) -> float:
    """Parse ffprobe r_frame_rate like '30000/1001' or '25/1'."""
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            den_val = float(den)
            if den_val == 0:
                return 25.0
            return float(num) / den_val
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 25.0
