"""
Miscellaneous ffmpeg helpers.
"""

from __future__ import annotations

import shutil
import subprocess


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def ffmpeg_version() -> str:
    """Return the ffmpeg version string, or 'unavailable'."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        first_line = result.stdout.decode("utf-8", errors="replace").splitlines()[0]
        return first_line
    except Exception:
        return "unavailable"


def reencode_video(
    input_path: str,
    output_path: str,
    fps: float,
    crf: int = 18,
) -> None:
    """
    Re-encode a video with H.264 at the specified FPS and quality.

    Used when the container format requires a proper H.264 stream
    (e.g. .mov output).  ``crf=18`` is visually lossless.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-r", str(fps),
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg re-encode failed: {stderr}")
