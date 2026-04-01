"""
Single-video processing pipeline.

This module contains the core face-blurring logic for a single video:

  1. Download from GCS to a local temp file.
  2. Probe: extract width/height/fps/duration.
  3. Detect + Track: run the face detector every N frames; track between
     detection frames; reset tracker on scene cuts.
  4. Smooth + Interpolate: stabilise bbox positions.
  5. Blur: apply elliptical Gaussian blur to each frame.
  6. Write: stream frames to a temp output file.
  7. Mux: re-attach audio with ffmpeg.
  8. Upload: push the finished video to GCS.

The function ``process_single_video`` is the main entry point and is
designed to be called by ``batch_runner.py`` (with retry wrapping).

All GCS I/O is routed through ``gcs_io.py`` so tests can inject fakes.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2

from src.blur.compositor import apply_face_blurs
from src.config import FaceBlurConfig
from src.detectors.base import FaceDetector
from src.detectors.factory import make_detector
from src.frame_reader import iter_frames
from src.frame_writer import FrameSink, FrameWriteError, mux_audio
from src.logging_utils import get_logger, log_video_end, log_video_start
from src.pipeline.result_types import VideoResult
from src.tracking.interpolation import fill_gaps
from src.tracking.scene_cut import SceneCutDetector
from src.tracking.smoothing import BBoxSmoother
from src.tracking.tracker import FaceTracker
from src.tracking.types import BBox
from src.utils.paths import gcs_uri_to_bucket_and_blob
from src.utils.timing import Stopwatch
from src.video_probe import VideoInfo, VideoProbeError, probe_video

logger = get_logger(__name__)

# Codec used for the intermediate (video-only) output file.
# "mp4v" is widely supported; final output is re-encoded via ffmpeg.
_INTERMEDIATE_CODEC = "mp4v"


def process_single_video(
    source_gcs_path: str,
    output_gcs_path: str,
    gcs_client,
    config: FaceBlurConfig,
    detector: Optional[FaceDetector] = None,
) -> VideoResult:
    """
    Download, process, and upload a single video.

    Parameters
    ----------
    source_gcs_path:
        Full GCS URI of the input video, e.g. ``gs://tbd-qa/for_fb/foo/v.mp4``.
    output_gcs_path:
        Full GCS URI where the blurred video should be written.
    gcs_client:
        A ``google.cloud.storage.Client`` (or compatible fake).
    config:
        Pipeline configuration.
    detector:
        Optional pre-initialised detector.  If None, one is created from config.
        Pass a detector explicitly when processing multiple videos in a batch
        to avoid reloading model weights each time.

    Returns
    -------
    VideoResult
        Contains status, metadata, and error details.
    """
    log_video_start(logger, source_gcs_path)
    sw = Stopwatch()

    src_ext = os.path.splitext(source_gcs_path)[1].lower()
    result = VideoResult(
        source_gcs_path=source_gcs_path,
        output_gcs_path=output_gcs_path,
        status="failed",
    )

    os.makedirs(config.local_temp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=src_ext, dir=config.local_temp_dir, delete=False
    ) as f_in:
        local_input = f_in.name

    with tempfile.NamedTemporaryFile(
        suffix=".mp4", dir=config.local_temp_dir, delete=False
    ) as f_vid:
        local_video_only = f_vid.name

    with tempfile.NamedTemporaryFile(
        suffix=src_ext, dir=config.local_temp_dir, delete=False
    ) as f_out:
        local_output = f_out.name

    try:
        # ── 1. Download ───────────────────────────────────────────────────
        from src import gcs_io

        src_bucket, src_blob = gcs_uri_to_bucket_and_blob(source_gcs_path)
        logger.info(f"Downloading {source_gcs_path} → {local_input}")
        gcs_io.download_blob_to_file(gcs_client, src_bucket, src_blob, local_input)

        # ── 2. Probe ──────────────────────────────────────────────────────
        info: VideoInfo = probe_video(local_input)

        logger.info(
            f"Video info: {info.width}x{info.height} @ {info.fps:.2f}fps "
            f"duration={info.duration_sec:.1f}s codec={info.codec}"
        )
        result.width = info.width
        result.height = info.height
        result.fps = info.fps
        result.duration_sec = info.duration_sec

        # ── 3. Detect + Track + Blur ──────────────────────────────────────
        own_detector = detector is None
        if own_detector:
            detector = make_detector(config)
            detector.warmup()

        try:
            runner = _run_single_pass if config.single_pass else _run_detection_and_blur
            local_video_only = runner(
                local_input=local_input,
                local_video_only=local_video_only,
                info=info,
                config=config,
                detector=detector,
                result=result,
            )
        finally:
            if own_detector:
                detector.close()

        # ── 4. Mux audio ──────────────────────────────────────────────────
        preserve_audio = config.preserve_audio and info.has_audio
        logger.info(
            f"Muxing (preserve_audio={preserve_audio}) → {local_output}"
        )
        mux_audio(
            video_only_path=local_video_only,
            original_video_path=local_input,
            output_path=local_output,
            has_audio=preserve_audio,
            timeout_sec=config.ffmpeg_mux_timeout_sec,
        )

        # ── 5. Upload ──────────────────────────────────────────────────────
        out_bucket, out_blob = gcs_uri_to_bucket_and_blob(output_gcs_path)
        logger.info(f"Uploading {local_output} → {output_gcs_path}")
        gcs_io.upload_file_to_blob(gcs_client, out_bucket, out_blob, local_output)

        result.status = "success"

    except (VideoProbeError, FrameWriteError, OSError, IOError) as exc:
        # Expected operational failures: corrupt video, ffmpeg error, disk/GCS I/O.
        # Log at ERROR (no traceback) — these are anticipated, not bugs.
        logger.error(
            f"Operational failure in process_single_video for {source_gcs_path!r}: "
            f"{type(exc).__name__}: {exc}"
        )
        result.status = "failed"
        result.error_message = f"{type(exc).__name__}: {exc}"

    except Exception as exc:
        # Unexpected failure — log full traceback so bugs are visible.
        logger.exception(
            f"Unexpected error in process_single_video for {source_gcs_path!r}"
        )
        result.status = "failed"
        result.error_message = f"Unexpected: {exc}"

    finally:
        # Clean up temp files
        for path in [local_input, local_video_only, local_output]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass

    elapsed = sw.elapsed()
    log_video_end(logger, source_gcs_path, result.status, elapsed)
    return result


# ── Internal helpers ───────────────────────────────────────────────────────


def _run_detection_and_blur(
    local_input: str,
    local_video_only: str,
    info: VideoInfo,
    config: FaceBlurConfig,
    detector: FaceDetector,
    result: VideoResult,
) -> str:
    """
    Core detect → track → smooth → interpolate → blur → write loop.

    Returns the path to the written video-only (no-audio) file.
    """
    tracker = FaceTracker(
        iou_threshold=config.tracker_iou_threshold,
        persistence_frames=config.track_persistence_frames,
    )
    smoother = BBoxSmoother(window=config.smoothing_window)
    scene_detector = SceneCutDetector(threshold=config.scene_cut_threshold)

    # Per-track bbox history: track_id → {frame_idx: smoothed_bbox}.
    # Stored per-track so interpolation is identity-safe (each track is
    # interpolated independently; no cross-face index matching).
    per_track_frames: Dict[int, Dict[int, BBox]] = defaultdict(dict)

    # ── Pass 1: detect + track ─────────────────────────────────────────────
    logger.info("Pass 1: detection + tracking …")
    total_faces = 0
    frame_idx = -1  # guard for empty video
    # Read actual frame dimensions from the first frame rather than relying on
    # ffprobe metadata.  For videos with a rotation side-data tag (common in
    # .mov files from cameras), ffprobe reports display dimensions while
    # cv2.VideoCapture reads the raw encoded dimensions.  When these disagree,
    # cv2.VideoWriter silently drops every frame → empty output file.
    actual_width: int = info.width
    actual_height: int = info.height

    for frame_idx, frame in iter_frames(local_input):
        if frame_idx == 0:
            actual_height, actual_width = frame.shape[:2]
            if (actual_width, actual_height) != (info.width, info.height):
                logger.warning(
                    f"Frame dimensions {actual_width}x{actual_height} differ from "
                    f"probe dimensions {info.width}x{info.height} — using frame "
                    f"dimensions for VideoWriter (likely rotation metadata)."
                )
        # Scene cut check
        if scene_detector.is_cut(frame):
            tracker.reset()
            smoother.reset_all()

        # Run detector every N frames
        run_detection = (frame_idx % config.detection_interval == 0)
        detections = detector.detect(frame) if run_detection else []

        tracked = tracker.update(frame_idx, detections)

        active_ids = [t.track_id for t in tracked.tracks]
        smoother.remove_stale_tracks(active_ids)

        for track in tracked.tracks:
            # Only anchor frames where detection actually ran AND the track
            # was matched (missed_frames == 0).  Writing stale carry-forward
            # bboxes from non-detection frames fills per_track_frames with
            # frozen positions, which prevents fill_gaps from interpolating
            # face motion — causing the blur region to freeze between
            # detection frames rather than smoothly following the face.
            if run_detection and track.missed_frames == 0:
                sb = smoother.smooth(track.track_id, track.bbox)
                per_track_frames[track.track_id][frame_idx] = sb
                total_faces += 1

    result.frames_processed = frame_idx + 1
    result.faces_detected_total = total_faces
    logger.info(
        f"Pass 1 complete: {result.frames_processed} frames, "
        f"{total_faces} face-frames"
    )

    if total_faces == 0:
        logger.info("No faces detected — copying video without blurring.")

    # ── Interpolation post-pass (per-track, identity-safe) ─────────────────
    # Each track is interpolated independently using fill_gaps().  Tracks
    # present on only one side of a gap are NOT extended across it (they
    # only appear in frames where they were genuinely detected/tracked).
    # This avoids both cross-identity bbox matching and silent face-count
    # truncation that occurred with the previous index-based approach.
    per_frame_bboxes: Dict[int, List[BBox]] = defaultdict(list)
    for tid, track_frames in per_track_frames.items():
        filled = fill_gaps(track_frames, max_gap=config.interpolation_max_gap)
        for fidx, bbox in filled.items():
            per_frame_bboxes[fidx].append(bbox)

    # ── Pass 2: apply blur + write ─────────────────────────────────────────
    logger.info("Pass 2: blur + write …")
    with FrameSink(
        output_path=local_video_only,
        fps=info.fps,
        width=actual_width,
        height=actual_height,
        codec=_INTERMEDIATE_CODEC,
    ) as sink:
        for frame_idx, frame in iter_frames(local_input):
            bboxes = per_frame_bboxes.get(frame_idx, [])
            out_frame = apply_face_blurs(frame, bboxes, config)
            sink.write(out_frame)

    logger.info("Pass 2 complete.")
    return local_video_only


def _run_single_pass(
    local_input: str,
    local_video_only: str,
    info: VideoInfo,
    config,
    detector: FaceDetector,
    result: VideoResult,
) -> str:
    """
    Combined detect → track → smooth → blur → write in a single video scan.

    Reads the video once instead of twice, cutting total I/O time roughly in
    half.  Between detection frames the tracker's linear-motion prediction
    holds face positions; this is equivalent to two-pass interpolation when
    ``detection_interval`` is small (≤ ~12 frames at 30 fps).

    Returns the path to the written video-only (no-audio) file.
    """
    tracker = FaceTracker(
        iou_threshold=config.tracker_iou_threshold,
        persistence_frames=config.track_persistence_frames,
    )
    smoother = BBoxSmoother(window=config.smoothing_window)
    scene_detector = SceneCutDetector(threshold=config.scene_cut_threshold)

    total_faces = 0
    frame_idx = -1
    actual_width: int = info.width
    actual_height: int = info.height

    frame_iter = iter_frames(local_input)

    # Peek at the first frame to get actual encoded dimensions.
    # ffprobe and cv2.VideoCapture can disagree for rotated .mov files; using
    # the real frame shape ensures FrameSink is configured correctly.
    try:
        frame_idx, first_frame = next(frame_iter)
        actual_height, actual_width = first_frame.shape[:2]
        if (actual_width, actual_height) != (info.width, info.height):
            logger.warning(
                f"Frame dimensions {actual_width}x{actual_height} differ from "
                f"probe dimensions {info.width}x{info.height} — using frame "
                f"dimensions for VideoWriter (likely rotation metadata)."
            )
    except StopIteration:
        first_frame = None  # empty video; FrameSink.close() will raise

    def _process(fidx: int, frame) -> "np.ndarray":
        nonlocal total_faces
        if scene_detector.is_cut(frame):
            tracker.reset()
            smoother.reset_all()

        run_det = (fidx % config.detection_interval == 0)
        detections = detector.detect(frame) if run_det else []
        tracked = tracker.update(fidx, detections)

        active_ids = [t.track_id for t in tracked.tracks]
        smoother.remove_stale_tracks(active_ids)

        bboxes: List[BBox] = []
        for track in tracked.tracks:
            sb = smoother.smooth(track.track_id, track.bbox)
            bboxes.append(sb)
            if run_det and track.missed_frames == 0:
                total_faces += 1

        return apply_face_blurs(frame, bboxes, config)

    logger.info("Single pass: detect + blur + write …")
    with FrameSink(
        output_path=local_video_only,
        fps=info.fps,
        width=actual_width,
        height=actual_height,
        codec=_INTERMEDIATE_CODEC,
    ) as sink:
        if first_frame is not None:
            sink.write(_process(frame_idx, first_frame))
        for frame_idx, frame in frame_iter:
            sink.write(_process(frame_idx, frame))

    result.frames_processed = frame_idx + 1
    result.faces_detected_total = total_faces
    logger.info(
        f"Single pass complete: {result.frames_processed} frames, "
        f"{total_faces} face-frames"
    )
    return local_video_only
