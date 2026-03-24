"""
Batch runner: orchestrates processing of all videos in the input GCS prefix.

Responsibilities
----------------
1. Discover all eligible videos (video_inventory).
2. Load the manifest and skip already-successful videos.
3. For each remaining video, call process_single_video with retry wrapping.
4. Update the manifest after each video.
5. Print a summary on completion.

The detector is initialised once per batch run and reused across all videos
to avoid reloading model weights on each video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.config import FaceBlurConfig
from src.constants import STATUS_FAILED, STATUS_SKIPPED, STATUS_SUCCESS
from src.detectors.factory import make_detector
from src.logging_utils import get_logger
from src.manifest import GCSManifest
from src.pipeline.process_video import process_single_video
from src.pipeline.result_types import VideoResult
from src.video_inventory import VideoItem, discover_videos

logger = get_logger(__name__)


@dataclass
class BatchSummary:
    """Summary statistics returned after a batch run."""

    total_discovered: int = 0
    skipped: int = 0
    succeeded: int = 0
    failed: int = 0
    results: List[VideoResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  FACE BLUR BATCH SUMMARY")
        print("=" * 60)
        print(f"  Discovered : {self.total_discovered}")
        print(f"  Skipped    : {self.skipped}  (already processed)")
        print(f"  Succeeded  : {self.succeeded}")
        print(f"  Failed     : {self.failed}")
        print("=" * 60)
        if self.failed:
            print("\n  Failed videos:")
            for r in self.results:
                if r.failed:
                    print(f"    • {r.source_gcs_path}")
                    if r.error_message:
                        print(f"      error: {r.error_message[:120]}")
        print()


def run_batch(
    gcs_client,
    config: FaceBlurConfig,
    dry_run: bool = False,
) -> BatchSummary:
    """
    Run the full batch face-blur pipeline.

    Parameters
    ----------
    gcs_client:
        google.cloud.storage.Client or compatible fake.
    config:
        Pipeline configuration.
    dry_run:
        If True, discover and print videos but do not process them.

    Returns
    -------
    BatchSummary
    """
    summary = BatchSummary()

    # ── 1. Load manifest ───────────────────────────────────────────────────
    manifest = GCSManifest(
        gcs_client=gcs_client,
        bucket_name=config.manifest_bucket,
        blob_name=config.manifest_blob,
        model_version=config.model_version,
    )
    manifest.load()

    # ── 2. Discover videos ─────────────────────────────────────────────────
    videos: List[VideoItem] = discover_videos(gcs_client, config)
    summary.total_discovered = len(videos)

    if not videos:
        logger.info("No videos discovered. Nothing to do.")
        summary.print_summary()
        return summary

    if dry_run:
        logger.info(f"Dry run — would process {len(videos)} video(s):")
        for v in videos:
            print(f"  {v.source_gcs_path}")
        return summary

    # ── 3. Initialise detector (once for the whole batch) ─────────────────
    logger.info(f"Initialising detector: {config.detector_type}")
    detector = make_detector(config)
    detector.warmup()

    try:
        # ── 4. Process each video ──────────────────────────────────────────
        for i, video in enumerate(videos, start=1):
            logger.info(
                f"[{i}/{len(videos)}] Processing {video.source_gcs_path}"
            )

            # Skip check
            if manifest.is_already_successful(video.source_gcs_path):
                logger.info(f"  → Skipping (already successful)")
                summary.skipped += 1
                result = VideoResult(
                    source_gcs_path=video.source_gcs_path,
                    output_gcs_path=video.output_gcs_path,
                    status=STATUS_SKIPPED,
                )
                summary.results.append(result)
                continue

            # Process with retry
            result = _process_with_retry(
                video=video,
                gcs_client=gcs_client,
                config=config,
                detector=detector,
            )
            summary.results.append(result)

            # Update manifest
            if result.succeeded:
                summary.succeeded += 1
                manifest.record_success(
                    source_gcs_path=result.source_gcs_path,
                    output_gcs_path=result.output_gcs_path,
                    retry_count=result.retry_count,
                    source_extension=video.source_extension,
                    duration_sec=result.duration_sec,
                    width=result.width,
                    height=result.height,
                    fps=result.fps,
                )
            else:
                summary.failed += 1
                manifest.record_failure(
                    source_gcs_path=result.source_gcs_path,
                    output_gcs_path=result.output_gcs_path,
                    error_message=result.error_message or "unknown",
                    retry_count=result.retry_count,
                    source_extension=video.source_extension,
                )

    finally:
        detector.close()

    summary.print_summary()
    return summary


def _process_with_retry(
    video: VideoItem,
    gcs_client,
    config: FaceBlurConfig,
    detector,
) -> VideoResult:
    """
    Call process_single_video, retrying up to ``config.retry_count`` times.

    Retry is triggered when the result has ``status="failed"`` **or** when
    the call raises unexpectedly.  ``process_single_video`` is designed to
    catch all errors internally and return a failed VideoResult, so the
    exception path is a safety net for unexpected situations.

    Returns a VideoResult with ``retry_count`` set to the zero-based attempt
    index on which processing concluded.  The batch never propagates
    exceptions — all failures are captured in the returned VideoResult.
    """
    last_result: Optional[VideoResult] = None

    for attempt in range(config.retry_count + 1):
        try:
            result = process_single_video(
                source_gcs_path=video.source_gcs_path,
                output_gcs_path=video.output_gcs_path,
                gcs_client=gcs_client,
                config=config,
                detector=detector,
            )
        except Exception as exc:
            # Unexpected raise from process_single_video (should not happen).
            logger.warning(
                f"Attempt {attempt + 1} raised unexpectedly for "
                f"{video.source_gcs_path!r}: {exc!r}"
            )
            result = VideoResult(
                source_gcs_path=video.source_gcs_path,
                output_gcs_path=video.output_gcs_path,
                status=STATUS_FAILED,
                error_message=str(exc),
            )

        result.retry_count = attempt

        if result.succeeded:
            return result

        last_result = result
        if attempt < config.retry_count:
            logger.warning(
                f"Attempt {attempt + 1}/{config.retry_count + 1} failed for "
                f"{video.source_gcs_path!r}: {result.error_message!r}. Retrying…"
            )

    # All attempts exhausted — return the last failed result.
    logger.error(
        f"All {config.retry_count + 1} attempt(s) failed for "
        f"{video.source_gcs_path!r}. Final error: {last_result.error_message!r}"  # type: ignore[union-attr]
    )
    return last_result  # type: ignore[return-value]
