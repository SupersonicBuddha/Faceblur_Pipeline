"""
Discover videos in GCS and resolve their output paths.

Kept separate so the discovery logic can be tested without real GCS access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List

from src.config import FaceBlurConfig
from src.logging_utils import get_logger
from src.utils.paths import build_output_gcs_path

logger = get_logger(__name__)


@dataclass(frozen=True)
class VideoItem:
    """A single video discovered in GCS, with its resolved output path."""

    source_gcs_path: str
    output_gcs_path: str
    source_extension: str  # e.g. ".mp4"


def discover_videos(
    gcs_client,
    config: FaceBlurConfig,
) -> List[VideoItem]:
    """
    List all eligible video blobs under ``config.input_prefix``.

    Returns a list of :class:`VideoItem` sorted by source path.
    Directories (blob names ending in '/') are silently skipped.
    Only blobs with a supported extension are included.
    """
    from src import gcs_io

    items: List[VideoItem] = []
    ext_set = {e.lower() for e in config.supported_extensions}

    for blob in gcs_io.list_blobs(
        gcs_client, config.input_bucket, config.input_blob_prefix
    ):
        name: str = blob.name  # e.g. "for_fb/foo/video.mp4"
        if name.endswith("/"):
            continue  # skip directory markers
        _, ext = os.path.splitext(name)
        if ext.lower() not in ext_set:
            logger.debug(f"Skipping non-video blob: gs://{config.input_bucket}/{name}")
            continue

        source_uri = f"gs://{config.input_bucket}/{name}"
        output_uri = build_output_gcs_path(
            source_uri=source_uri,
            input_prefix=config.input_prefix,
            output_prefix=config.output_prefix,
        )
        items.append(
            VideoItem(
                source_gcs_path=source_uri,
                output_gcs_path=output_uri,
                source_extension=ext.lower(),
            )
        )

    items.sort(key=lambda v: v.source_gcs_path)
    logger.info(
        f"Discovered {len(items)} video(s) under {config.input_prefix}"
    )
    return items
