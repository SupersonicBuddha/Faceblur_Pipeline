"""Tests for video discovery and skip logic based on manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from src.config import FaceBlurConfig
from src.video_inventory import VideoItem, discover_videos


@dataclass
class _FakeBlob:
    name: str  # blob name as returned by GCS, e.g. "for_fb/v.mp4"


class _FakeGCSClient:
    """Fake GCS client that returns a pre-defined blob list."""

    def __init__(self, blobs: List[_FakeBlob]) -> None:
        self._blobs = blobs

    def list_blobs(self, bucket, prefix="", **_):
        return iter(self._blobs)

    def bucket(self, name: str):
        return MagicMock()


def _client(blob_names: List[str]) -> _FakeGCSClient:
    return _FakeGCSClient([_FakeBlob(n) for n in blob_names])


class TestDiscoverVideos:
    def test_discovers_mp4(self):
        cfg = FaceBlurConfig()
        client = _client(["for_fb/video.mp4"])
        videos = discover_videos(client, cfg)
        assert len(videos) == 1
        assert videos[0].source_gcs_path == "gs://tbd-qa/for_fb/video.mp4"
        assert videos[0].source_extension == ".mp4"

    def test_discovers_mov(self):
        cfg = FaceBlurConfig()
        client = _client(["for_fb/clip.mov"])
        videos = discover_videos(client, cfg)
        assert len(videos) == 1
        assert videos[0].source_extension == ".mov"

    def test_skips_non_video(self):
        cfg = FaceBlurConfig()
        client = _client(["for_fb/readme.txt", "for_fb/video.mp4"])
        videos = discover_videos(client, cfg)
        assert len(videos) == 1

    def test_skips_directory_markers(self):
        cfg = FaceBlurConfig()
        client = _client(["for_fb/subdir/", "for_fb/video.mp4"])
        videos = discover_videos(client, cfg)
        assert len(videos) == 1

    def test_empty_folder(self):
        cfg = FaceBlurConfig()
        client = _client([])
        videos = discover_videos(client, cfg)
        assert videos == []

    def test_output_path_naming(self):
        cfg = FaceBlurConfig()
        client = _client(["for_fb/foo/bar/session.mp4"])
        videos = discover_videos(client, cfg)
        assert videos[0].output_gcs_path == "gs://tbd-qa/post_fb/foo/bar/session_blurred.mp4"

    def test_sorted_by_source_path(self):
        cfg = FaceBlurConfig()
        client = _client(["for_fb/b.mp4", "for_fb/a.mp4", "for_fb/c.mp4"])
        videos = discover_videos(client, cfg)
        paths = [v.source_gcs_path for v in videos]
        assert paths == sorted(paths)

    def test_case_insensitive_extension(self):
        cfg = FaceBlurConfig()
        client = _client(["for_fb/video.MP4", "for_fb/clip.MOV"])
        videos = discover_videos(client, cfg)
        assert len(videos) == 2
