"""
Smoke tests for the batch runner.

Tests GCS discovery, manifest skip logic, success/failure recording,
retry-on-failure behaviour, and batch-level resilience (one failure does
not abort the batch).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.config import FaceBlurConfig
from src.constants import STATUS_FAILED, STATUS_SUCCESS
from src.manifest import GCSManifest, ManifestRecord
from src.pipeline.batch_runner import BatchSummary, run_batch
from src.pipeline.result_types import VideoResult


# ── Fake GCS infrastructure ────────────────────────────────────────────────


@dataclass
class _FakeBlob:
    name: str


class _FakeGCSBlobResult:
    def __init__(self, key, store):
        self._key = key
        self._store = store

    def exists(self):
        return self._key in self._store

    def download_as_text(self, encoding="utf-8"):
        val = self._store.get(self._key, "")
        return val if isinstance(val, str) else val.decode("utf-8")

    def upload_from_string(self, data: bytes, content_type=""):
        self._store[self._key] = data.decode("utf-8")

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
        return _FakeGCSBlobResult(f"{self._name}/{name}", self._store)


class FakeGCSClient:
    def __init__(self, blobs: List[str], store: Optional[Dict] = None):
        self._blobs = [_FakeBlob(b) for b in blobs]
        self._store: Dict = store or {}

    def list_blobs(self, bucket, prefix="", **_):
        return iter(self._blobs)

    def bucket(self, name):
        return _FakeBucket(name, self._store)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_config(tmp_path) -> FaceBlurConfig:
    return FaceBlurConfig(local_temp_dir=str(tmp_path / "tmp"), retry_count=1)


def _success_result(source, output):
    return VideoResult(
        source_gcs_path=source,
        output_gcs_path=output,
        status="success",
        duration_sec=5.0,
        width=1280,
        height=720,
        fps=30.0,
    )


def _failed_result(source, output, msg="deliberate failure"):
    return VideoResult(
        source_gcs_path=source,
        output_gcs_path=output,
        status="failed",
        error_message=msg,
    )


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestBatchRunnerSmoke:

    def test_empty_bucket_produces_zero_counts(self, tmp_path):
        config = _make_config(tmp_path)
        client = FakeGCSClient([])
        with patch("src.pipeline.batch_runner.make_detector") as mock_det:
            mock_det.return_value = MagicMock()
            summary = run_batch(client, config)
        assert summary.total_discovered == 0
        assert summary.succeeded == 0

    def test_already_successful_video_is_skipped(self, tmp_path):
        config = _make_config(tmp_path)
        src_path = "gs://tbd-qa/for_fb/v.mp4"

        # Pre-load a success record in the manifest store
        rec = ManifestRecord(
            source_gcs_path=src_path,
            output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
            status=STATUS_SUCCESS,
        )
        store = {"tbd-qa/post_fb/processed_manifest.jsonl": rec.to_json() + "\n"}
        client = FakeGCSClient(["for_fb/v.mp4"], store=store)

        with patch("src.pipeline.batch_runner.make_detector") as mock_det:
            mock_det.return_value = MagicMock()
            summary = run_batch(client, config)

        assert summary.skipped == 1
        assert summary.succeeded == 0

    def test_successful_video_recorded_in_manifest(self, tmp_path):
        config = _make_config(tmp_path)
        client = FakeGCSClient(["for_fb/v.mp4"])

        def _always_success(source_gcs_path, output_gcs_path, **_):
            return _success_result(source_gcs_path, output_gcs_path)

        with patch("src.pipeline.batch_runner.make_detector") as mock_det, \
             patch(
                 "src.pipeline.batch_runner.process_single_video",
                 side_effect=_always_success,
             ):
            mock_det.return_value = MagicMock()
            summary = run_batch(client, config)

        assert summary.succeeded == 1
        assert summary.failed == 0

    def test_failed_video_recorded_and_batch_continues(self, tmp_path):
        """
        Video a.mp4 fails on every attempt (both initial + retry).
        Video b.mp4 succeeds.
        Batch must not abort — it should record a.mp4 as failed and b.mp4 as
        succeeded, giving failed=1, succeeded=1.
        """
        config = _make_config(tmp_path)  # retry_count=1 → 2 total attempts

        def _mixed(source_gcs_path, output_gcs_path, **_):
            if "a.mp4" in source_gcs_path:
                # Always return failed regardless of retry
                return _failed_result(source_gcs_path, output_gcs_path)
            return _success_result(source_gcs_path, output_gcs_path)

        client = FakeGCSClient(["for_fb/a.mp4", "for_fb/b.mp4"])

        with patch("src.pipeline.batch_runner.make_detector") as mock_det, \
             patch(
                 "src.pipeline.batch_runner.process_single_video",
                 side_effect=_mixed,
             ):
            mock_det.return_value = MagicMock()
            summary = run_batch(client, config)

        assert summary.total_discovered == 2
        assert summary.failed == 1
        assert summary.succeeded == 1

    def test_retry_succeeds_on_second_attempt(self, tmp_path):
        """
        First attempt for a video returns failed; second attempt returns success.
        Batch must record success with retry_count=1.
        """
        config = _make_config(tmp_path)  # retry_count=1

        call_count = [0]

        def _fail_then_succeed(source_gcs_path, output_gcs_path, **_):
            call_count[0] += 1
            if call_count[0] == 1:
                return _failed_result(source_gcs_path, output_gcs_path, "transient error")
            return _success_result(source_gcs_path, output_gcs_path)

        client = FakeGCSClient(["for_fb/v.mp4"])

        with patch("src.pipeline.batch_runner.make_detector") as mock_det, \
             patch(
                 "src.pipeline.batch_runner.process_single_video",
                 side_effect=_fail_then_succeed,
             ):
            mock_det.return_value = MagicMock()
            summary = run_batch(client, config)

        assert summary.succeeded == 1
        assert summary.failed == 0
        assert call_count[0] == 2, "Expected exactly 2 attempts (first fail, second succeed)"

        successful_result = next(r for r in summary.results if r.succeeded)
        assert successful_result.retry_count == 1

    def test_retry_fails_both_attempts_records_final_error(self, tmp_path):
        """
        Both attempts fail.  The manifest entry must reflect the final failure
        with the correct retry_count and a non-empty error_message.
        """
        config = _make_config(tmp_path)  # retry_count=1

        call_count = [0]

        def _always_fail(source_gcs_path, output_gcs_path, **_):
            call_count[0] += 1
            return _failed_result(
                source_gcs_path, output_gcs_path,
                msg=f"attempt {call_count[0]} failed",
            )

        client = FakeGCSClient(["for_fb/v.mp4"])

        with patch("src.pipeline.batch_runner.make_detector") as mock_det, \
             patch(
                 "src.pipeline.batch_runner.process_single_video",
                 side_effect=_always_fail,
             ):
            mock_det.return_value = MagicMock()
            summary = run_batch(client, config)

        assert summary.failed == 1
        assert summary.succeeded == 0
        assert call_count[0] == 2, "Expected 2 attempts (initial + 1 retry)"

        failed_result = next(r for r in summary.results if r.failed)
        assert failed_result.retry_count == 1
        assert failed_result.error_message is not None
        assert "attempt 2" in failed_result.error_message

    def test_summary_counts_are_consistent_with_results(self, tmp_path):
        """BatchSummary counters must always match the actual results list."""
        config = _make_config(tmp_path)

        def _mixed(source_gcs_path, output_gcs_path, **_):
            if "fail" in source_gcs_path:
                return _failed_result(source_gcs_path, output_gcs_path)
            return _success_result(source_gcs_path, output_gcs_path)

        client = FakeGCSClient([
            "for_fb/ok1.mp4",
            "for_fb/fail1.mp4",
            "for_fb/ok2.mp4",
            "for_fb/fail2.mp4",
        ])

        with patch("src.pipeline.batch_runner.make_detector") as mock_det, \
             patch(
                 "src.pipeline.batch_runner.process_single_video",
                 side_effect=_mixed,
             ):
            mock_det.return_value = MagicMock()
            summary = run_batch(client, config)

        actual_succeeded = sum(1 for r in summary.results if r.succeeded)
        actual_failed = sum(1 for r in summary.results if r.failed)
        assert summary.succeeded == actual_succeeded
        assert summary.failed == actual_failed
        assert summary.succeeded + summary.failed + summary.skipped == summary.total_discovered
