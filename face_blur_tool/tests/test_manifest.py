"""
Tests for manifest parsing, skip logic, and GCSManifest behaviour.

GCS interactions are faked with an in-memory store so no real credentials
are needed.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.constants import STATUS_FAILED, STATUS_SUCCESS
from src.manifest import GCSManifest, ManifestIndex, ManifestRecord


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_record(
    path: str = "gs://b/v.mp4",
    status: str = STATUS_SUCCESS,
    output: str = "gs://b/o/v_blurred.mp4",
) -> ManifestRecord:
    return ManifestRecord(
        source_gcs_path=path,
        output_gcs_path=output,
        status=status,
    )


# ── ManifestIndex unit tests ───────────────────────────────────────────────


class TestManifestIndex:
    def test_empty_index_not_successful(self):
        idx = ManifestIndex()
        assert not idx.is_already_successful("gs://b/v.mp4")

    def test_success_record_is_successful(self):
        rec = _make_record(status=STATUS_SUCCESS)
        idx = ManifestIndex([rec])
        assert idx.is_already_successful(rec.source_gcs_path)

    def test_failed_record_is_not_successful(self):
        rec = _make_record(status=STATUS_FAILED)
        idx = ManifestIndex([rec])
        assert not idx.is_already_successful(rec.source_gcs_path)

    def test_upsert_replaces_existing(self):
        rec_fail = _make_record(status=STATUS_FAILED)
        idx = ManifestIndex([rec_fail])
        rec_ok = _make_record(status=STATUS_SUCCESS)
        idx.upsert(rec_ok)
        assert idx.is_already_successful(rec_ok.source_gcs_path)

    def test_all_records_returns_all(self):
        recs = [_make_record(path=f"gs://b/v{i}.mp4") for i in range(5)]
        idx = ManifestIndex(recs)
        assert len(idx.all_records()) == 5

    def test_to_jsonl_roundtrips(self):
        recs = [_make_record(path=f"gs://b/v{i}.mp4") for i in range(3)]
        idx = ManifestIndex(recs)
        jsonl = idx.to_jsonl()
        lines = [l for l in jsonl.splitlines() if l.strip()]
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert "source_gcs_path" in data
            assert "status" in data

    def test_duplicate_key_last_write_wins(self):
        r1 = ManifestRecord(
            source_gcs_path="gs://b/v.mp4",
            output_gcs_path="gs://b/out.mp4",
            status=STATUS_FAILED,
        )
        r2 = ManifestRecord(
            source_gcs_path="gs://b/v.mp4",
            output_gcs_path="gs://b/out.mp4",
            status=STATUS_SUCCESS,
        )
        idx = ManifestIndex([r1, r2])
        assert idx.is_already_successful("gs://b/v.mp4")


# ── ManifestRecord serialisation ──────────────────────────────────────────


class TestManifestRecord:
    def test_to_json_and_from_dict_roundtrip(self):
        rec = ManifestRecord(
            source_gcs_path="gs://b/v.mp4",
            output_gcs_path="gs://b/o/v_blurred.mp4",
            status=STATUS_SUCCESS,
            retry_count=1,
            model_version="v1.0.0",
            source_extension=".mp4",
            duration_sec=42.5,
            width=1920,
            height=1080,
            fps=29.97,
        )
        d = json.loads(rec.to_json())
        rec2 = ManifestRecord.from_dict(d)
        assert rec2.source_gcs_path == rec.source_gcs_path
        assert rec2.status == rec.status
        assert rec2.retry_count == rec.retry_count
        assert rec2.width == rec.width
        assert abs(rec2.fps - rec.fps) < 0.01

    def test_from_dict_handles_missing_optional_fields(self):
        d = {
            "source_gcs_path": "gs://b/v.mp4",
            "output_gcs_path": "gs://b/o/v_blurred.mp4",
            "status": "success",
        }
        rec = ManifestRecord.from_dict(d)
        assert rec.error_message is None
        assert rec.duration_sec is None


# ── GCSManifest with fake GCS ──────────────────────────────────────────────


class FakeGCSClient:
    """Minimal fake that stores blobs as strings in a dict."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def bucket(self, name: str):
        return _FakeBucket(name, self._store)


class _FakeBucket:
    def __init__(self, name: str, store: Dict[str, str]) -> None:
        self._name = name
        self._store = store

    def blob(self, blob_name: str):
        return _FakeBlob(self._name, blob_name, self._store)


class _FakeBlob:
    def __init__(self, bucket: str, name: str, store: Dict[str, str]) -> None:
        self._key = f"{bucket}/{name}"
        self._store = store

    def exists(self) -> bool:
        return self._key in self._store

    def download_as_text(self, encoding: str = "utf-8") -> str:
        return self._store.get(self._key, "")

    def upload_from_string(self, data: bytes, content_type: str = "") -> None:
        self._store[self._key] = data.decode("utf-8")

    def download_to_filename(self, path: str) -> None:  # not used in manifest tests
        pass

    def upload_from_filename(self, path: str, content_type: str = "") -> None:
        pass


class TestGCSManifest:
    def _make_manifest(self, store=None):
        client = FakeGCSClient()
        if store is not None:
            client._store = store
        return GCSManifest(
            gcs_client=client,
            bucket_name="tbd-qa",
            blob_name="post_fb/processed_manifest.jsonl",
        ), client

    def test_load_empty_manifest(self):
        m, _ = self._make_manifest()
        m.load()  # no file in store → should not raise

    def test_record_success_then_skip(self):
        m, _ = self._make_manifest()
        m.load()
        m.record_success(
            source_gcs_path="gs://tbd-qa/for_fb/v.mp4",
            output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
        )
        assert m.is_already_successful("gs://tbd-qa/for_fb/v.mp4")

    def test_record_failure_not_skipped(self):
        m, _ = self._make_manifest()
        m.load()
        m.record_failure(
            source_gcs_path="gs://tbd-qa/for_fb/v.mp4",
            output_gcs_path="gs://tbd-qa/post_fb/v_blurred.mp4",
            error_message="boom",
        )
        assert not m.is_already_successful("gs://tbd-qa/for_fb/v.mp4")

    def test_failure_then_success_is_skipped(self):
        m, _ = self._make_manifest()
        m.load()
        path = "gs://tbd-qa/for_fb/v.mp4"
        m.record_failure(path, "gs://tbd-qa/post_fb/v_blurred.mp4", "err")
        m.record_success(path, "gs://tbd-qa/post_fb/v_blurred.mp4")
        assert m.is_already_successful(path)

    def test_summary(self):
        m, _ = self._make_manifest()
        m.load()
        m.record_success("gs://b/v1.mp4", "gs://b/o/v1_blurred.mp4")
        m.record_success("gs://b/v2.mp4", "gs://b/o/v2_blurred.mp4")
        m.record_failure("gs://b/v3.mp4", "gs://b/o/v3_blurred.mp4", "err")
        s = m.summary()
        assert s["success"] == 2
        assert s["failed"] == 1

    def test_persists_to_gcs_store(self):
        m, client = self._make_manifest()
        m.load()
        m.record_success("gs://tbd-qa/for_fb/v.mp4", "gs://tbd-qa/post_fb/v_blurred.mp4")
        key = "tbd-qa/post_fb/processed_manifest.jsonl"
        assert key in client._store
        content = client._store[key]
        assert "gs://tbd-qa/for_fb/v.mp4" in content

    def test_load_preexisting_records(self):
        rec = ManifestRecord(
            source_gcs_path="gs://tbd-qa/for_fb/existing.mp4",
            output_gcs_path="gs://tbd-qa/post_fb/existing_blurred.mp4",
            status=STATUS_SUCCESS,
        )
        store = {
            "tbd-qa/post_fb/processed_manifest.jsonl": rec.to_json() + "\n"
        }
        m, _ = self._make_manifest(store=store)
        m.load()
        assert m.is_already_successful("gs://tbd-qa/for_fb/existing.mp4")
