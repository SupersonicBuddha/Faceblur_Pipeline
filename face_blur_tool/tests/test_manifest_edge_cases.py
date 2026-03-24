"""
Edge-case tests for manifest serialisation and GCSManifest loading.

Covers gaps not hit by test_manifest.py:
- Malformed JSON lines are silently skipped on load (not fatal)
- Whitespace-only lines in JSONL are skipped
- Empty ManifestIndex serialises to "" (no trailing newline)
- ManifestRecord with all-None optional metadata round-trips cleanly
- Large manifest (100 records) loads and deduplicates correctly
- summary() on empty manifest returns an empty dict
- ManifestRecord.from_dict tolerates extra/unknown keys (forward-compat)
- GCSManifest.load handles a manifest where every line is malformed
"""

from __future__ import annotations

import json
from typing import Dict

import pytest

from src.constants import STATUS_FAILED, STATUS_SUCCESS
from src.manifest import GCSManifest, ManifestIndex, ManifestRecord


# ── Fake GCS (identical to test_manifest.py, minimal) ──────────────────────


class _FakeBlob:
    def __init__(self, key: str, store: Dict) -> None:
        self._key = key
        self._store = store

    def exists(self) -> bool:
        return self._key in self._store

    def download_as_text(self, encoding: str = "utf-8") -> str:
        return self._store.get(self._key, "")

    def upload_from_string(self, data: bytes, content_type: str = "") -> None:
        self._store[self._key] = data.decode("utf-8") if isinstance(data, bytes) else data

    def download_to_filename(self, path: str) -> None:
        pass

    def upload_from_filename(self, path: str, content_type: str = "") -> None:
        pass


class _FakeBucket:
    def __init__(self, name: str, store: Dict) -> None:
        self._name = name
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(f"{self._name}/{name}", self._store)


class FakeGCSClient:
    def __init__(self, store: Dict = None) -> None:
        self._store: Dict = store if store is not None else {}

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(name, self._store)


def _manifest(store: Dict = None) -> GCSManifest:
    client = FakeGCSClient(store=store)
    return GCSManifest(
        gcs_client=client,
        bucket_name="tbd-qa",
        blob_name="post_fb/manifest.jsonl",
    )


# ── ManifestIndex edge cases ────────────────────────────────────────────────


class TestManifestIndexEdgeCases:

    def test_empty_index_to_jsonl_is_empty_string(self):
        """to_jsonl() on an empty index must return '' (no spurious newlines)."""
        idx = ManifestIndex()
        assert idx.to_jsonl() == ""

    def test_single_record_to_jsonl_ends_with_newline(self):
        """A non-empty index must end with a newline so JSONL is appendable."""
        rec = ManifestRecord(
            source_gcs_path="gs://b/v.mp4",
            output_gcs_path="gs://b/o/v_blurred.mp4",
            status=STATUS_SUCCESS,
        )
        idx = ManifestIndex([rec])
        assert idx.to_jsonl().endswith("\n")

    def test_summary_on_empty_manifest_returns_empty_dict(self):
        m = _manifest()
        m.load()
        assert m.summary() == {}

    def test_100_records_all_distinct_loaded(self):
        records = [
            ManifestRecord(
                source_gcs_path=f"gs://b/v{i}.mp4",
                output_gcs_path=f"gs://b/o/v{i}_blurred.mp4",
                status=STATUS_SUCCESS,
            )
            for i in range(100)
        ]
        idx = ManifestIndex(records)
        assert len(idx.all_records()) == 100

    def test_100_duplicate_records_collapse_to_one(self):
        """Upserting the same key 100 times keeps only the last write."""
        records = [
            ManifestRecord(
                source_gcs_path="gs://b/v.mp4",
                output_gcs_path="gs://b/o/v_blurred.mp4",
                status=STATUS_FAILED if i < 99 else STATUS_SUCCESS,
            )
            for i in range(100)
        ]
        idx = ManifestIndex(records)
        assert len(idx.all_records()) == 1
        assert idx.is_already_successful("gs://b/v.mp4")

    def test_get_unknown_path_returns_none(self):
        idx = ManifestIndex()
        assert idx.get("gs://b/nonexistent.mp4") is None


# ── ManifestRecord serialisation edge cases ─────────────────────────────────


class TestManifestRecordEdgeCases:

    def test_all_none_optional_fields_roundtrip(self):
        """
        A record with every optional field set to None must survive
        to_json() → from_dict() without raising or changing to non-None.
        """
        rec = ManifestRecord(
            source_gcs_path="gs://b/v.mp4",
            output_gcs_path="gs://b/o/v_blurred.mp4",
            status=STATUS_FAILED,
            error_message=None,
            duration_sec=None,
            width=None,
            height=None,
            fps=None,
        )
        d = json.loads(rec.to_json())
        rec2 = ManifestRecord.from_dict(d)
        assert rec2.error_message is None
        assert rec2.duration_sec is None
        assert rec2.width is None
        assert rec2.height is None
        assert rec2.fps is None

    def test_from_dict_ignores_unknown_keys(self):
        """
        Future schema additions must not crash old from_dict() calls.
        from_dict uses .get() with defaults, so unknown keys in the dict
        are simply ignored.
        """
        d = {
            "source_gcs_path": "gs://b/v.mp4",
            "output_gcs_path": "gs://b/o/v_blurred.mp4",
            "status": "success",
            "future_field_unknown": "some_value",
            "another_new_field": 42,
        }
        # Must not raise
        rec = ManifestRecord.from_dict(d)
        assert rec.status == "success"

    def test_error_message_with_special_chars_roundtrips(self):
        """Error messages may contain quotes, newlines, and unicode."""
        msg = 'ffmpeg error: "codec not found"\nstderr: línea 1'
        rec = ManifestRecord(
            source_gcs_path="gs://b/v.mp4",
            output_gcs_path="gs://b/o.mp4",
            status=STATUS_FAILED,
            error_message=msg,
        )
        rec2 = ManifestRecord.from_dict(json.loads(rec.to_json()))
        assert rec2.error_message == msg


# ── GCSManifest load robustness ──────────────────────────────────────────────


class TestGCSManifestLoadRobustness:

    def _store_with_content(self, content: str) -> Dict:
        return {"tbd-qa/post_fb/manifest.jsonl": content}

    def test_malformed_json_line_is_skipped_not_fatal(self):
        """
        A single corrupt line must be skipped; valid lines around it are loaded.
        """
        good = ManifestRecord(
            source_gcs_path="gs://b/good.mp4",
            output_gcs_path="gs://b/o/good_blurred.mp4",
            status=STATUS_SUCCESS,
        )
        content = good.to_json() + "\n{NOT VALID JSON}\n"
        m = _manifest(self._store_with_content(content))
        m.load()
        # The valid record survives; the corrupt line is dropped silently.
        assert m.is_already_successful("gs://b/good.mp4")

    def test_all_malformed_lines_produce_empty_manifest(self):
        """If every line is malformed, the result is an empty manifest — not a crash."""
        content = "{bad json\n}{also bad\n"
        m = _manifest(self._store_with_content(content))
        m.load()  # must not raise
        assert m.summary() == {}

    def test_whitespace_only_lines_are_skipped(self):
        """
        Blank lines and lines containing only whitespace must not produce
        JSON parse errors or empty records.
        """
        good = ManifestRecord(
            source_gcs_path="gs://b/v.mp4",
            output_gcs_path="gs://b/o/v_blurred.mp4",
            status=STATUS_SUCCESS,
        )
        content = "\n   \n" + good.to_json() + "\n\n"
        m = _manifest(self._store_with_content(content))
        m.load()
        assert m.is_already_successful("gs://b/v.mp4")

    def test_load_mixed_valid_and_malformed_lines(self):
        """
        Valid lines before and after a malformed line must all be loaded.
        """
        r1 = ManifestRecord("gs://b/a.mp4", "gs://b/o/a_blurred.mp4", STATUS_SUCCESS)
        r2 = ManifestRecord("gs://b/b.mp4", "gs://b/o/b_blurred.mp4", STATUS_FAILED)
        content = r1.to_json() + "\n{BAD}\n" + r2.to_json() + "\n"
        m = _manifest(self._store_with_content(content))
        m.load()
        assert m.is_already_successful("gs://b/a.mp4")
        assert not m.is_already_successful("gs://b/b.mp4")  # failed, not success

    def test_duplicate_lines_in_stored_jsonl_last_wins(self):
        """
        If the same key appears twice in the stored JSONL (e.g. from a
        partial-write race), the last occurrence wins after load().
        """
        r_fail = ManifestRecord("gs://b/v.mp4", "gs://b/o.mp4", STATUS_FAILED)
        r_ok   = ManifestRecord("gs://b/v.mp4", "gs://b/o.mp4", STATUS_SUCCESS)
        content = r_fail.to_json() + "\n" + r_ok.to_json() + "\n"
        m = _manifest(self._store_with_content(content))
        m.load()
        assert m.is_already_successful("gs://b/v.mp4")

    def test_record_with_missing_required_key_is_skipped(self):
        """
        A line missing 'source_gcs_path' (required key) must be skipped,
        not crash the load.
        """
        bad = json.dumps({"output_gcs_path": "gs://b/o.mp4", "status": "success"})
        good = ManifestRecord("gs://b/ok.mp4", "gs://b/o/ok_blurred.mp4", STATUS_SUCCESS)
        content = bad + "\n" + good.to_json() + "\n"
        m = _manifest(self._store_with_content(content))
        m.load()
        assert m.is_already_successful("gs://b/ok.mp4")
