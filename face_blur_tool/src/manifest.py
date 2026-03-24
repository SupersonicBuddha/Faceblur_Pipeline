"""
Manifest read/write logic.

The manifest is stored as JSONL at gs://tbd-qa/post_fb/processed_manifest.jsonl.
Each line is one JSON record.  The unique key per record is ``source_gcs_path``.

Reads load all lines into memory (the manifest stays small).
Writes download-mutate-upload (GCS has no native append / partial update).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.constants import STATUS_FAILED, STATUS_SUCCESS
from src.logging_utils import get_logger, log_manifest_update

logger = get_logger(__name__)


# ── Record dataclass ───────────────────────────────────────────────────────


@dataclass
class ManifestRecord:
    source_gcs_path: str
    output_gcs_path: str
    status: str  # "success" | "failed" | "skipped"
    processed_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    retry_count: int = 0
    error_message: Optional[str] = None
    model_version: str = "v1.0.0"
    source_extension: str = ""
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def from_dict(d: Dict) -> "ManifestRecord":
        return ManifestRecord(
            source_gcs_path=d["source_gcs_path"],
            output_gcs_path=d["output_gcs_path"],
            status=d["status"],
            processed_at=d.get("processed_at", ""),
            retry_count=d.get("retry_count", 0),
            error_message=d.get("error_message"),
            model_version=d.get("model_version", ""),
            source_extension=d.get("source_extension", ""),
            duration_sec=d.get("duration_sec"),
            width=d.get("width"),
            height=d.get("height"),
            fps=d.get("fps"),
        )


# ── In-memory index ────────────────────────────────────────────────────────


class ManifestIndex:
    """
    In-process view of the manifest.

    Loaded once at batch start, updated in memory, flushed to GCS after
    each video completes (success or failure).
    """

    def __init__(self, records: Optional[List[ManifestRecord]] = None) -> None:
        # dict keyed by source_gcs_path; last write wins
        self._records: Dict[str, ManifestRecord] = {}
        for r in records or []:
            self._records[r.source_gcs_path] = r

    # ── Queries ────────────────────────────────────────────────────────────

    def is_already_successful(self, source_gcs_path: str) -> bool:
        """Return True if the path has a success record in the manifest."""
        rec = self._records.get(source_gcs_path)
        return rec is not None and rec.status == STATUS_SUCCESS

    def get(self, source_gcs_path: str) -> Optional[ManifestRecord]:
        return self._records.get(source_gcs_path)

    def all_records(self) -> List[ManifestRecord]:
        return list(self._records.values())

    # ── Mutations ──────────────────────────────────────────────────────────

    def upsert(self, record: ManifestRecord) -> None:
        """Insert or replace the record for *source_gcs_path*."""
        self._records[record.source_gcs_path] = record

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_jsonl(self) -> str:
        """Serialise all records as JSONL (one JSON object per line)."""
        lines = [r.to_json() for r in self._records.values()]
        return "\n".join(lines) + ("\n" if lines else "")


# ── GCS-backed manifest ─────────────────────────────────────────────────────


class GCSManifest:
    """
    High-level manifest object that reads from and writes to GCS.

    Uses the GCS I/O adapter so tests can inject a fake client.
    """

    def __init__(
        self,
        gcs_client,  # google.cloud.storage.Client or fake
        bucket_name: str,
        blob_name: str,
        model_version: str = "v1.0.0",
    ) -> None:
        self._client = gcs_client
        self._bucket = bucket_name
        self._blob = blob_name
        self._model_version = model_version
        self._index: ManifestIndex = ManifestIndex()

    def load(self) -> None:
        """Load the manifest from GCS into memory.  Safe to call repeatedly."""
        from src import gcs_io

        lines = gcs_io.read_jsonl_blob(self._client, self._bucket, self._blob)
        records = []
        for i, line in enumerate(lines):
            try:
                records.append(ManifestRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Skipping malformed manifest line {i}: {exc}")
        self._index = ManifestIndex(records)
        logger.info(f"Loaded {len(records)} manifest records from GCS")

    def is_already_successful(self, source_gcs_path: str) -> bool:
        return self._index.is_already_successful(source_gcs_path)

    def record_success(
        self,
        source_gcs_path: str,
        output_gcs_path: str,
        retry_count: int = 0,
        source_extension: str = "",
        duration_sec: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        record = ManifestRecord(
            source_gcs_path=source_gcs_path,
            output_gcs_path=output_gcs_path,
            status=STATUS_SUCCESS,
            retry_count=retry_count,
            model_version=self._model_version,
            source_extension=source_extension,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
        )
        self._upsert_and_flush(record)

    def record_failure(
        self,
        source_gcs_path: str,
        output_gcs_path: str,
        error_message: str,
        retry_count: int = 0,
        source_extension: str = "",
    ) -> None:
        record = ManifestRecord(
            source_gcs_path=source_gcs_path,
            output_gcs_path=output_gcs_path,
            status=STATUS_FAILED,
            error_message=error_message,
            retry_count=retry_count,
            model_version=self._model_version,
            source_extension=source_extension,
        )
        self._upsert_and_flush(record)

    def _upsert_and_flush(self, record: ManifestRecord) -> None:
        from src import gcs_io

        self._index.upsert(record)
        log_manifest_update(logger, record.source_gcs_path, record.status)
        content = self._index.to_jsonl()
        gcs_io.upload_text_as_blob(self._client, self._bucket, self._blob, content)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._index.all_records():
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts
