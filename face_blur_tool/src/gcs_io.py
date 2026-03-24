"""
Thin adapter around google-cloud-storage.

All GCS interactions are routed through this module so that tests can
easily swap in a fake implementation without touching business logic.

The google-cloud-storage import is deferred to function bodies so that
this module can be imported in test environments where the package is
not installed.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Generator, Iterator, Optional

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ── Client factory ─────────────────────────────────────────────────────────


def get_gcs_client(project: Optional[str] = None):
    """Return a GCS client, optionally scoped to *project*."""
    from google.cloud import storage  # type: ignore[import]

    if project:
        return storage.Client(project=project)
    return storage.Client()


# ── Blob listing ───────────────────────────────────────────────────────────


def list_blobs(
    client,
    bucket_name: str,
    prefix: str,
) -> Iterator:
    """Yield all blobs under *prefix* inside *bucket_name*."""
    logger.debug(f"Listing gs://{bucket_name}/{prefix}")
    yield from client.list_blobs(bucket_name, prefix=prefix)


# ── Download ───────────────────────────────────────────────────────────────


def download_blob_to_file(
    client,
    bucket_name: str,
    blob_name: str,
    local_path: str,
) -> None:
    """Download a single blob to *local_path* (creates parent dirs)."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info(f"Downloading gs://{bucket_name}/{blob_name} → {local_path}")
    blob.download_to_filename(local_path)


@contextmanager
def download_blob_temp(
    client,
    bucket_name: str,
    blob_name: str,
    suffix: str = "",
) -> Generator[str, None, None]:
    """
    Context manager: download blob to a named temp file.

    Yields the local path; deletes the file on exit.
    """
    fd, local_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        download_blob_to_file(client, bucket_name, blob_name, local_path)
        yield local_path
    finally:
        try:
            os.unlink(local_path)
        except FileNotFoundError:
            pass


# ── Upload ─────────────────────────────────────────────────────────────────


def upload_file_to_blob(
    client,
    bucket_name: str,
    blob_name: str,
    local_path: str,
    content_type: Optional[str] = None,
) -> None:
    """Upload *local_path* to gs://*bucket_name*/*blob_name*."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info(f"Uploading {local_path} → gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(local_path, content_type=content_type)


# ── Text / bytes helpers ───────────────────────────────────────────────────


def download_blob_as_text(
    client,
    bucket_name: str,
    blob_name: str,
) -> str:
    """Download a blob and return its content as a UTF-8 string."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(encoding="utf-8")


def upload_text_as_blob(
    client,
    bucket_name: str,
    blob_name: str,
    text: str,
) -> None:
    """Upload a UTF-8 string as a blob."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text.encode("utf-8"), content_type="text/plain")


def blob_exists(
    client,
    bucket_name: str,
    blob_name: str,
) -> bool:
    """Return True if the blob exists."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()


def append_line_to_blob(
    client,
    bucket_name: str,
    blob_name: str,
    line: str,
) -> None:
    """
    Append *line* to a JSONL blob in GCS.

    GCS does not support true append, so we download the full content,
    append the new line, and re-upload.  This is acceptable for the
    manifest because manifest writes are infrequent and the file is small.
    """
    existing = ""
    if blob_exists(client, bucket_name, blob_name):
        existing = download_blob_as_text(client, bucket_name, blob_name)
    updated = existing.rstrip("\n") + "\n" + line + "\n"
    upload_text_as_blob(client, bucket_name, blob_name, updated)


def read_jsonl_blob(
    client,
    bucket_name: str,
    blob_name: str,
) -> list:
    """Return non-empty lines from a JSONL blob, or [] if it doesn't exist."""
    if not blob_exists(client, bucket_name, blob_name):
        return []
    content = download_blob_as_text(client, bucket_name, blob_name)
    return [ln for ln in content.splitlines() if ln.strip()]
