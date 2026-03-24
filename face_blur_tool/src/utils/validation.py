"""
Input validation helpers.
"""

from __future__ import annotations

import os
from typing import List


def validate_gcs_uri(uri: str) -> None:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError(f"Expected a GCS URI (gs://...), got: {uri!r}")


def validate_supported_extension(ext: str, supported: List[str]) -> None:
    if ext.lower() not in [e.lower() for e in supported]:
        raise ValueError(
            f"Extension {ext!r} not in supported list {supported}"
        )


def validate_local_path_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected local path does not exist: {path!r}")
