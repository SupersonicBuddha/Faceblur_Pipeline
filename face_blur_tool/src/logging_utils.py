"""
Structured logging helpers.

All modules should import ``get_logger`` from here rather than calling
``logging.getLogger`` directly, so we can enforce a consistent format.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key.startswith("_fb_"):
                payload[key[4:]] = val
        return json.dumps(payload, default=str)


_root_configured = False


def configure_root_logger(level: str = "INFO", json_output: bool = False) -> None:
    """Set up the root logger.  Safe to call multiple times."""
    global _root_configured
    root = logging.getLogger()
    if _root_configured:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)

    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    # Remove any handlers Colab / IPython may have already attached
    root.handlers.clear()
    root.addHandler(handler)
    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configures the root logger on first call."""
    configure_root_logger()
    return logging.getLogger(name)


def log_video_start(logger: logging.Logger, gcs_path: str) -> None:
    logger.info("Starting video processing", extra={"_fb_gcs_path": gcs_path})


def log_video_end(
    logger: logging.Logger,
    gcs_path: str,
    status: str,
    duration_s: Optional[float] = None,
) -> None:
    logger.info(
        f"Finished video processing status={status}",
        extra={
            "_fb_gcs_path": gcs_path,
            "_fb_status": status,
            "_fb_duration_s": duration_s,
        },
    )


def log_upload(logger: logging.Logger, local_path: str, gcs_path: str) -> None:
    logger.info(
        f"Uploading {local_path} → {gcs_path}",
        extra={"_fb_local": local_path, "_fb_gcs": gcs_path},
    )


def log_manifest_update(
    logger: logging.Logger, gcs_path: str, status: str
) -> None:
    logger.info(
        f"Manifest update gcs_path={gcs_path} status={status}",
        extra={"_fb_gcs_path": gcs_path, "_fb_status": status},
    )
