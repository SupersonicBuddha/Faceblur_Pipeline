"""
Retry decorator / helper for video processing.

The retry logic is intentionally simple: try once, on failure try once more,
then give up and log.  We do not use exponential back-off because this is
offline batch processing where transient network errors are the main failure
mode (GCS upload) and a single immediate retry is sufficient.
"""

from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

from src.logging_utils import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable)


def retry_once(fn: Callable, *args, label: str = "", **kwargs):
    """
    Call ``fn(*args, **kwargs)``; if it raises, wait 2 s and try once more.

    Parameters
    ----------
    fn:
        Callable to invoke.
    *args, **kwargs:
        Forwarded to *fn*.
    label:
        Human-readable label used in log messages (e.g. the video path).

    Returns
    -------
    (result, retry_count)
        ``result`` is the return value of *fn* on success.
        ``retry_count`` is 0 (first attempt) or 1 (after one retry).

    Raises
    ------
    Exception
        Re-raises the exception from the second attempt if both fail.
    """
    try:
        return fn(*args, **kwargs), 0
    except Exception as exc:
        logger.warning(
            f"First attempt failed for {label!r}: {exc!r}. Retrying in 2 s…"
        )
        time.sleep(2)
        try:
            return fn(*args, **kwargs), 1
        except Exception as exc2:
            logger.error(
                f"Second attempt also failed for {label!r}: {exc2!r}. Giving up."
            )
            raise
