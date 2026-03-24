"""
Lightweight timing utilities.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timed(label: str = "") -> Generator[None, None, None]:
    """Context manager that prints elapsed time on exit."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    tag = f" [{label}]" if label else ""
    print(f"Elapsed{tag}: {elapsed:.2f}s")


class Stopwatch:
    """Simple wall-clock stopwatch."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self) -> float:
        elapsed = self.elapsed()
        self._start = time.perf_counter()
        return elapsed
