"""
Abstract base class for face detectors.

All detectors must conform to this interface so the rest of the pipeline
is decoupled from the specific detection library in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Detection result types ─────────────────────────────────────────────────

# Bounding box as (x1, y1, x2, y2) in pixel coordinates
BBox = Tuple[int, int, int, int]

# Landmark point as (x, y) in pixel coordinates
Landmark = Tuple[int, int]


@dataclass
class Detection:
    """A single face detection on one frame."""

    # Bounding box: top-left (x1,y1), bottom-right (x2,y2)
    bbox: BBox

    # Detector confidence score [0.0, 1.0]
    confidence: float

    # Optional 5-point landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    landmarks: Optional[List[Landmark]] = field(default=None)

    @property
    def width(self) -> int:
        return self._bbox_dim(0)

    @property
    def height(self) -> int:
        return self._bbox_dim(1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def _bbox_dim(self, axis: int) -> int:
        if axis == 0:
            return max(0, self.bbox[2] - self.bbox[0])
        return max(0, self.bbox[3] - self.bbox[1])

    def center(self) -> Tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        return cx, cy


# ── Abstract base ──────────────────────────────────────────────────────────


class FaceDetector(ABC):
    """Abstract face detector."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect faces in a single BGR frame.

        Parameters
        ----------
        frame:
            uint8 BGR numpy array of shape (H, W, 3).

        Returns
        -------
        List[Detection]
            Possibly empty list of detections for this frame.
        """

    @abstractmethod
    def warmup(self) -> None:
        """
        Optional warmup pass (load models, allocate GPU memory, etc.).
        Called once before the first frame is processed.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the detector."""

    def __enter__(self) -> "FaceDetector":
        self.warmup()
        return self

    def __exit__(self, *_) -> None:
        self.close()
