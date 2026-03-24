"""
MediaPipe-based face detector (fallback / alternative to RetinaFace).

MediaPipe Face Detection is fast and has good recall for front-facing
faces.  It is included as an alternative for environments where insightface
cannot be installed (e.g. ARM Macs without Rosetta, some Colab TPU runtimes).

Limitations vs RetinaFace:
- Lower recall on partial occlusion and extreme angles.
- No landmark output in the short-range model.
- Slightly lower confidence on wearable/GoPro perspectives.

We still prefer RetinaFace for production; MediaPipe is the fallback.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.detectors.base import BBox, Detection, FaceDetector, Landmark
from src.logging_utils import get_logger

logger = get_logger(__name__)

_mp = None


def _get_mediapipe():
    global _mp
    if _mp is None:
        import mediapipe as mp  # type: ignore[import]
        _mp = mp
    return _mp


class MediaPipeDetector(FaceDetector):
    """
    Face detector backed by MediaPipe Face Detection.

    Parameters
    ----------
    model_selection:
        0 = short-range model (≤2 m), 1 = full-range model (≤5 m).
        Use 1 for wearable / GoPro footage where the camera is further away.
    min_detection_confidence:
        Detections below this score are discarded.
    min_face_size:
        Minimum bbox dimension in pixels.
    """

    def __init__(
        self,
        model_selection: int = 1,
        min_detection_confidence: float = 0.35,
        min_face_size: int = 30,
    ) -> None:
        self._model_selection = model_selection
        self._conf_thresh = min_detection_confidence
        self._min_face_size = min_face_size
        self._detector = None

    def warmup(self) -> None:
        mp = _get_mediapipe()
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=self._conf_thresh,
        )
        logger.info(
            f"MediaPipeDetector loaded model_selection={self._model_selection}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._detector is None:
            self.warmup()

        mp = _get_mediapipe()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        detections: List[Detection] = []
        if not results.detections:
            return detections

        for det in results.detections:
            score = det.score[0] if det.score else 0.0
            if score < self._conf_thresh:
                continue

            bbox_mp = det.location_data.relative_bounding_box
            x1 = int(bbox_mp.xmin * w)
            y1 = int(bbox_mp.ymin * h)
            x2 = int((bbox_mp.xmin + bbox_mp.width) * w)
            y2 = int((bbox_mp.ymin + bbox_mp.height) * h)

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw < self._min_face_size or bh < self._min_face_size:
                continue

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score),
                    landmarks=None,
                )
            )
        return detections

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        self._detector = None
