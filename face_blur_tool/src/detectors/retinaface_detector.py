"""
RetinaFace-based face detector.

We use the ``insightface`` package which bundles a performant RetinaFace
implementation (``buffalo_sc`` or ``buffalo_l`` models).  ``insightface``
runs well on CPU for offline batch use, and scales to GPU with zero code
changes when a CUDA device is available.

Design rationale
----------------
RetinaFace was chosen over MTCNN / Haar cascades because:
  * Higher recall on medium/large faces in wearable/GoPro footage.
  * Built-in 5-point landmark output (useful for downstream orientation).
  * Handles partial occlusion and tilted faces better than sliding-window methods.
  * Well-maintained via insightface (Apache-2.0 licensed).
  * Runs on CPU without special hardware.

The ``buffalo_sc`` model (small, CPU-friendly) is the default.
Switch to ``buffalo_l`` for higher recall at the cost of speed.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.detectors.base import BBox, Detection, FaceDetector, Landmark
from src.logging_utils import get_logger

logger = get_logger(__name__)

# insightface is imported lazily so that the module can be imported
# in environments where insightface is not installed (e.g. CI test runner
# that stubs the detector).
_insightface = None


def _get_insightface():
    global _insightface
    if _insightface is None:
        import insightface  # type: ignore[import]
        _insightface = insightface
    return _insightface


class RetinaFaceDetector(FaceDetector):
    """
    Face detector backed by insightface RetinaFace.

    Parameters
    ----------
    model_name:
        insightface model pack name.  ``buffalo_sc`` is CPU-optimised;
        ``buffalo_l`` gives higher recall.
    confidence_threshold:
        Detections below this score are discarded.
    min_face_size:
        Detections whose bbox dimension is smaller than this (pixels) are
        discarded (filters tiny background faces that should not be blurred).
    providers:
        ONNX Runtime execution providers, e.g. ``['CUDAExecutionProvider',
        'CPUExecutionProvider']``.  Defaults to CPU-only.
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        confidence_threshold: float = 0.35,
        min_face_size: int = 30,
        providers: Optional[List[str]] = None,
    ) -> None:
        self._model_name = model_name
        self._conf_thresh = confidence_threshold
        self._min_face_size = min_face_size
        self._providers = providers or ["CPUExecutionProvider"]
        self._app = None

    def warmup(self) -> None:
        insightface = _get_insightface()
        self._app = insightface.app.FaceAnalysis(
            name=self._model_name,
            providers=self._providers,
        )
        # det_size controls the internal resolution; 640×640 is a good trade-off
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info(
            f"RetinaFaceDetector loaded model={self._model_name} "
            f"providers={self._providers}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._app is None:
            self.warmup()

        faces = self._app.get(frame)
        detections: List[Detection] = []
        for face in faces:
            score = float(face.det_score)
            if score < self._conf_thresh:
                continue

            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w < self._min_face_size or h < self._min_face_size:
                continue

            landmarks: Optional[List[Landmark]] = None
            if face.kps is not None:
                landmarks = [(int(p[0]), int(p[1])) for p in face.kps]

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    landmarks=landmarks,
                )
            )
        return detections

    def close(self) -> None:
        self._app = None
