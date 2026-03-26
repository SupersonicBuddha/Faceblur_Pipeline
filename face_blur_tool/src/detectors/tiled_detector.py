"""
Tiled detection wrapper.

Splits each frame into overlapping tiles, runs the base detector on each tile,
maps detections back to full-frame coordinates, and deduplicates with NMS.

This improves recall for small/background faces that would otherwise be too
small to detect at the full-frame detection resolution.  For example, a face
that is 20 px tall in a 1080p frame occupies only ~12 px after the frame is
downscaled to 640×640 — likely below the model's detection limit.  With
640×640 tiles and 25% overlap the same face occupies ~37 px in its tile,
well within detection range.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.detectors.base import Detection, FaceDetector


# ── NMS helpers ────────────────────────────────────────────────────────────


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """Greedy NMS: keep highest-confidence detection, suppress overlapping ones."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    suppressed = set()
    for i, det in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(det)
        for j in range(i + 1, len(detections)):
            if j not in suppressed and _iou(det.bbox, detections[j].bbox) > iou_threshold:
                suppressed.add(j)
    return kept


# ── Tiled detector ─────────────────────────────────────────────────────────


class TiledDetector(FaceDetector):
    """
    Wraps any FaceDetector and runs it on overlapping tiles of each frame.

    Parameters
    ----------
    base_detector:
        The underlying detector to run on each tile.
    tile_size:
        Width and height of each square tile in pixels.
    overlap:
        Fractional overlap between adjacent tiles [0, 1).
        0.25 means adjacent tiles share 25% of their width/height.
    nms_iou_threshold:
        IoU threshold for suppressing duplicate detections that span
        tile boundaries.
    """

    def __init__(
        self,
        base_detector: FaceDetector,
        tile_size: int = 640,
        overlap: float = 0.25,
        nms_iou_threshold: float = 0.4,
    ) -> None:
        self._base = base_detector
        self._tile_size = tile_size
        self._overlap = overlap
        self._nms_iou = nms_iou_threshold

    def warmup(self) -> None:
        self._base.warmup()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        H, W = frame.shape[:2]
        step = max(1, int(self._tile_size * (1.0 - self._overlap)))
        all_detections: List[Detection] = []

        y0 = 0
        while True:
            y1 = min(y0 + self._tile_size, H)
            y0c = max(0, y1 - self._tile_size)  # shift back if near bottom edge

            x0 = 0
            while True:
                x1 = min(x0 + self._tile_size, W)
                x0c = max(0, x1 - self._tile_size)  # shift back if near right edge

                tile = frame[y0c:y1, x0c:x1]
                for det in self._base.detect(tile):
                    tx1, ty1, tx2, ty2 = det.bbox
                    full_bbox = (tx1 + x0c, ty1 + y0c, tx2 + x0c, ty2 + y0c)
                    landmarks = (
                        [(lx + x0c, ly + y0c) for lx, ly in det.landmarks]
                        if det.landmarks
                        else None
                    )
                    all_detections.append(
                        Detection(bbox=full_bbox, confidence=det.confidence, landmarks=landmarks)
                    )

                if x1 >= W:
                    break
                x0 += step

            if y1 >= H:
                break
            y0 += step

        return _nms(all_detections, self._nms_iou)

    def close(self) -> None:
        self._base.close()
