"""Scale calibration with ArUco markers.

Four 4x4 ArUco markers (ids 0-3) of known side length are placed at the corners of the tray.
Their size in pixels gives mm-per-pixel; their pixels are excluded from grain segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)
DEFAULT_MARKER_MM = 20.0

_dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
_detector = cv2.aruco.ArucoDetector(_dictionary, cv2.aruco.DetectorParameters())


@dataclass
class Calibration:
    px_per_mm: float
    source: str                      # "markers" or "manual"
    markers: int = 0
    exclude_mask: np.ndarray | None = field(default=None, repr=False)
    corners: list = field(default_factory=list, repr=False)

    def info(self) -> dict:
        return {"px_per_mm": round(self.px_per_mm, 2), "source": self.source, "markers": self.markers}


def detect(img_bgr: np.ndarray, marker_mm: float = DEFAULT_MARKER_MM,
           fallback_px_per_mm: float = 12.0) -> Calibration:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    corners, ids, _ = _detector.detectMarkers(gray)
    h, w = gray.shape[:2]
    if ids is None or len(ids) == 0:
        return Calibration(fallback_px_per_mm, "manual", 0, None)
    sides = []
    mask = np.zeros((h, w), np.uint8)
    kept = []
    for c, i in zip(corners, ids.flatten()):
        if int(i) not in MARKER_IDS:
            continue
        pts = c.reshape(4, 2)
        sides.extend(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1).tolist())
        # Exclude the marker plus its white quiet zone (about 1/4 of the side on each edge).
        centre = pts.mean(axis=0)
        grown = centre + (pts - centre) * 1.6
        cv2.fillConvexPoly(mask, grown.astype(np.int32), 255)
        kept.append(pts.tolist())
    if not sides:
        return Calibration(fallback_px_per_mm, "manual", 0, None)
    px_per_mm = float(np.median(sides)) / marker_mm
    return Calibration(px_per_mm, "markers", len(kept), mask, kept)


def marker_image(marker_id: int, side_px: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(_dictionary, marker_id, side_px)


def calibration_sheet(dpi: int = 300, marker_mm: float = DEFAULT_MARKER_MM, black_tray: bool = True) -> np.ndarray:
    """A4 landscape sheet: 4 markers in the corners, an 18% grey patch, and (optionally)
    a matte-black sample area in the middle that can serve as the tray background."""
    mm = dpi / 25.4
    W, H = int(297 * mm), int(210 * mm)
    sheet = np.full((H, W), 255, np.uint8)
    side = int(marker_mm * mm)
    margin = int(12 * mm)
    pos = [(margin, margin), (W - margin - side, margin),
           (W - margin - side, H - margin - side), (margin, H - margin - side)]
    for mid, (x, y) in zip(MARKER_IDS, pos):
        sheet[y:y + side, x:x + side] = marker_image(mid, side)
    if black_tray:
        x0, y0 = margin + side + int(8 * mm), margin
        x1, y1 = W - margin - side - int(8 * mm), H - margin
        sheet[y0:y1, x0:x1] = 18
    # 18% grey patch (sRGB ~ 119) for white balance, under marker 0.
    gy = margin + side + int(8 * mm)
    sheet[gy:gy + side, margin:margin + side] = 119
    txt = f"FCI AI Grain Analyser calibration sheet - markers {marker_mm:.0f} mm - print at 100% scale"
    cv2.putText(sheet, txt, (margin, H - int(3 * mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * dpi / 300, 60, 2, cv2.LINE_AA)
    return sheet
