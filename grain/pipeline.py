"""One call from image to result: calibrate -> segment -> measure -> classify -> summarise."""
from __future__ import annotations

import time
from dataclasses import asdict

import cv2
import numpy as np

from . import calibration, rice, sugar
from .segment import measure_colour, segment


def classes_for(commodity: str) -> dict:
    return rice.CLASSES if commodity == "rice" else sugar.CLASSES


def analyse(img: np.ndarray, commodity: str, settings, calib: calibration.Calibration | None = None,
            fallback_px_per_mm: float = 12.0) -> dict:
    """Full analysis of one still image. Returns a JSON-able dict plus private '_contours'."""
    t0 = time.perf_counter()
    if calib is None:
        calib = calibration.detect(img, fallback_px_per_mm=fallback_px_per_mm)
    ppm = calib.px_per_mm
    min_area = 0.05 if commodity == "rice" else 0.004
    objs, bright = segment(img, ppm, calib.exclude_mask, min_area_mm2=min_area)
    if commodity == "rice":
        gmed = measure_colour(img, objs, bright, chalk_margin=settings.chalk_margin)
        res = rice.classify(objs, ppm, gmed, settings)
    else:
        measure_colour(img, objs, bright, dark_delta=80, rim_tol=0.0)
        res = sugar.classify(objs, ppm, settings)
    res.update({
        "commodity": commodity, "calibration": calib.info(), "backlit": bright,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "ms": round((time.perf_counter() - t0) * 1000), "settings": asdict(settings),
        "_contours": {o.id: o.contour for o in objs},
    })
    return res


def summarise(kernels: list[dict], commodity: str, settings) -> dict:
    return rice.summarise(kernels, settings) if commodity == "rice" else sugar.summarise(kernels, settings)


def draw(img: np.ndarray, result: dict, numbers: bool = False, thickness: int = 2) -> np.ndarray:
    """Draw coloured outlines (and optionally numbers) for every object."""
    out = img.copy()
    cls = classes_for(result["commodity"])
    contours = result["_contours"]
    for k in result["kernels"]:
        c = contours.get(k["id"])
        if c is None:
            continue
        col = cls[k["cls"]]["bgr"]
        cv2.drawContours(out, [c], -1, col, thickness, cv2.LINE_AA)
        if k.get("review"):
            x, y, w, h = k["bbox"]
            cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 230, 255), 1, cv2.LINE_AA)
        if numbers:
            cv2.putText(out, str(k["id"] + 1), (int(k["cx"]) - 8, int(k["cy"]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if result["calibration"]["source"] == "markers":
        cv2.putText(out, f"{result['calibration']['px_per_mm']:.1f} px/mm", (12, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
    return out
