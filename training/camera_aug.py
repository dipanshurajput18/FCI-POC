"""Simulate the tray camera on a canonical training crop.

The trained model sees kernels cut out of camera frames by grain.crop.canonical_crop. v1 was
trained on clean down-sampled crops and failed on realistic trays: sensor noise and JPEG
blocking at about 12 px/mm made sound kernels look damaged or chalky (tests/realtray.py).
This function reproduces the camera chain and then cuts the kernel out with the SAME crop code
the live app uses:

  kernel on a dark tray -> down-sample to the camera resolution -> lens blur -> sensor noise
  -> JPEG -> mask from the down-sampled outline -> canonical_crop (soft edge, upscale to 128)
"""
from __future__ import annotations

import cv2
import numpy as np

from grain.crop import FILL, SIZE, canonical_crop


def camera_degrade(x: np.ndarray, length_mm: float, rng: np.random.Generator,
                   ppm: float | None = None, noise: float | None = None, jpeg: int | None = None,
                   blur: float | None = None, tray: float | None = None) -> np.ndarray:
    """x: canonical 128 px BGR crop (kernel long side = SIZE*FILL px = length_mm). Returns a 128 px crop."""
    ppm = rng.uniform(7, 24) if ppm is None else ppm
    noise = rng.uniform(0.0, 4.0) if noise is None else noise
    jpeg = (int(rng.integers(60, 96)) if rng.random() < 0.8 else 100) if jpeg is None else jpeg
    blur = rng.uniform(0.0, 0.9) if blur is None else blur
    tray = rng.uniform(0, 35) if tray is None else tray
    crop_ppm = SIZE * FILL / max(length_mm, 0.3)
    f = min(1.0, ppm / crop_ppm)
    alpha = (x.max(2) > 8).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    img = x.astype(np.float32) * alpha[..., None] + tray * (1 - alpha[..., None])
    k = max(8, int(round(SIZE * f)))
    img = cv2.resize(img, (k, k), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (k, k), interpolation=cv2.INTER_AREA)
    if blur > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if jpeg < 100:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
    mask = (a > 0.5).astype(np.uint8)
    out = canonical_crop(img, mask)
    return x if out is None else out
