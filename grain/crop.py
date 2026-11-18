"""Canonical single-kernel crop shared by training and the live pipeline.

The kernel is rotated so its long axis is vertical, masked onto black, scaled so
its long side fills `fill` of a `size` x `size` square and centred. Absolute size
is deliberately removed: length is judged by the FCI rule (grain/rice.py), the
model judges condition only (sound / chalky / damaged / discoloured).
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = 128
FILL = 0.875


def canonical_crop(img: np.ndarray, mask: np.ndarray, size: int = SIZE, fill: float = FILL) -> np.ndarray | None:
    """img: BGR image, mask: uint8 mask of ONE kernel (same shape). Returns size x size BGR."""
    pts = cv2.findNonZero((mask > 0).astype(np.uint8))
    if pts is None or len(pts) < 5:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if w > h:                        # make the long side vertical
        ang += 90.0
        w, h = h, w
    side = int(np.ceil(np.hypot(w, h))) + 4
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += side / 2 - cx
    M[1, 2] += side / 2 - cy
    rot = cv2.warpAffine(img, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rm = cv2.warpAffine((mask > 0).astype(np.uint8) * 255, M, (side, side), flags=cv2.INTER_NEAREST)
    rot[rm == 0] = 0
    ys, xs = np.where(rm > 0)
    if len(ys) == 0:
        return None
    rot = rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = size * fill / max(rot.shape[:2])
    rot = cv2.resize(rot, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - rot.shape[0]) // 2, (size - rot.shape[1]) // 2
    out[y0:y0 + rot.shape[0], x0:x0 + rot.shape[1]] = rot
    return out


def mask_from_black_bg(img: np.ndarray, thr: int = 12) -> np.ndarray | None:
    """Largest bright component on a black background (IIT Indore segmented images)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n, lab, st, _ = cv2.connectedComponentsWithStats((gray > thr).astype(np.uint8))
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)
