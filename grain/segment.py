"""Find every kernel / crystal and measure it.

Classical pipeline (works on CPU, no training needed):
  Otsu threshold -> clean-up -> connected components -> split touching objects with a
  distance-transform watershed -> per-object morphometry and colour statistics.
A trained instance-segmentation model can later replace `segment()` behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Obj:
    id: int
    contour: np.ndarray = field(repr=False)
    bbox: tuple                     # x, y, w, h (px)
    centroid: tuple                 # x, y (px)
    area_px: float
    length_px: float
    breadth_px: float
    angle: float
    solidity: float
    touches_edge: bool
    # colour statistics (filled by measure_colour)
    L: float = 0.0                  # mean L* 0-100
    a: float = 0.0                  # mean a* (-128..127)
    b: float = 0.0                  # mean b*
    chroma: float = 0.0
    red_frac: float = 0.0
    bright_frac: float = 0.0
    dark_frac: float = 0.0
    split: bool = False             # came from a watershed split
    feats: dict = field(default_factory=dict)


def background_is_bright(gray: np.ndarray, exclude: np.ndarray | None = None) -> bool:
    """The tray is the most common grey level: grains cover well under half the tray.

    The whole frame is used, not just its border: when the camera sees the printed calibration
    sheet, the border is white paper while the tray (the black sample area) fills the middle.
    """
    s = max(1, min(gray.shape) // 400)               # subsample for speed
    g = gray[::s, ::s]
    if exclude is not None:
        keep = exclude[::s, ::s] == 0
        if keep.any():
            g = g[keep]
    return float(np.median(g)) > 127


def foreground_mask(img_bgr: np.ndarray, exclude: np.ndarray | None = None,
                    max_px: float | None = None) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bright = background_is_bright(gray, exclude)
    if exclude is not None and exclude.any():
        # Paint marker areas with the tray level so they do not skew the Otsu threshold.
        gray = gray.copy()
        gray[exclude > 0] = 230 if bright else int(np.percentile(gray[exclude == 0], 10))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    flag = cv2.THRESH_BINARY_INV if bright else cv2.THRESH_BINARY
    t, fg = cv2.threshold(blur, 0, 255, flag | cv2.THRESH_OTSU)
    # Translucent kernels on a dark tray are dim at the edges: lower the threshold a little.
    if not bright:
        _, fg = cv2.threshold(blur, max(8, t * 0.8), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg0 = fg.copy()                      # before large regions are dropped: used to find the tray
    if max_px:
        # Drop regions far larger than any grain (e.g. white paper framing the black sample area)
        # before hole filling, otherwise the frame's "hole" - the whole tray - would be filled.
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        big = np.where(st[1:, cv2.CC_STAT_AREA] > max_px)[0] + 1
        if len(big):
            fg[np.isin(lab, big)] = 0
    # Fill holes (dark spots inside kernels must stay part of the kernel).
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    # Keep only objects lying on the tray: the largest connected tray-coloured region with its holes
    # (the grains) filled. Drops things outside it, e.g. the insides of printed letters on the sheet.
    bg = cv2.bitwise_not(fg0)
    if exclude is not None:
        bg[exclude > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    if n > 1:
        tray = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
        tc, _ = cv2.findContours(tray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        region = np.zeros_like(tray)
        cv2.drawContours(region, tc, -1, 255, cv2.FILLED)
        filled &= region
    if exclude is not None:
        filled[exclude > 0] = 0
    return filled, bright


def _split(comp: np.ndarray, expected: int) -> list[np.ndarray]:
    """Split one binary component into `expected`-ish pieces with a distance-transform watershed."""
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    best = None
    for frac in (0.75, 0.65, 0.55, 0.45, 0.35):
        peaks = (dist > frac * dist.max()).astype(np.uint8)
        n, markers = cv2.connectedComponents(peaks)
        n -= 1
        if n < 2:
            continue
        score = abs(n - expected)
        if best is None or score < best[0]:
            best = (score, markers, n)
        if n >= expected:
            break
    if best is None:
        return [comp]
    _, markers, n = best
    markers = markers.astype(np.int32) + 1
    markers[comp == 0] = 0
    unknown = (comp > 0) & (markers == 1)
    markers[unknown] = 0
    markers[comp == 0] = 1  # background
    ws = cv2.watershed(cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR), markers)
    parts = []
    for lab in range(2, n + 2):
        m = (ws == lab).astype(np.uint8) * 255
        if cv2.countNonZero(m) > 0:
            parts.append(m)
    return parts or [comp]


EDGE_SPREAD_PX = 1.5   # blur + threshold make objects ~1.5 px wider; calibrate per camera


def _make_obj(oid: int, cnt: np.ndarray, shape: tuple, split: bool) -> Obj | None:
    area = cv2.contourArea(cnt)
    if area <= 0:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    length, breadth = max(w, h) - EDGE_SPREAD_PX, min(w, h) - EDGE_SPREAD_PX
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull) or area
    x, y, bw, bh = cv2.boundingRect(cnt)
    H, W = shape[:2]
    edge = x <= 1 or y <= 1 or x + bw >= W - 1 or y + bh >= H - 1
    return Obj(oid, cnt, (x, y, bw, bh), (float(cx), float(cy)), float(area), float(length),
               float(max(breadth, 1.0)), float(ang), float(area / ha), edge, split=split)


MAX_OBJECT_MM2 = 300.0   # larger regions are not grains (paper around the tray, grey patch, tray edge)


def segment(img_bgr: np.ndarray, px_per_mm: float, exclude: np.ndarray | None = None,
            min_area_mm2: float = 0.05, max_area_mm2: float = MAX_OBJECT_MM2) -> tuple[list[Obj], bool]:
    min_px = max(6.0, min_area_mm2 * px_per_mm * px_per_mm)
    max_px = max_area_mm2 * px_per_mm * px_per_mm
    fg, bright = foreground_mask(img_bgr, exclude, max_px)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps = [i for i in range(1, n) if min_px <= stats[i, cv2.CC_STAT_AREA] <= max_px]
    if not comps:
        return [], bright
    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in comps], float)
    # Typical single-object area: median of the upper-middle of the distribution.
    single = float(np.median(areas[areas >= np.percentile(areas, 30)])) if len(areas) > 4 else float(np.median(areas))

    objs: list[Obj] = []
    oid = 0
    for i in comps:
        x, y, w, h, a = stats[i]
        sub = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        sub = cv2.copyMakeBorder(sub, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        hull_a = cv2.contourArea(cv2.convexHull(cnt)) or 1
        solidity = cv2.contourArea(cnt) / hull_a
        expected = int(round(a / single)) if single > 0 else 1
        pieces = [sub]
        if (solidity < 0.86 and a > 1.3 * single) or expected >= 2:
            pieces = _split(sub, max(2, expected))
        for p in pieces:
            pc, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not pc:
                continue
            c = max(pc, key=cv2.contourArea)
            if cv2.contourArea(c) < min_px * 0.5:
                continue
            c = c + np.array([[x - 2, y - 2]])
            o = _make_obj(oid, c, fg.shape, len(pieces) > 1)
            if o:
                objs.append(o)
                oid += 1
    return objs, bright


def _spot_fraction(Lpatch: np.ndarray, m: np.ndarray, thr: float, rim_px: int = 2, min_px: int = 4,
                   rim_tol: float = 0.3) -> float:
    """Share of the kernel covered by compact dark blobs that do NOT touch the outline
    (shaded edges are not damage)."""
    dark = ((Lpatch < thr) & (m > 0)).astype(np.uint8)
    if not dark.any():
        return 0.0
    k = 2 * rim_px + 1
    rim = (m > 0) & (cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) == 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_px:
            continue
        if rim[lab == i].mean() > rim_tol:  # on the outline -> edge shading, not a spot
            continue
        keep += st[i, cv2.CC_STAT_AREA]
    return keep / max(1, int((m > 0).sum()))


def measure_colour(img_bgr: np.ndarray, objs: list[Obj], bright_bg: bool, chalk_margin: float = 30.0,
                   red_delta: float = 18.0, dark_delta: float = 55.0, rim_tol: float = 0.3) -> float:
    """Fill per-object colour statistics. Returns the sample's median kernel brightness (0-255)."""
    if not objs:
        return 0.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    patches = []
    allL = []
    H, W = img_bgr.shape[:2]
    for o in objs:
        # Pad the patch: erosion does not shrink a mask where it touches the patch border.
        bx, by, bw, bh = o.bbox
        x, y = max(0, bx - 4), max(0, by - 4)
        w, h = min(W, bx + bw + 4) - x, min(H, by + bh + 4) - y
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [o.contour - np.array([[x, y]])], -1, 255, cv2.FILLED)
        # Erode 1 px so the dark tray does not leak into the edge pixels.
        if w > 6 and h > 6:
            m = cv2.erode(m, np.ones((3, 3), np.uint8))
        patch = lab[y:y + h, x:x + w]
        px = patch[m > 0]
        if len(px) == 0:
            px = lab[int(o.centroid[1]), int(o.centroid[0])].reshape(1, 3)
        patches.append((px, patch[:, :, 0], m))
        allL.append(px[:, 0])
    gmed = float(np.median(np.concatenate(allL)))
    for o, (px, Lpatch, m) in zip(objs, patches):
        L8 = px[:, 0].astype(np.float32)
        a8 = px[:, 1].astype(np.float32) - 128
        b8 = px[:, 2].astype(np.float32) - 128
        o.L = float(L8.mean() * 100 / 255)
        o.a = float(a8.mean())
        o.b = float(b8.mean())
        o.chroma = float(np.hypot(o.a, o.b))
        o.red_frac = float(np.mean(a8 > red_delta))
        if bright_bg:   # backlit: opaque (chalky) parts transmit less light -> darker
            o.bright_frac = float(np.mean(L8 < gmed - chalk_margin))
        else:           # top-lit on dark tray: chalky parts scatter more light -> brighter
            o.bright_frac = float(np.mean(L8 > gmed + chalk_margin))
        # Dark spots are judged against the kernel's own brightness, so naturally darker
        # kernels (red, discoloured) are not mistaken for damaged ones.
        kmed = float(np.median(L8))
        o.dark_frac = _spot_fraction(Lpatch, m, min(kmed, gmed) - dark_delta,
                                     rim_px=max(2, int(round(0.14 * o.breadth_px))), rim_tol=rim_tol)
    return gmed
