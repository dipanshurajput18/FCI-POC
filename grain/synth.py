"""Synthetic tray scenes for development, demos and tests (no camera needed).

Renders rice kernels or sugar crystals on a dark tray with four ArUco markers, with known
ground truth per object. `SimScene` also animates a "shake" (tray re-spread) so the live
stability gate and multi-shot aggregation can be exercised end to end.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .calibration import marker_image

W, H = 1600, 1100
TRAY = (22, 23, 26)

RICE_PRESETS = {
    "faq":     {"name": "FAQ raw rice (good lot)",   "mix": {"sound": 88, "broken": 8, "small_broken": 0.5, "chalky": 2, "red": 1, "discoloured": 0.5}},
    "broken":  {"name": "Broken-rich lot",           "mix": {"sound": 52, "broken": 38, "small_broken": 6, "chalky": 2, "red": 1, "discoloured": 1}},
    "chalky":  {"name": "Chalky lot",                "mix": {"sound": 76, "broken": 9, "chalky": 13, "red": 1, "discoloured": 1}},
    "red":     {"name": "Red-grain mix",             "mix": {"sound": 80, "broken": 8, "red": 10, "chalky": 1, "discoloured": 1}},
    "mixed":   {"name": "Below-FAQ mix (several defects)", "mix": {"sound": 55, "broken": 17, "small_broken": 2, "chalky": 7, "red": 5, "discoloured": 6, "damaged": 5, "foreign": 3}},
    "foreign": {"name": "Stones and husk",           "mix": {"sound": 82, "broken": 9, "foreign": 6, "damaged": 3}},
}

SUGAR_PRESETS = {
    "L-31":   {"name": "L-31 (large, whitest)",        "median_mm": 2.45, "sigma": 0.16, "white": 245, "yellow": 0, "specks": 0},
    "M-30":   {"name": "M-30 (medium)",                 "median_mm": 1.62, "sigma": 0.17, "white": 236, "yellow": 3, "specks": 0},
    "S-30":   {"name": "S-30 (small)",                  "median_mm": 0.92, "sigma": 0.17, "white": 236, "yellow": 3, "specks": 0},
    "M-29":   {"name": "M-29 (medium, off-white)",      "median_mm": 1.62, "sigma": 0.17, "white": 222, "yellow": 12, "specks": 0},
    "specks": {"name": "M-30 with black specks",        "median_mm": 1.62, "sigma": 0.17, "white": 236, "yellow": 3, "specks": 14},
    "mixed":  {"name": "Mixed sizes (off-grade)",       "median_mm": 0.95, "sigma": 0.45, "white": 232, "yellow": 4, "specks": 2},
}


def _superellipse(L: float, B: float, n: float = 2.6, k: int = 40) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = (L / 2) * np.sign(c) * np.abs(c) ** (2 / n)
    y = (B / 2) * np.sign(s) * np.abs(s) ** (2 / n)
    return np.stack([x, y], 1)


def _place(pts: np.ndarray, cx: float, cy: float, ang: float) -> np.ndarray:
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return (pts @ R.T + [cx, cy]).astype(np.int32)


class SimScene:
    def __init__(self, commodity: str = "rice", preset: str = "faq", seed: int = 7):
        self.commodity = commodity
        self.preset = preset
        self.ppm = 12.0 if commodity == "rice" else 24.0
        self.marker_mm = 20.0 if commodity == "rice" else 8.0
        self.rng = np.random.default_rng(seed)
        self._noise = [self.rng.normal(0, 2.2, (H, W, 1)).astype(np.float32) for _ in range(4)]
        self._ni = 0
        self.objects = self._make_objects()
        self.shake_t = -10.0
        self._layout()

    # ---------- objects ----------
    def _make_objects(self) -> list[dict]:
        r = self.rng
        objs = []
        if self.commodity == "rice":
            mix = RICE_PRESETS[self.preset]["mix"]
            N = 190
            tot = sum(mix.values())
            for kind, v in mix.items():
                for _ in range(int(round(N * v / tot))):
                    L0, B0 = 6.6 * (1 + 0.04 * r.normal()), 2.0 * (1 + 0.04 * r.normal())
                    sub = None
                    if kind == "broken":
                        L0 *= r.uniform(0.36, 0.68)
                    elif kind == "small_broken":
                        L0 *= r.uniform(0.13, 0.20)
                        B0 *= 0.65
                    elif kind == "foreign":
                        sub = "stone" if r.random() < 0.55 else "husk"
                        if sub == "stone":
                            L0, B0 = r.uniform(1.8, 3.0), r.uniform(1.4, 2.4)
                        else:
                            L0, B0 = r.uniform(4.5, 6.5), r.uniform(1.2, 1.7)
                    objs.append({"kind": kind, "sub": sub, "L": L0 * self.ppm, "B": B0 * self.ppm,
                                 "seed": int(r.integers(1 << 30))})
        else:
            p = SUGAR_PRESETS[self.preset]
            N = 330
            for _ in range(N):
                size = float(np.exp(np.log(p["median_mm"]) + p["sigma"] * r.normal()))
                size = max(0.2, size)
                objs.append({"kind": "crystal", "sub": None, "B": size * self.ppm,
                             "L": size * r.uniform(1.0, 1.5) * self.ppm, "seed": int(r.integers(1 << 30))})
            for o in objs[:p["specks"]]:      # dark inclusion (bagasse / carbon) in some crystals
                o["kind"] = "speck"
        r.shuffle(objs)
        return objs

    def _layout(self) -> None:
        """Random non-overlapping placement; ~10% of objects are allowed to touch a neighbour."""
        r = self.rng
        occ = np.zeros((H, W), np.uint8)
        q = int(self.marker_mm * self.ppm * 1.5) + 20
        for (x, y) in [(0, 0), (W - q, 0), (0, H - q), (W - q, H - q)]:
            occ[y:y + q, x:x + q] = 255
        gap = 5 if self.commodity == "rice" else 3
        for o in self.objects:
            touch = r.random() < 0.10
            for _ in range(200):
                cx, cy = r.uniform(30, W - 30), r.uniform(30, H - 30)
                ang = r.uniform(0, np.pi)
                poly = _place(self._shape(o), cx, cy, ang)
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [poly], 255)
                if not touch:
                    m = cv2.dilate(m, np.ones((gap * 2 + 1, gap * 2 + 1), np.uint8))
                x, y, w, h = cv2.boundingRect(poly)
                x0, y0 = max(0, x - gap), max(0, y - gap)
                if np.any(occ[y0:y + h + gap, x0:x + w + gap] & m[y0:y + h + gap, x0:x + w + gap]):
                    continue
                o["cx"], o["cy"], o["ang"] = cx, cy, ang
                cv2.fillPoly(occ, [poly], 255)
                break
            else:
                o["cx"] = None
        self.base = self._render()

    def _shape(self, o: dict) -> np.ndarray:
        rr = np.random.default_rng(o["seed"])
        if self.commodity == "rice":
            if o["sub"] == "stone":
                k = 9
                t = np.sort(rr.uniform(0, 2 * np.pi, k))
                rad = rr.uniform(0.75, 1.0, k)
                return np.stack([np.cos(t) * rad * o["L"] / 2, np.sin(t) * rad * o["B"] / 2], 1)
            n = 2.6 if o["kind"] not in ("broken", "small_broken") else 3.4
            return _superellipse(o["L"], o["B"], n)
        k = int(rr.integers(4, 7))
        t = np.sort(rr.uniform(0, 2 * np.pi, k)) if k > 4 else np.array([0.1, 1.6, 3.2, 4.7]) + rr.uniform(-.2, .2, 4)
        return np.stack([np.cos(t) * o["L"] / 2 * rr.uniform(.85, 1, k), np.sin(t) * o["B"] / 2 * rr.uniform(.85, 1, k)], 1) * 1.15

    # ---------- rendering ----------
    def _draw_obj(self, img: np.ndarray, o: dict, dx: float = 0, dy: float = 0, da: float = 0) -> None:
        if o.get("cx") is None:
            return
        cx, cy, ang = o["cx"] + dx, o["cy"] + dy, o["ang"] + da
        rr = np.random.default_rng(o["seed"])
        shape = self._shape(o)
        poly = _place(shape, cx, cy, ang)
        kind = o["kind"]
        if self.commodity == "sugar":
            p = SUGAR_PRESETS[self.preset]
            wv = p["white"] + rr.uniform(-8, 6)
            col = (wv - p["yellow"] * 2.2, wv - p["yellow"] * 0.6, wv)
            cv2.fillPoly(img, [poly], tuple(c * 0.78 for c in col), cv2.LINE_AA)
            cv2.fillPoly(img, [_place(shape * 0.72, cx, cy, ang)], col, cv2.LINE_AA)
            if kind == "speck":
                rad = max(2, int(min(o["B"], o["L"]) * rr.uniform(0.12, 0.17)))
                mo = cv2.moments(poly)
                px, py = int(mo["m10"] / mo["m00"]), int(mo["m01"] / mo["m00"])
                cv2.circle(img, (px, py), rad, (35, 38, 42), -1, cv2.LINE_AA)
            return
        if kind == "foreign":
            if o["sub"] == "stone":
                g = rr.uniform(70, 95)
                cv2.fillPoly(img, [poly], (g, g + 3, g + 7), cv2.LINE_AA)
            else:
                cv2.fillPoly(img, [poly], (38, 92, 128), cv2.LINE_AA)
            return
        base = {"red": (70, 86, 165), "discoloured": (112, 172, 208)}.get(kind, (150, 156, 160))
        inner = {"red": (84, 100, 182), "discoloured": (128, 188, 220)}.get(kind, (186, 191, 194))
        cv2.fillPoly(img, [poly], base, cv2.LINE_AA)
        cv2.fillPoly(img, [_place(shape * np.array([0.82, 0.62]), cx, cy, ang)], inner, cv2.LINE_AA)
        if kind == "chalky":
            off = rr.uniform(-0.1, 0.1) * o["L"]
            cv2.fillPoly(img, [_place(_superellipse(o["L"] * 0.66, o["B"] * 0.78, 2.2) + [off, 0], cx, cy, ang)],
                         (244, 246, 246), cv2.LINE_AA)
        if kind == "red":
            p1 = _place(np.array([[-o["L"] * 0.35, 0], [o["L"] * 0.35, 0]]), cx, cy, ang)
            cv2.line(img, tuple(p1[0]), tuple(p1[1]), (50, 60, 125), 2, cv2.LINE_AA)
        if kind == "damaged":
            for _ in range(int(rr.integers(2, 5))):
                sx, sy = rr.uniform(-0.3, 0.3) * o["L"], rr.uniform(-0.2, 0.2) * o["B"]
                pt = _place(np.array([[sx, sy]]), cx, cy, ang)[0]
                cv2.circle(img, tuple(int(v) for v in pt), int(rr.uniform(0.28, 0.42) * self.ppm), (40, 45, 52), -1, cv2.LINE_AA)

    def _background(self) -> np.ndarray:
        img = np.empty((H, W, 3), np.uint8)
        img[:] = TRAY
        side = int(self.marker_mm * self.ppm)
        qz = int(side * 0.25)
        for mid, (x, y) in enumerate([(qz, qz), (W - side - qz, qz), (W - side - qz, H - side - qz), (qz, H - side - qz)]):
            cv2.rectangle(img, (x - qz, y - qz), (x + side + qz, y + side + qz), (235, 235, 235), -1)
            img[y:y + side, x:x + side] = cv2.cvtColor(marker_image(mid, side), cv2.COLOR_GRAY2BGR)
        return img

    def _render(self, motion: float = 0.0) -> np.ndarray:
        img = self._background()
        r = self.rng
        sx, sy = (r.normal(0, 25 * motion), r.normal(0, 25 * motion)) if motion else (0, 0)
        for o in self.objects:
            if motion:
                self._draw_obj(img, o, sx + r.normal(0, 30 * motion), sy + r.normal(0, 30 * motion), r.normal(0, 0.6 * motion))
            else:
                self._draw_obj(img, o)
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        if motion:
            img = cv2.blur(img, (int(3 + 12 * motion), 1))
        return img

    # ---------- camera API ----------
    def shake(self) -> None:
        self.shake_t = time.monotonic()

    def read(self) -> np.ndarray:
        dt = time.monotonic() - self.shake_t
        if dt < 1.2:
            frame = self._render(motion=max(0.15, 1 - dt / 1.2))
            self._shaken = True
        else:
            if getattr(self, "_shaken", False):
                self._shaken = False
                self._layout()
            frame = self.base
        self._ni = (self._ni + 1) % len(self._noise)
        return np.clip(frame.astype(np.float32) + self._noise[self._ni], 0, 255).astype(np.uint8)

    def truth(self) -> list[dict]:
        out = []
        for o in self.objects:
            if o.get("cx") is None:
                continue
            poly = _place(self._shape(o), o["cx"], o["cy"], o["ang"]).astype(np.float32)
            (_, _), (w, h), _ = cv2.minAreaRect(poly)
            out.append({"kind": o["kind"], "sub": o["sub"], "cx": o["cx"], "cy": o["cy"],
                        "L_mm": max(w, h) / self.ppm, "B_mm": min(w, h) / self.ppm,
                        "area_mm2": cv2.contourArea(poly) / self.ppm ** 2})
        return out
