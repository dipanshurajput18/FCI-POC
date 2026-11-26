"""Dual-camera glass rig: one phone above the glass plate, one below it.

The bottom phone looks up through the glass, so its picture is a mirror image of the top view.
Each capture takes one photo from each phone, analyses both with the normal pipeline, maps the
bottom grains onto the top picture, pairs every grain with its other face, and fuses the two
results into one FCI verdict:

* length, broken / small broken, foreign matter and weight come from the TOP view (no glass in the way);
* damaged, discoloured and red count if EITHER face shows them;
* chalky follows a selectable rule (either face / both faces / top only) until FCI answers Q2.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from grain import calibration, pipeline, rice
from grain.spec import rice_spec
from grain.synth import SimScene, _place, _superellipse

from .camera import CvSource

LIVE_H = 540
STILL_FRAMES = 6
MOTION_THR = 2.5
MAX_SHOTS = 12
FACE_DEFECTS = ("damaged", "discoloured", "red", "chalky", "foreign")
CHALKY_RULES = ("either", "both", "top")


# ---------------------------------------------------------------- sources
class DualSimSource:
    """Simulated glass rig. The bottom view is the mirror of the top view, slightly misaligned, softened
    by the glass, and some kernels carry a defect on their underside only (invisible from the top)."""
    kind = "sim"

    def __init__(self, preset: str = "mixed", seed: int = 7):
        self.scene = SimScene("rice", preset, seed)
        self.marker_mm = self.scene.marker_mm
        self.label = f"Simulated glass rig - {preset}"
        self._last = 0.0
        self._base_id = None
        self._rng = np.random.default_rng(seed + 1)
        # small misalignment between the two phones: rotation (deg), scale, shift (px)
        self._warp = (1.8, 0.96, (22.0, -14.0))

    def _mark_undersides(self) -> None:
        """Pick kernels whose defect is only on the face that rests on the glass."""
        r = self._rng
        for o in self.scene.objects:
            o.pop("under", None)
            if o.get("cx") is None or o["kind"] not in ("sound", "broken", "chalky"):
                continue
            u = r.random()
            if u < 0.07:
                o["under"] = "damaged"
            elif u < 0.11:
                o["under"] = "discoloured"

    def _draw_underside(self, img: np.ndarray) -> None:
        sc = self.scene
        for o in sc.objects:
            kind = o.get("under")
            if not kind:
                continue
            rr = np.random.default_rng(o["seed"] + 99)
            shape = sc._shape(o)
            if kind == "discoloured":
                cv2.fillPoly(img, [_place(shape, o["cx"], o["cy"], o["ang"])], (112, 172, 208), cv2.LINE_AA)
                cv2.fillPoly(img, [_place(shape * np.array([0.82, 0.62]), o["cx"], o["cy"], o["ang"])],
                             (128, 188, 220), cv2.LINE_AA)
            else:
                for _ in range(int(rr.integers(2, 5))):
                    sx, sy = rr.uniform(-0.3, 0.3) * o["L"], rr.uniform(-0.2, 0.2) * o["B"]
                    pt = _place(np.array([[sx, sy]]), o["cx"], o["cy"], o["ang"])[0]
                    cv2.circle(img, tuple(int(v) for v in pt), int(rr.uniform(0.28, 0.42) * sc.ppm),
                               (40, 45, 52), -1, cv2.LINE_AA)

    def _bottom(self, top: np.ndarray, moving: bool) -> np.ndarray:
        img = top.copy()
        if not moving:
            self._draw_underside(img)
        img = cv2.flip(img, 1)
        # The markers are printed twice and glued back to back, so from below they read normally:
        # undo the mirror inside each marker square.
        sc = self.scene
        side = int(sc.marker_mm * sc.ppm)
        qz = int(side * 0.25)
        Hh, Ww = img.shape[:2]
        for (x, y) in [(qz, qz), (Ww - side - qz, qz), (Ww - side - qz, Hh - side - qz), (qz, Hh - side - qz)]:
            img[y - qz:y + side + qz, x - qz:x + side + qz] = cv2.flip(img[y - qz:y + side + qz, x - qz:x + side + qz], 1)
        ang, s, (tx, ty) = self._warp
        M = cv2.getRotationMatrix2D((Ww / 2, Hh / 2), ang, s)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, M, (Ww, Hh), borderValue=(20, 21, 24))
        # glass: a little softer, lower contrast, faint green cast
        img = cv2.GaussianBlur(img, (3, 3), 0.9).astype(np.float32)
        img = img * 0.93 + 6
        img[..., 1] *= 1.02
        return np.clip(img, 0, 255).astype(np.uint8)

    def read_pair(self) -> tuple[np.ndarray, np.ndarray]:
        dt = time.monotonic() - self._last
        if dt < 1 / 10:
            time.sleep(1 / 10 - dt)
        self._last = time.monotonic()
        top = self.scene.read()
        if id(self.scene.base) != self._base_id:
            self._base_id = id(self.scene.base)
            self._mark_undersides()
        moving = time.monotonic() - self.scene.shake_t < 1.3
        return top, self._bottom(top, moving)

    def still_pair(self):
        return None

    def shake(self) -> None:
        self.scene.shake()

    def hidden_truth(self) -> int:
        return sum(1 for o in self.scene.objects if o.get("under") and o.get("cx") is not None)

    def close(self) -> None:
        pass


class PhonePairSource:
    """Two phone cameras (e.g. Android IP Webcam), top and bottom."""
    kind = "camera"

    def __init__(self, top_url: str, bottom_url: str, marker_mm: float = 20.0):
        self.marker_mm = marker_mm
        self.top = CvSource(top_url, marker_mm)
        try:
            self.bottom = CvSource(bottom_url, marker_mm)
        except Exception:
            self.top.close()
            raise
        self.label = f"Top {top_url}  ·  Bottom {bottom_url}"

    def read_pair(self):
        return self.top.read(), self.bottom.read()

    def still_pair(self):
        with ThreadPoolExecutor(2) as ex:
            a, b = ex.submit(self.top.still), ex.submit(self.bottom.still)
            return a.result(), b.result()

    def shake(self) -> None:
        pass

    def close(self) -> None:
        for s in (self.top, self.bottom):
            try:
                s.close()
            except Exception:
                pass


# ---------------------------------------------------------------- geometry
def marker_centres(img: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = calibration._detector.detectMarkers(gray)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in calibration.MARKER_IDS:
                out[int(i)] = c.reshape(4, 2).mean(axis=0)
    return out


def bottom_calibration(raw: np.ndarray, flip: bool, marker_mm: float, fallback: float):
    """Detect markers on the raw bottom picture (they read normally from below), then express
    everything in the flipped frame that matches the top view."""
    cal = calibration.detect(raw, marker_mm=marker_mm, fallback_px_per_mm=fallback)
    cen = marker_centres(raw)
    if flip:
        w = raw.shape[1]
        cen = {i: np.array([w - 1 - p[0], p[1]]) for i, p in cen.items()}
        if cal.exclude_mask is not None:
            cal.exclude_mask = cv2.flip(cal.exclude_mask, 1)
    return cal, cen


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    if M.shape == (3, 3):
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def align(bot_pts: np.ndarray, top_pts: np.ndarray, cen_b: dict, cen_t: dict,
          bot_size: tuple[int, int], top_size: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Transform from the (flipped) bottom picture to the top picture."""
    common = sorted(set(cen_b) & set(cen_t))
    if len(common) >= 4:
        H, _ = cv2.findHomography(np.array([cen_b[i] for i in common], np.float32),
                                  np.array([cen_t[i] for i in common], np.float32))
        if H is not None:
            return H, f"markers ({len(common)} shared)"
    if len(common) >= 2:
        M, _ = cv2.estimateAffinePartial2D(np.array([cen_b[i] for i in common], np.float32),
                                           np.array([cen_t[i] for i in common], np.float32))
        if M is not None:
            return M, f"markers ({len(common)} shared)"
    # No shared markers: start from "same framing" and refine on the grain positions (ICP).
    s = top_size[0] / max(1, bot_size[0])
    M = np.array([[s, 0, 0], [0, s, 0]], np.float64)
    if len(bot_pts) < 6 or len(top_pts) < 6:
        return M, "assumed aligned (too few grains)"
    thr = 0.08 * top_size[0]
    for _ in range(25):
        p = _apply(M, bot_pts)
        d = np.linalg.norm(p[:, None, :] - top_pts[None, :, :], axis=2)
        j = d.argmin(1)
        keep = d[np.arange(len(p)), j] < thr
        if keep.sum() < 6:
            break
        M2, _ = cv2.estimateAffinePartial2D(bot_pts[keep].astype(np.float32), top_pts[j[keep]].astype(np.float32),
                                            method=cv2.RANSAC, ransacReprojThreshold=thr / 3)
        if M2 is None:
            break
        M = M2
        thr = max(0.012 * top_size[0], thr * 0.7)
    return M, "grain positions (no shared markers)"


def pair(bp: np.ndarray, tp: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one nearest pairing within tol pixels."""
    if len(bp) == 0 or len(tp) == 0:
        return []
    d = np.linalg.norm(bp[:, None, :] - tp[None, :, :], axis=2)
    order = np.argsort(d, axis=None)
    used_b, used_t, out = set(), set(), []
    for flat in order:
        i, j = divmod(int(flat), d.shape[1])
        if d[i, j] > tol:
            break
        if i in used_b or j in used_t:
            continue
        used_b.add(i)
        used_t.add(j)
        out.append((i, j))
    return out


# ---------------------------------------------------------------- fusion
def _flags(k: dict) -> set[str]:
    return {c for c in [k["cls"], *k.get("also", [])] if c in FACE_DEFECTS}


def _brief(k: dict) -> dict:
    return {"cls": k["cls"], "also": k.get("also", []), "why": k["why"], "model": k.get("model"),
            "rules_cls": k.get("rules_cls"), "bbox": k["bbox"], "cx": k["cx"], "cy": k["cy"],
            "opaque_pct": k.get("opaque_pct"), "red_pct": k.get("red_pct"), "dark_pct": k.get("dark_pct"),
            "L": k.get("L"), "a": k.get("a"), "b": k.get("b")}


def fuse(top: dict, bot: dict, chalky_rule: str) -> dict:
    prec = rice_spec()["precedence"]
    ft, fb = _flags(top), _flags(bot)
    found = {c for c in [top["cls"], *top.get("also", [])] if c in ("broken", "small_broken")}
    if "foreign" in ft:
        found.add("foreign")
    found |= (ft | fb) & {"damaged", "discoloured", "red"}
    if chalky_rule == "either" and ("chalky" in ft or "chalky" in fb):
        found.add("chalky")
    elif chalky_rule == "both" and "chalky" in ft and "chalky" in fb:
        found.add("chalky")
    elif chalky_rule == "top" and "chalky" in ft:
        found.add("chalky")
    if "red" in found:
        found.discard("chalky")
    cls = next((c for c in prec if c in found), "sound")
    labels = rice.CLASSES
    why = [f"Top: {w}" for w in top["why"]] + [f"Bottom: {w}" for w in bot["why"]]
    hidden = sorted((fb - ft) & found - {"foreign"})
    if hidden:
        why.insert(0, "Underside shows " + ", ".join(labels[h]["label"].lower() for h in hidden)
                   + " - not visible from the top")
    out = {k: v for k, v in top.items()}
    out.update({"cls": cls, "also": [c for c in found if c != cls], "why": why,
                "review": sorted(set(top.get("review", [])) | set(bot.get("review", []))),
                "side": "both", "hidden": hidden, "top": _brief(top), "bottom": _brief(bot),
                "bbox_t": top["bbox"], "bbox_b": bot["bbox"]})
    return out


def _single(k: dict, side: str) -> dict:
    out = {kk: v for kk, v in k.items()}
    out.update({"side": side, "hidden": [], side: _brief(k),
                "bbox_t": k["bbox"] if side == "top" else None, "bbox_b": k["bbox"] if side == "bottom" else None,
                "review": [*k.get("review", []), f"seen from the {side} only - not matched to the other camera"]})
    return out


def _live_preview(img: np.ndarray, settings, marker_mm: float, fallback: float, title: str,
                  detect_on: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    sc = LIVE_H / img.shape[0]
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    cal = calibration.detect(small if detect_on is None else cv2.resize(detect_on, None, fx=sc, fy=sc),
                             marker_mm=marker_mm, fallback_px_per_mm=fallback * sc)
    if detect_on is not None and cal.exclude_mask is not None:
        cal.exclude_mask = cv2.flip(cal.exclude_mask, 1)
    res = pipeline.analyse(small, "rice", settings, cal, use_model=False)
    out = pipeline.draw(small, res, thickness=2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(out, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    n = sum(1 for k in res["kernels"] if k["cls"] != "edge")
    return out, {"n": n, "calibration": {**cal.info(), "px_per_mm": round(cal.px_per_mm / sc, 2)},
                 "resolution": [int(img.shape[1]), int(img.shape[0])]}


# ---------------------------------------------------------------- engine
class DualEngine:
    """Shares rice settings and lot details with the single-camera engine."""

    def __init__(self, main):
        self.main = main
        self.lock = threading.RLock()
        self.source = DualSimSource("mixed")
        self.preset = "mixed"
        self.flip_bottom = True
        self.chalky_rule = "either"
        self.auto_capture = True
        self.use_still = True
        self.fallback_ppm = 12.0
        self.shots: list[dict] = []
        self.summary = None
        self.stats = None
        self.sample_id = self._new_id()
        self.result_ver = 0
        self.live_jpeg = None
        self.live_ver = 0
        self.live = {"state": "starting"}
        self.error = None
        self.clients = 0
        self._capture_now = False
        self._armed = True
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _new_id() -> str:
        return "AGD-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4].upper()

    @property
    def settings(self) -> rice.RiceSettings:
        return self.main.rice

    # ---- control
    def set_source(self, src, preset: str | None = None) -> None:
        with self.lock:
            old, self.source = self.source, src
            if preset:
                self.preset = preset
            self._reset()
        try:
            old.close()
        except Exception:
            pass

    def _reset(self) -> None:
        self.shots, self.summary, self.stats = [], None, None
        self.sample_id = self._new_id()
        self._armed = True
        self.result_ver += 1

    def new_sample(self) -> None:
        with self.lock:
            self._reset()

    def capture(self) -> None:
        self._capture_now = True

    def update(self, d: dict) -> None:
        with self.lock:
            if d.get("chalky_rule") in CHALKY_RULES and d["chalky_rule"] != self.chalky_rule:
                self.chalky_rule = d["chalky_rule"]
                for s in self.shots:
                    s["kernels"] = self._fuse_shot(s["_top"], s["_bot"], s["_pairs"])
                    for k in s["kernels"]:
                        k["shot"] = s["n"]
            if "flip_bottom" in d:
                self.flip_bottom = bool(d["flip_bottom"])
            if "auto_capture" in d:
                self.auto_capture = bool(d["auto_capture"])
            if "use_still" in d:
                self.use_still = bool(d["use_still"])
            self.resummarise()

    def resummarise(self) -> None:
        with self.lock:
            if not self.shots:
                self.summary, self.stats = None, None
            else:
                ks = [k for s in self.shots for k in s["kernels"]]
                self.summary = pipeline.summarise(ks, "rice", self.settings)
                valid = [k for k in ks if k["cls"] != "edge"]
                hidden: dict[str, int] = {}
                for k in valid:
                    for h in k["hidden"]:
                        hidden[h] = hidden.get(h, 0) + 1
                self.stats = {"paired": sum(k["side"] == "both" for k in valid),
                              "top_only": sum(k["side"] == "top" for k in valid),
                              "bottom_only": sum(k["side"] == "bottom" for k in valid),
                              "hidden_kernels": sum(bool(k["hidden"]) for k in valid),
                              "hidden": hidden, "changed": sum(1 for k in valid if k["side"] == "both"
                                                                and k["cls"] != k["top"]["cls"])}
            self.result_ver += 1

    # ---- analysis
    def _fuse_shot(self, top_res: dict, bot_res: dict, pairs: list) -> list[dict]:
        tk = [k for k in top_res["kernels"]]
        bk = [k for k in bot_res["kernels"]]
        out, used_t, used_b = [], set(), set()
        for i, j in pairs:
            out.append(fuse(tk[j], bk[i], self.chalky_rule))
            used_t.add(j)
            used_b.add(i)
        for j, k in enumerate(tk):
            if j not in used_t:
                out.append(_single(k, "top") if k["cls"] != "edge" else {**k, "side": "top", "hidden": [],
                                                                         "bbox_t": k["bbox"], "bbox_b": None})
        for i, k in enumerate(bk):
            if i not in used_b and k["cls"] != "edge" and k.get("_inside", True):
                out.append(_single(k, "bottom"))
        for n, k in enumerate(out):
            k["fid"] = n
        return out

    def _analyse_pair(self, top: np.ndarray, bot_raw: np.ndarray) -> None:
        s = self.settings
        mm = self.source.marker_mm
        real = self.source.kind != "sim"
        bot = cv2.flip(bot_raw, 1) if self.flip_bottom else bot_raw
        cal_t = calibration.detect(top, marker_mm=mm, fallback_px_per_mm=self.fallback_ppm)
        cen_t = marker_centres(top)
        cal_b, cen_b = bottom_calibration(bot_raw, self.flip_bottom, mm, self.fallback_ppm)
        with ThreadPoolExecutor(2) as ex:
            ft = ex.submit(pipeline.analyse, top, "rice", s, cal_t, self.fallback_ppm, None, real)
            fb = ex.submit(pipeline.analyse, bot, "rice", s, cal_b, self.fallback_ppm, None, real)
            rt, rb = ft.result(), fb.result()
        tv = [k for k in rt["kernels"]]
        bv = [k for k in rb["kernels"]]
        tp = np.array([[k["cx"], k["cy"]] for k in tv], np.float64).reshape(-1, 2)
        bp = np.array([[k["cx"], k["cy"]] for k in bv], np.float64).reshape(-1, 2)
        ok_t = np.array([k["cls"] != "edge" for k in tv], bool)
        ok_b = np.array([k["cls"] != "edge" for k in bv], bool)
        M, how = align(bp[ok_b], tp[ok_t], cen_b, cen_t, (bot.shape[1], bot.shape[0]), (top.shape[1], top.shape[0]))
        bpt = _apply(M, bp)
        Ht, Wt = top.shape[:2]
        for k, p in zip(bv, bpt):
            k["_inside"] = bool(8 < p[0] < Wt - 8 and 8 < p[1] < Ht - 8)
            k["cx_t"], k["cy_t"] = round(float(p[0]), 1), round(float(p[1]), 1)
        tol = 1.6 * cal_t.px_per_mm
        idx_t, idx_b = np.where(ok_t)[0], np.where(ok_b)[0]
        pr = pair(bpt[idx_b], tp[idx_t], tol)
        pairs = [(int(idx_b[i]), int(idx_t[j])) for i, j in pr]
        res_err = None
        if pairs:
            res_err = float(np.median([np.linalg.norm(bpt[i] - tp[j]) for i, j in pairs]) / cal_t.px_per_mm)
        fused = self._fuse_shot(rt, rb, pairs)

        # pictures: top with fused colours, bottom with what the bottom camera saw; hidden defects ringed.
        ann_t = pipeline.draw(top, {**rt, "kernels": [{"id": k["id"], "cls": k["cls"], "bbox": k["bbox_t"],
                                                         "review": k["review"]} for k in fused
                                                        if k.get("bbox_t") is not None]}, thickness=2)
        ann_b = pipeline.draw(bot, rb, thickness=2)
        for k in fused:
            if k["hidden"]:
                for img, bb in ((ann_t, k.get("bbox_t")), (ann_b, k.get("bbox_b"))):
                    if bb:
                        x, y, w, h = bb
                        cv2.circle(img, (x + w // 2, y + h // 2), int(max(w, h) * 0.75) + 6, (255, 255, 0), 2, cv2.LINE_AA)

        def jpg(img, q=85):
            sc = min(1.0, 1600 / img.shape[1])
            if sc < 1:
                img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()

        with self.lock:
            n = (self.shots[-1]["n"] + 1) if self.shots else 0
            for k in fused:
                k["shot"] = n
            shot = {"n": n, "kernels": fused, "_top": rt, "_bot": rb, "_pairs": pairs,
                    "jpeg_t": jpg(ann_t), "jpeg_b": jpg(ann_b), "raw_t": jpg(top, 92), "raw_b": jpg(bot, 92),
                    "size_t": rt["image_size"], "size_b": rb["image_size"],
                    "cal_t": rt["calibration"], "cal_b": rb["calibration"], "align": how,
                    "align_err_mm": round(res_err, 2) if res_err is not None else None,
                    "classifier_t": rt.get("classifier"), "classifier_b": rb.get("classifier"),
                    "ms": rt["ms"] + rb["ms"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            if len(self.shots) >= MAX_SHOTS:
                self.shots.pop(0)
            self.shots.append(shot)
            self.resummarise()

    def _loop(self) -> None:
        prev, still, fps, t_last = None, 0, 0.0, time.monotonic()
        while True:
            try:
                if self.clients <= 0 and not self._capture_now:
                    time.sleep(0.25)
                    continue
                src = self.source
                top, bot = src.read_pair()
                if top is None or bot is None:
                    self.live = {"state": "no signal", "missing": "top" if top is None else "bottom"}
                    time.sleep(0.2)
                    continue
                th = cv2.cvtColor(cv2.resize(top, (160, int(160 * top.shape[0] / top.shape[1]))), cv2.COLOR_BGR2GRAY)
                th = cv2.GaussianBlur(th, (5, 5), 0).astype(np.int16)
                motion = float(np.mean(np.abs(th - prev))) if prev is not None and prev.shape == th.shape else 0.0
                prev = th
                if motion > MOTION_THR:
                    still, self._armed, state = 0, True, "moving"
                else:
                    still += 1
                    state = "settling" if still < STILL_FRAMES else "still"

                bshow = cv2.flip(bot, 1) if self.flip_bottom else bot
                lt, it = _live_preview(top, self.settings, src.marker_mm, self.fallback_ppm, "TOP camera")
                lb, ib = _live_preview(bshow, self.settings, src.marker_mm, self.fallback_ppm,
                                       "BOTTOM camera (through glass" + (", mirror corrected)" if self.flip_bottom else ")"),
                                       detect_on=bot if self.flip_bottom else None)
                gap = np.full((LIVE_H, 8, 3), 30, np.uint8)
                live = np.hstack([lt, gap, lb])
                ok, jpg = cv2.imencode(".jpg", live, [cv2.IMWRITE_JPEG_QUALITY, 72])
                now = time.monotonic()
                fps = 0.8 * fps + 0.2 / max(1e-3, now - t_last)
                t_last = now

                do_full = self._capture_now or (self.auto_capture and self._armed and still >= STILL_FRAMES)
                if src is not self.source:
                    continue
                if do_full:
                    self._capture_now, self._armed = False, False
                    self.live = {**self.live, "state": "analysing"}
                    if self.use_still:
                        pair_hi = src.still_pair()
                        if pair_hi and pair_hi[0] is not None and pair_hi[1] is not None:
                            top, bot = pair_hi
                    self._analyse_pair(top, bot)
                    state = "analysed"
                elif state == "still" and not self._armed:
                    state = "analysed"
                with self.lock:
                    self.live_jpeg = jpg.tobytes()
                    self.live_ver += 1
                    self.live = {"state": state, "fps": round(fps, 1), "motion": round(motion, 2),
                                 "source": src.label, "source_kind": src.kind, "shots": len(self.shots),
                                 "top": it, "bottom": ib}
                self.error = None
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                time.sleep(0.5)

    # ---- views
    def result(self) -> dict:
        def pub(k):
            return {kk: v for kk, v in k.items() if not kk.startswith("_")}
        with self.lock:
            shots = [{"n": s["n"], "time": s["time"], "ms": s["ms"], "size_t": s["size_t"], "size_b": s["size_b"],
                      "cal_t": s["cal_t"], "cal_b": s["cal_b"], "align": s["align"], "align_err_mm": s["align_err_mm"],
                      "classifier_t": s["classifier_t"], "classifier_b": s["classifier_b"],
                      "kernels": [pub(k) for k in s["kernels"]]} for s in self.shots]
            return {"ver": self.result_ver, "sample_id": self.sample_id, "commodity": "rice",
                    "grade": self.settings.grade, "lot": dict(self.main.lot), "summary": self.summary,
                    "stats": self.stats, "chalky_rule": self.chalky_rule, "shots": shots,
                    "classes": {k: {"label": v["label"], "color": "#%02x%02x%02x" % v["bgr"][::-1]}
                                for k, v in rice.CLASSES.items()}}

    def config(self) -> dict:
        from grain.synth import RICE_PRESETS
        return {"source": {"kind": self.source.kind, "label": self.source.label, "preset": self.preset},
                "flip_bottom": self.flip_bottom, "chalky_rule": self.chalky_rule,
                "auto_capture": self.auto_capture, "use_still": self.use_still,
                "presets": {k: v["name"] for k, v in RICE_PRESETS.items()},
                "grades": list(rice_spec()["grades"].keys()), "rice": {"grade": self.settings.grade,
                                                                       "sample_g": self.settings.sample_g,
                                                                       "classifier": self.settings.classifier},
                "lot": self.main.lot}
