"""Live analysis engine.

Loop: read newest frame -> fast analysis on a downscaled copy (live overlay) -> motion check.
When the tray has been still for ~1 s after moving (or on manual capture), the full-resolution
frame is analysed and added as a "shot" to the current sample. Shots are aggregated so the
officer can re-spread the sample and capture both sides / more kernels.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import fields

import cv2
import numpy as np

from grain import calibration, condition_model, pipeline, rice, sugar
from grain.spec import rice_spec

from .camera import ImageSource, SimSource, Source

LIVE_MAX_W = 960
STILL_FRAMES = 6
MOTION_THR = 2.5
MAX_SHOTS = 12


def _public(k: dict) -> dict:
    return {key: v for key, v in k.items() if not key.startswith("_")}


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.commodity = "rice"
        self.rice = rice.RiceSettings()
        self.sugar = sugar.SugarSettings()
        self.lot = {"lot_id": "LOT/PB/2026/000123", "officer": "QC Officer", "depot": "Demo depot"}
        self.auto_capture = True
        self.use_still = True     # phone sources: analyse a full-resolution photo, not the video frame
        self.fallback_ppm = 12.0
        self.source: Source = SimSource("rice", "mixed")
        self.preset = "mixed"
        self.shots: list[dict] = []
        self.summary: dict | None = None
        self.sample_id = self._new_id()
        self.result_ver = 0
        self.live_jpeg: bytes | None = None
        self.live_ver = 0
        self.live = {"state": "starting", "fps": 0.0, "counts": {}, "n": 0, "motion": 0.0, "calibration": None}
        self.error: str | None = None
        self.clients = 0          # open pages; the loop idles when nobody is watching
        self._capture_now = False
        self._armed = True
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    # ---------------- configuration ----------------
    @staticmethod
    def _new_id() -> str:
        return "AGA-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4].upper()

    @property
    def settings(self):
        return self.rice if self.commodity == "rice" else self.sugar

    def set_source(self, src: Source, commodity: str | None = None, preset: str | None = None) -> None:
        with self.lock:
            old = self.source
            self.source = src
            if commodity:
                self.commodity = commodity
            if preset:
                self.preset = preset
            self._reset_sample()
        try:
            old.close()
        except Exception:
            pass

    def set_commodity(self, commodity: str) -> None:
        with self.lock:
            if commodity == self.commodity:
                return
            self.commodity = commodity
            if isinstance(self.source, SimSource):
                preset = "mixed" if commodity == "rice" else "M-30"
                self.source = SimSource(commodity, preset)
                self.preset = preset
            self._reset_sample()

    def update_settings(self, d: dict) -> None:
        with self.lock:
            if "commodity" in d:
                self.set_commodity(d["commodity"])
            for key in ("lot_id", "officer", "depot"):
                if key in d:
                    self.lot[key] = str(d[key])[:80]
            if "auto_capture" in d:
                self.auto_capture = bool(d["auto_capture"])
            if "use_still" in d:
                self.use_still = bool(d["use_still"])
            if "fallback_px_per_mm" in d:
                self.fallback_ppm = float(d["fallback_px_per_mm"])
            for obj in (self.rice, self.sugar):
                for f in fields(obj):
                    if f.name in d and d[f.name] is not None:
                        val = d[f.name]
                        setattr(obj, f.name, type(getattr(obj, f.name))(val))
            if self.rice.classifier not in ("auto", "model", "rules"):
                self.rice.classifier = "auto"
            if self.rice.grade not in rice_spec()["grades"]:
                self.rice.grade = next(iter(rice_spec()["grades"]))
            self._resummarise()

    def _reset_sample(self) -> None:
        self.shots = []
        self.summary = None
        self.sample_id = self._new_id()
        self._armed = True
        self.result_ver += 1

    def new_sample(self) -> None:
        with self.lock:
            self._reset_sample()

    def capture(self) -> None:
        self._capture_now = True

    def shake(self) -> bool:
        if isinstance(self.source, SimSource):
            self.source.shake()
            return True
        return False

    # ---------------- analysis ----------------
    def _calib(self, img: np.ndarray) -> calibration.Calibration:
        return calibration.detect(img, marker_mm=self.source.marker_mm, fallback_px_per_mm=self.fallback_ppm)

    def _resummarise(self) -> None:
        if not self.shots:
            self.summary = None
        else:
            kernels = [k for s in self.shots for k in s["kernels"]]
            self.summary = pipeline.summarise(kernels, self.shots[0]["commodity"], self.settings)
        self.result_ver += 1

    def _full_analysis(self, frame: np.ndarray) -> None:
        still = getattr(self.source, "still", None)
        if self.use_still and callable(still):
            hi = still()
            if hi is not None:
                frame = hi
        cal = self._calib(frame)
        res = pipeline.analyse(frame, self.commodity, self.settings, cal, self.fallback_ppm,
                               real_camera=not isinstance(self.source, SimSource))
        ann = pipeline.draw(frame, res, numbers=False, thickness=2)
        scale = min(1.0, 1600 / ann.shape[1])
        if scale < 1:
            ann = cv2.resize(ann, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok2, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with self.lock:
            n = len(self.shots)
            for k in res["kernels"]:
                k["shot"] = n
            shot = {"n": n, "commodity": self.commodity, "kernels": res["kernels"], "ms": res["ms"],
                    "calibration": res["calibration"], "backlit": res["backlit"], "image_size": res["image_size"],
                    "ref_length_mm": res.get("ref_length_mm"), "classifier": res.get("classifier"),
                    "jpeg": jpg.tobytes(), "raw": raw.tobytes(),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            if len(self.shots) >= MAX_SHOTS:
                self.shots.pop(0)
            self.shots.append(shot)
            self._resummarise()

    def _live_frame(self, frame: np.ndarray) -> tuple[bytes, dict]:
        scale = min(1.0, LIVE_MAX_W / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        cal = calibration.detect(small, marker_mm=self.source.marker_mm, fallback_px_per_mm=self.fallback_ppm * scale)
        res = pipeline.analyse(small, self.commodity, self.settings, cal, use_model=False)
        out = pipeline.draw(small, res, numbers=False, thickness=1 if self.commodity == "sugar" else 2)
        counts: dict[str, int] = {}
        for k in res["kernels"]:
            counts[k["cls"]] = counts.get(k["cls"], 0) + 1
        ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 72])
        info = {"counts": counts, "n": sum(v for c, v in counts.items() if c != "edge"),
                "calibration": {**cal.info(), "px_per_mm": round(cal.px_per_mm / scale, 2)},
                "resolution": [int(frame.shape[1]), int(frame.shape[0])]}
        return jpg.tobytes(), info

    def _loop(self) -> None:
        prev = None
        still = 0
        t_last = time.monotonic()
        fps = 0.0
        while not self._stop:
            try:
                if self.clients <= 0 and not self._capture_now:
                    time.sleep(0.25)
                    continue
                src = self.source
                frame = src.read()
                if frame is None:
                    self.live["state"] = "no signal"
                    time.sleep(0.2)
                    continue
                thumb = cv2.cvtColor(cv2.resize(frame, (160, int(160 * frame.shape[0] / frame.shape[1]))), cv2.COLOR_BGR2GRAY)
                thumb = cv2.GaussianBlur(thumb, (5, 5), 0).astype(np.int16)
                motion = float(np.mean(np.abs(thumb - prev))) if prev is not None and prev.shape == thumb.shape else 0.0
                prev = thumb
                if motion > MOTION_THR:
                    still = 0
                    self._armed = True
                    state = "moving"
                else:
                    still += 1
                    state = "settling" if still < STILL_FRAMES else "still"

                jpg, info = self._live_frame(frame)
                now = time.monotonic()
                fps = 0.8 * fps + 0.2 * (1 / max(1e-3, now - t_last))
                t_last = now

                do_full = self._capture_now or (self.auto_capture and self._armed and still >= STILL_FRAMES)
                if src is not self.source:
                    continue
                if do_full:
                    self._capture_now = False
                    self._armed = False
                    state = "analysing"
                    self.live.update(state=state)
                    self._full_analysis(frame)
                    state = "analysed"
                elif state == "still" and not self._armed:
                    state = "analysed"
                with self.lock:
                    self.live_jpeg = jpg
                    self.live_ver += 1
                    self.live = {**info, "state": state, "fps": round(fps, 1), "motion": round(motion, 2),
                                 "source": src.label, "source_kind": src.kind, "shots": len(self.shots)}
                self.error = None
            except Exception as e:  # keep the loop alive; report to the UI
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                time.sleep(0.5)

    # ---------------- views ----------------
    def result(self) -> dict:
        with self.lock:
            shots = [{"n": s["n"], "time": s["time"], "ms": s["ms"], "calibration": s["calibration"],
                      "backlit": s["backlit"], "image_size": s["image_size"], "ref_length_mm": s["ref_length_mm"],
                      "classifier": s.get("classifier"), "kernels": [_public(k) for k in s["kernels"]]} for s in self.shots]
            return {"ver": self.result_ver, "sample_id": self.sample_id, "commodity": self.commodity,
                    "grade": self.rice.grade if self.commodity == "rice" else None,
                    "lot": dict(self.lot), "summary": self.summary, "shots": shots,
                    "classes": {k: {"label": v["label"], "color": "#%02x%02x%02x" % v["bgr"][::-1]}
                                for k, v in pipeline.classes_for(self.commodity).items()}}

    def config(self) -> dict:
        from dataclasses import asdict
        from grain.synth import RICE_PRESETS, SUGAR_PRESETS
        return {"commodity": self.commodity, "grades": list(rice_spec()["grades"].keys()),
                "rice": asdict(self.rice), "sugar": asdict(self.sugar), "lot": self.lot,
                "auto_capture": self.auto_capture, "use_still": self.use_still, "fallback_px_per_mm": self.fallback_ppm,
                "model_available": condition_model.available(),
                "presets": {"rice": {k: v["name"] for k, v in RICE_PRESETS.items()},
                            "sugar": {k: v["name"] for k, v in SUGAR_PRESETS.items()}},
                "source": {"kind": self.source.kind, "label": self.source.label, "preset": self.preset}}
