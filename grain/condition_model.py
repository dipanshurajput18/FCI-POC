"""Trained rice condition classifier (sound / chalky / damaged / discoloured), ONNX on CPU.

Settings and the FCI class mapping live in specs/rice_kms_2026_27.yaml -> condition_model.
Model card: models/README.md. Uses onnxruntime when installed, otherwise OpenCV DNN
(same results; no extra dependency).

The model judges condition only. Length (broken / small broken), foreign matter and red
kernels stay with the FCI rules in grain/rice.py: the model was not trained on them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import spec as specmod
from .crop import canonical_crop
from .segment import Obj

ROOT = Path(__file__).resolve().parent.parent


class ConditionModel:
    def __init__(self, onnx_path: Path):
        meta_path = onnx_path.with_suffix(".json")
        self.meta = json.load(open(meta_path, encoding="utf-8"))
        self.name = self.meta["name"]
        self.classes: list[str] = self.meta["classes"]
        inp = self.meta["input"]
        self.size = int(inp["size"])
        self.mean = np.array(inp["mean"], np.float32)
        self.std = np.array(inp["std"], np.float32)
        try:
            import onnxruntime as ort
            self._ort = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
        except ImportError:
            self._ort = None
            self._net = cv2.dnn.readNetFromONNX(str(onnx_path))
            self.backend = "opencv-dnn"

    def _run(self, batch: np.ndarray) -> np.ndarray:
        if self._ort is not None:
            return self._ort.run(None, {"input": batch})[0]
        self._net.setInput(batch)
        return self._net.forward()

    def crops(self, img: np.ndarray, objs: list[Obj]) -> list[np.ndarray | None]:
        out = []
        H, W = img.shape[:2]
        for o in objs:
            x, y, w, h = o.bbox
            p = 3
            x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
            roi = img[y0:y1, x0:x1]
            m = np.zeros(roi.shape[:2], np.uint8)
            cv2.drawContours(m, [o.contour - [x0, y0]], -1, 1, -1)
            out.append(canonical_crop(roi, m, self.size))
        return out

    def predict(self, img: np.ndarray, objs: list[Obj], bs: int = 64) -> dict[int, dict[str, float]]:
        """Return {obj.id: {class: probability}} for every object that could be cropped."""
        crops = self.crops(img, objs)
        ids = [o.id for o, c in zip(objs, crops) if c is not None]
        X = [c for c in crops if c is not None]
        probs = {}
        for i in range(0, len(X), bs):
            x = np.stack(X[i:i + bs])[..., ::-1].astype(np.float32) / 255.0
            x = ((x - self.mean) / self.std).transpose(0, 3, 1, 2).copy()
            z = self._run(x)
            z = np.exp(z - z.max(1, keepdims=True))
            z /= z.sum(1, keepdims=True)
            for j, row in enumerate(z):
                probs[ids[i + j]] = {c: float(v) for c, v in zip(self.classes, row)}
        return probs


def config() -> dict:
    return specmod.rice_spec().get("condition_model", {})


@lru_cache(maxsize=1)
def load() -> ConditionModel | None:
    """The configured model, or None if the file is missing (the app then runs on rules only)."""
    cfg = config()
    path = ROOT / cfg.get("file", "models/rice_condition_v1.onnx")
    if not path.exists() or not path.with_suffix(".json").exists():
        return None
    return ConditionModel(path)


def available() -> bool:
    return load() is not None
