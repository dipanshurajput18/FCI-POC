"""Frame sources: simulated tray, USB webcam, phone / IP camera stream, video or image file.

Each source runs a capture thread that always keeps the newest frame, so slow analysis never
builds up a backlog of stale frames.
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from grain.synth import SimScene


def _check_reachable(url: str) -> None:
    """Fail fast with a clear message instead of OpenCV hanging on an unreachable phone."""
    from urllib.parse import urlparse
    import socket
    u = urlparse(url)
    try:
        socket.create_connection((u.hostname, u.port or 80), timeout=4).close()
    except OSError:
        raise RuntimeError(f"Phone not reachable at {u.hostname}:{u.port or 80}. Check that the phone and this PC "
                           "are on the same Wi-Fi, the camera app's server is started, and the address is correct.")


class Source:
    kind = "base"
    marker_mm = 20.0
    label = ""

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SimSource(Source):
    kind = "sim"

    def __init__(self, commodity: str, preset: str, seed: int = 7):
        self.scene = SimScene(commodity, preset, seed)
        self.marker_mm = self.scene.marker_mm
        self.label = f"Simulated {commodity} tray - {preset}"
        self.fps = 12.0
        self._last = 0.0

    def read(self) -> np.ndarray:
        dt = time.monotonic() - self._last
        if dt < 1 / self.fps:
            time.sleep(1 / self.fps - dt)
        self._last = time.monotonic()
        return self.scene.read()

    def shake(self) -> None:
        self.scene.shake()


class CvSource(Source):
    """OpenCV VideoCapture: webcam index, MJPEG/RTSP URL (phone IP-camera apps) or video file."""
    kind = "camera"

    def __init__(self, spec: str, marker_mm: float = 20.0, width: int = 3840, height: int = 2160,
                 still_url: str | None = None):
        self.marker_mm = marker_mm
        self.still_url = None
        self.spec = spec
        self.is_file = False
        if spec.isdigit():
            idx = int(spec)
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(idx, backend)
            # Ask for the highest resolution; the driver picks the nearest it supports.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.label = f"USB camera {idx}"
        else:
            self.is_file = Path(spec).exists()
            if spec.startswith("http"):
                _check_reachable(spec)
                # IP-camera apps (e.g. Android "IP Webcam") also serve a full-resolution,
                # autofocused photo next to the video stream - used for the actual analysis.
                if still_url is None and "/video" in spec:
                    still_url = spec.rsplit("/video", 1)[0] + "/photo.jpg"
                self.still_url = still_url or None
            self.cap = cv2.VideoCapture(spec)
            self.label = f"Video file {Path(spec).name}" if self.is_file else f"Phone / IP camera {spec}"
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera / stream: {spec}")
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:   # loop video files
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = f
            if self.is_file:
                time.sleep(1 / min(fps, 30))

    def still(self) -> np.ndarray | None:
        """Full-resolution photo from the phone (None if not available)."""
        if not self.still_url:
            return None
        try:
            with urllib.request.urlopen(self.still_url, timeout=15) as r:
                data = r.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def read(self) -> np.ndarray | None:
        for _ in range(100):
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.02)
        return None

    def close(self) -> None:
        self._stop = True
        self._t.join(timeout=1)
        self.cap.release()


class ImageSource(Source):
    """A still photo behaves like a camera that never moves."""
    kind = "image"

    def __init__(self, path: str | None = None, data: bytes | None = None, marker_mm: float = 20.0):
        self.marker_mm = marker_mm
        if data is not None:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            self.label = "Uploaded photo"
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            self.label = f"Photo {Path(path).name}"
        if img is None:
            raise RuntimeError("Could not read image")
        self.img = img

    def read(self) -> np.ndarray:
        time.sleep(1 / 8)
        return self.img


def list_usb_cameras(max_idx: int = 4) -> list[int]:
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found
