"""Trained condition model inside the pipeline, and the printed-sheet segmentation case."""
import cv2
import numpy as np
import pytest

from grain import calibration, condition_model, pipeline, rice
from grain.segment import segment
from tests import realtray

needs_model = pytest.mark.skipif(not condition_model.available(), reason="model file not present")


def sheet_with_kernels(ppm=12, n=30, seed=0):
    """Whole A4 calibration sheet in view (white paper border) with light 'kernels' on the black area."""
    rng = np.random.default_rng(seed)
    img = cv2.cvtColor(calibration.calibration_sheet(dpi=int(ppm * 25.4)), cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    pts = []
    while len(pts) < n:
        x, y = int(rng.uniform(0.25, 0.75) * W), int(rng.uniform(0.15, 0.85) * H)
        if all(np.hypot(x - a, y - b) > 10 * ppm for a, b in pts):
            pts.append((x, y))
            cv2.ellipse(img, (x, y), (int(2.6 * ppm), int(1.0 * ppm)), float(rng.uniform(0, 180)), 0, 360,
                        (200, 205, 210), -1, cv2.LINE_AA)
    return img, pts


def test_printed_sheet_with_white_border_finds_every_kernel():
    img, pts = sheet_with_kernels()
    cal = calibration.detect(img)
    assert cal.source == "markers"
    objs, bright = segment(img, cal.px_per_mm, cal.exclude_mask)
    assert not bright                       # tray = the black sample area, not the white paper
    assert len(objs) == len(pts)            # no paper / grey patch / tray outline counted as grains


def test_auto_uses_rules_on_simulator_and_reports_it():
    img, _ = sheet_with_kernels(n=10)
    res = pipeline.analyse(img, "rice", rice.RiceSettings(classifier="auto"), real_camera=False)
    assert res["classifier"]["used"] == "rules"
    assert all(k["model"] is None for k in res["kernels"])


def test_rules_setting_never_runs_model():
    img, _ = sheet_with_kernels(n=10)
    res = pipeline.analyse(img, "rice", rice.RiceSettings(classifier="rules"), real_camera=True)
    assert res["classifier"]["used"] == "rules"


def test_low_resolution_falls_back_to_rules():
    img, _ = sheet_with_kernels(ppm=6, n=10)
    res = pipeline.analyse(img, "rice", rice.RiceSettings(classifier="model"), real_camera=True)
    assert res["classifier"]["used"] == "rules" and "px/mm" in res["classifier"]["note"]


@needs_model
def test_model_runs_end_to_end_and_explains_each_kernel():
    img, pts = sheet_with_kernels(n=12)
    res = pipeline.analyse(img, "rice", rice.RiceSettings(classifier="model"), real_camera=True)
    assert res["classifier"]["used"] == "model"
    ks = [k for k in res["kernels"] if k["cls"] != "edge"]
    assert len(ks) == len(pts)
    for k in ks:
        assert abs(sum(k["model"].values()) - 1) < 0.01      # probabilities are rounded to 3 decimals
        assert k["why"] and k["rules_cls"]


@needs_model
@pytest.mark.skipif(not realtray.available(), reason="IIT Indore dataset not downloaded")
def test_real_kernel_tray_model_beats_rules():
    """Real IIT test-split kernels on a camera-like tray (12 px/mm, noise, JPEG)."""
    rows = realtray.test_rows()
    img, truth = realtray.build_tray(rows, 12, realtray.native_ppm(rows), seed=0)
    ev_rules = realtray.evaluate(img, truth, "rules")
    ev_model = realtray.evaluate(img, truth, "model")
    assert len(ev_model["pairs"]) >= 0.95 * len(truth)
    assert ev_model["acc"] >= 0.85
    assert ev_model["acc"] > ev_rules["acc"] + 0.10
