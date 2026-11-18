"""Regression tests on synthetic trays with known ground truth.

These guard the classical pipeline. Real-image accuracy is validated separately against
manual FCI analysis (see docs/PHASE1_PLAN.md section 7).
"""
import collections

import numpy as np
import pytest

from grain import calibration, rice, spec, sugar, synth
from tests.evalsim import run


def accuracy(pairs):
    return sum(1 for t, k in pairs if t["kind"] == k["cls"]) / max(1, len(pairs))


def test_specs_load():
    r = spec.rice_spec()
    assert "Raw rice - Grade A" in r["grades"]
    assert r["grades"]["Raw rice - Improved Rice (KMS 2026-27)"]["broken"] == 10.0
    s = spec.sugar_spec()
    assert s["group_order"] == ["L", "M", "S", "SS"]


def test_calibration_scale():
    sc = synth.SimScene("rice", "faq", seed=3)
    cal = calibration.detect(sc.read(), marker_mm=sc.marker_mm)
    assert cal.source == "markers" and cal.markers == 4
    assert abs(cal.px_per_mm - sc.ppm) / sc.ppm < 0.02


def test_calibration_sheet_has_markers():
    sheet = calibration.calibration_sheet(dpi=150)
    import cv2
    cal = calibration.detect(cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR), marker_mm=20)
    assert cal.markers == 4
    assert abs(cal.px_per_mm - 150 / 25.4) < 0.15


@pytest.mark.parametrize("preset", ["faq", "broken", "chalky", "red", "mixed", "foreign"])
def test_rice_kernel_accuracy(preset):
    sc, img, res, summ, truth, pairs, n_det = run("rice", preset)
    assert len(pairs) >= 0.95 * len(truth), "kernels missed"
    assert accuracy(pairs) >= 0.93


def test_rice_verdicts_follow_presets():
    def beyond(preset):
        return {r["key"] for r in run("rice", preset)[3]["params"] if r["status"] == "beyond"}
    assert run("rice", "faq")[3]["verdict"]["code"] == "WITHIN"
    assert "broken" in beyond("broken")
    assert "chalky" in beyond("chalky")
    assert "red" in beyond("red")
    assert "foreign" in beyond("foreign")


def test_rice_length_measurement():
    sc, img, res, summ, truth, pairs, n_det = run("rice", "faq")
    err = [k["length_mm"] - t["L_mm"] for t, k in pairs if t["kind"] == "sound"]
    assert abs(np.mean(err)) < 0.15 and np.std(err) < 0.2


@pytest.mark.parametrize("preset,group", [("L-31", "L"), ("M-30", "M"), ("S-30", "S")])
def test_sugar_size_group_and_no_false_specks(preset, group):
    sc, img, res, summ, truth, pairs, n_det = run("sugar", preset)
    assert summ["size_group"] == group
    assert summ["counts"]["speck"] == 0
    err = [k["breadth_mm"] - t["B_mm"] for t, k in pairs]
    assert abs(np.mean(err)) < 0.03


def test_sugar_specks_detected():
    sc, img, res, summ, truth, pairs, n_det = run("sugar", "specks")
    assert summ["counts"]["speck"] >= 7
    false = sum(1 for t, k in pairs if k["cls"] == "speck" and t["kind"] != "speck")
    assert false <= 1


def test_sugar_colour_grade():
    assert run("sugar", "M-29")[3]["colour_grade"] == "29"
    assert run("sugar", "L-31")[3]["colour_grade"] == "31"


def test_api_smoke():
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    cfg = c.get("/api/config").json()
    assert cfg["commodity"] in ("rice", "sugar") and cfg["grades"]
    png = c.get("/api/calibration-sheet.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
