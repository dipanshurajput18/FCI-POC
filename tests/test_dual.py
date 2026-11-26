"""Dual-camera glass rig: alignment, pairing and fusion on the two-sided simulator."""
from types import SimpleNamespace

import server.dual as D
from grain import rice


def _engine():
    main = SimpleNamespace(rice=rice.RiceSettings(), lot={"lot_id": "T", "officer": "T", "depot": "T"})
    return D.DualEngine(main)


def test_dual_finds_underside_defects():
    d = _engine()
    src = D.DualSimSource("mixed")
    d.set_source(src, "mixed")
    top, bot = src.read_pair()
    d._analyse_pair(top, bot)
    shot = d.shots[-1]
    assert shot["align"].startswith("markers")
    assert shot["align_err_mm"] < 0.3
    assert d.stats["paired"] >= 170
    assert d.stats["hidden_kernels"] >= 0.8 * src.hidden_truth()


def test_dual_aligns_without_markers(monkeypatch):
    monkeypatch.setattr(D, "marker_centres", lambda img: {})
    d = _engine()
    src = D.DualSimSource("faq")
    d.set_source(src, "faq")
    d._analyse_pair(*src.read_pair())
    assert d.shots[-1]["align"].startswith("grain positions")
    assert d.shots[-1]["align_err_mm"] < 0.3
    assert d.stats["hidden_kernels"] >= 0.8 * src.hidden_truth()


def test_fuse_rules():
    base = {"why": ["x"], "also": [], "review": [], "bbox": [0, 0, 1, 1], "cx": 0, "cy": 0}
    top = {**base, "cls": "chalky"}
    bot = {**base, "cls": "sound"}
    assert D.fuse(top, bot, "either")["cls"] == "chalky"
    assert D.fuse(top, bot, "both")["cls"] == "sound"
    f = D.fuse({**base, "cls": "broken"}, {**base, "cls": "damaged"}, "either")
    assert f["cls"] == "damaged" and "broken" in f["also"] and f["hidden"] == ["damaged"]
