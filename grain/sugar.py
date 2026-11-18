"""Sugar: crystal size distribution -> IS 498 size group, whiteness proxy, specks.

A crystal passes a square-mesh sieve when its smallest cross-section is below the aperture,
so the "sieve size" of a crystal is taken as its minor (breadth) dimension. Mass is estimated
from projected area^1.5 (volume of a roughly equant crystal), then the virtual sieve gives
% retained on each IS 498 aperture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec as specmod
from .segment import Obj

CLASSES = {
    "crystal": {"label": "Crystal",                 "bgr": (110, 181, 70)},
    "lump":    {"label": "Lump / agglomerate",      "bgr": (31, 154, 242)},
    "speck":   {"label": "Black speck / particle",  "bgr": (255, 0, 255)},
    "edge":    {"label": "Cut by frame edge (ignored)", "bgr": (120, 120, 120)},
}

APERTURES = [2.00, 1.70, 1.18, 0.85, 0.60, 0.425, 0.30, 0.212, 0.175, 0.147]


@dataclass
class SugarSettings:
    sample_g: float = 100.0
    lump_factor: float = 6.0        # area above this x median -> lump
    speck_L: float = 45.0           # mean L* below -> dark particle
    speck_spot_frac: float = 0.01   # share of crystal covered by an interior dark blob -> speck


def classify(objs: list[Obj], px_per_mm: float, s: SugarSettings) -> dict:
    mm = 1.0 / px_per_mm
    valid = [o for o in objs if not o.touches_edge]
    med_area = float(np.median([o.area_px for o in valid])) if valid else 1.0
    out = []
    for o in objs:
        L, B = o.length_px * mm, o.breadth_px * mm
        area = o.area_px * mm * mm
        why, review = [], []
        if o.touches_edge:
            cls = "edge"
        elif o.L < s.speck_L:
            cls = "speck"
            why.append(f"Dark particle (L* {o.L:.0f} < {s.speck_L:.0f})")
        elif o.dark_frac >= s.speck_spot_frac:
            cls = "speck"
            why.append(f"{o.dark_frac*100:.1f}% dark inclusion inside crystal")
        elif o.area_px > s.lump_factor * med_area and o.solidity < 0.9:
            cls = "lump"
            why.append(f"Area {o.area_px/med_area:.1f}x median, irregular (solidity {o.solidity:.2f})")
        else:
            cls = "crystal"
            why.append(f"Sieve size {B:.2f} mm (minor dimension)")
        if o.split and o.solidity < 0.8:
            review.append("touching crystals were split - check outline")
        out.append({
            "id": o.id, "cls": cls, "also": [], "why": why, "review": review,
            "length_mm": round(L, 3), "breadth_mm": round(B, 3), "lb": round(L / B, 2) if B else 0,
            "area_mm2": round(area, 4), "sieve_mm": round(B, 3),
            "L": round(o.L, 1), "a": round(o.a, 1), "b": round(o.b, 1), "dark_pct": round(o.dark_frac * 100, 1),
            "bbox": [int(v) for v in o.bbox], "cx": round(o.centroid[0], 1), "cy": round(o.centroid[1], 1),
            "_vol": area ** 1.5,
        })
    return {"kernels": out}


def _percentile_size(sizes: np.ndarray, w: np.ndarray, q: float) -> float:
    """Size at which q% of the mass is finer (mass-weighted percentile)."""
    order = np.argsort(sizes)
    cs = np.cumsum(w[order]) / w.sum()
    return float(np.interp(q / 100, cs, sizes[order]))


def summarise(crystals: list[dict], s: SugarSettings) -> dict:
    sp = specmod.sugar_spec()
    valid = [c for c in crystals if c["cls"] in ("crystal", "lump")]
    counts = {k: sum(1 for c in crystals if c["cls"] == k) for k in CLASSES if k != "edge"}
    if not valid:
        return {"counts": counts, "n_kernels": 0, "distribution": [], "size_group": "-", "params": [],
                "verdict": {"code": "NONE", "text": "No crystals found", "params": []}}
    sizes = np.array([c["sieve_mm"] for c in valid])
    w = np.array([c["_vol"] for c in valid])
    tot = w.sum()
    for c in valid:   # sucrose 1.587 g/cm3; equant crystal volume ~ 0.6 x area^1.5
        c["mass_g"] = round(1.587 * 0.6 * c["_vol"] / 1000, 7)
    dist = []
    for a in APERTURES:
        dist.append({"aperture_mm": a, "retained_pct": round(100 * w[sizes > a].sum() / tot, 1)})
    retained = {d["aperture_mm"]: d["retained_pct"] for d in dist}

    group, checks = None, []
    for g in sp["group_order"]:
        req = sp["size_groups"][g]
        ok = all(retained.get(float(ap), 0) >= mn for ap, mn in req.items())
        checks.append({"group": g, "ok": ok,
                       "rules": [{"aperture_mm": float(ap), "min_pct": mn, "actual_pct": retained.get(float(ap), 0)}
                                 for ap, mn in req.items()]})
        if ok and group is None:
            group = g
    d50 = _percentile_size(sizes, w, 50)
    d16, d84 = _percentile_size(sizes, w, 16), _percentile_size(sizes, w, 84)
    cv = 100 * (d84 - d16) / (2 * d50) if d50 else 0

    # Whiteness proxy (indicative until calibrated against NSI standards).
    Ls = np.array([c["L"] for c in valid]); bs = np.array([c["b"] for c in valid])
    mean_L, mean_b = float(np.average(Ls, weights=w)), float(np.average(bs, weights=w))
    colour = "below 29"
    for g, t in sp["colour_grades"].items():
        if mean_L >= t["min_L"] and mean_b <= t["max_b"]:
            colour = g
            break
    specks = counts.get("speck", 0)
    per_k = 1000 * specks / max(1, len(valid))
    speck_alert = per_k >= sp["specks"]["alert_per_1000_crystals"]

    grade = f"{group}-{colour}" if group and colour != "below 29" else (group or "Off-grade (size)")
    params = [
        {"key": "size", "name": "Size group (IS 498 virtual sieve)", "value": group or "None", "status": "within" if group else "beyond"},
        {"key": "ma", "name": "Mean aperture MA (mm)", "value": round(d50, 3), "status": "info"},
        {"key": "cv", "name": "Coefficient of variation CV (%)", "value": round(cv, 1), "status": "info"},
        {"key": "colour", "name": "Colour grade (whiteness proxy, indicative)", "value": colour,
         "status": "within" if colour != "below 29" else "beyond"},
        {"key": "specks", "name": "Black specks / particles per 1000 crystals", "value": round(per_k, 1),
         "status": "beyond" if speck_alert else "within"},
        {"key": "lumps", "name": "Lumps / agglomerates", "value": counts.get("lump", 0), "status": "info"},
    ]
    beyond = [p["name"] for p in params if p["status"] == "beyond"]
    verdict = ({"code": "BEYOND", "text": f"Grade {grade} - check flagged parameters", "params": beyond} if beyond
               else {"code": "WITHIN", "text": f"Grade {grade}", "params": []})
    return {
        "counts": counts, "n_kernels": len(valid), "n_review": sum(1 for c in crystals if c["review"]),
        "imaged_g": round(sum(c["mass_g"] for c in valid), 4), "sample_g": s.sample_g,
        "distribution": dist, "group_checks": checks, "size_group": group or "-", "colour_grade": colour,
        "grade": grade, "ma_mm": round(d50, 3), "cv_pct": round(cv, 1), "mean_L": round(mean_L, 1),
        "mean_b": round(mean_b, 1), "params": params, "verdict": verdict, "not_assessed": sp["not_assessed"],
    }
