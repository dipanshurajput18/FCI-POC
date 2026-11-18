"""Rice: per-kernel classification with explanations, weight estimate, and spec evaluation.

Rule-based (Phase 1A). Each rule writes a plain-language reason so the officer can see *why*
a kernel was classified. Precedence follows the official picking order (IS 4333 Part 1):
one kernel is counted under one refraction only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import spec as specmod
from .segment import Obj

CLASSES = {
    "sound":        {"label": "Sound (whole)",   "bgr": (110, 181, 70)},
    "broken":       {"label": "Broken",          "bgr": (31, 154, 242)},
    "small_broken": {"label": "Small broken",    "bgr": (12, 89, 232)},
    "chalky":       {"label": "Chalky",          "bgr": (245, 167, 59)},
    "red":          {"label": "Red",             "bgr": (60, 60, 225)},
    "discoloured":  {"label": "Discoloured",     "bgr": (0, 205, 235)},
    "damaged":      {"label": "Damaged",         "bgr": (190, 85, 170)},
    "foreign":      {"label": "Foreign matter",  "bgr": (255, 0, 255)},
    "edge":         {"label": "Cut by frame edge (ignored)", "bgr": (120, 120, 120)},
}


DENSITY_G_CM3 = 1.45      # milled rice kernel density (calibrate per variety with the balance)
THICKNESS_RATIO = 0.72    # kernel thickness / breadth


@dataclass
class RiceSettings:
    grade: str = "Raw rice - Grade A"
    sample_g: float = 20.0
    broken_frac: float = 0.75
    small_broken_frac: float = 0.25
    foreign_frac: float = 0.125
    chalky_frac: float = 0.28       # share of kernel area that is opaque white
    chalk_margin: float = 30.0      # brightness margin (0-255) defining "opaque"
    red_frac: float = 0.30          # share of kernel surface that is red
    disc_delta: float = 12.0        # b* above sample median -> yellow / discoloured
    damaged_frac: float = 0.025     # share of kernel with dark spots
    ref_length_mm: float = 0.0      # 0 = estimate whole-kernel length from the sample


def reference_length(lengths_mm: np.ndarray) -> float:
    if len(lengths_mm) == 0:
        return 0.0
    p90 = np.percentile(lengths_mm, 90)
    whole = lengths_mm[lengths_mm >= 0.85 * p90]
    return float(np.median(whole)) if len(whole) else float(p90)


def size_class(L: float, lb: float) -> str:
    """BIS / FCI rice size-shape classes (length mm, L/B ratio)."""
    if L >= 6.0:
        return "Long slender" if lb >= 3.0 else "Long bold"
    if lb >= 3.0:
        return "Short slender"
    if lb >= 2.5:
        return "Medium slender"
    return "Short bold"


def classify(objs: list[Obj], px_per_mm: float, gmed: float, s: RiceSettings) -> dict:
    sp = specmod.rice_spec()
    precedence = sp["precedence"]
    mm = 1.0 / px_per_mm
    valid = [o for o in objs if not o.touches_edge]
    lengths = np.array([o.length_px * mm for o in valid], float)
    ref = s.ref_length_mm if s.ref_length_mm > 0 else reference_length(lengths)
    gmedL = gmed * 100 / 255
    med_b = float(np.median([o.b for o in valid])) if valid else 0.0

    kernels = []
    for o in objs:
        L, B = o.length_px * mm, o.breadth_px * mm
        area = o.area_px * mm * mm
        rel = L / ref if ref else 1.0
        found: dict[str, str] = {}
        review = []
        if o.touches_edge:
            cls = "edge"
        else:
            # Foreign matter: dark stones / mud, brown husk or tiny fragments.
            if o.L < 0.55 * gmedL and o.red_frac < s.red_frac:
                found["foreign"] = f"Dark object (L* {o.L:.0f} vs sample {gmedL:.0f}) - stone / mud"
            elif o.b > med_b + 22 and o.L < 0.8 * gmedL and o.red_frac < s.red_frac:
                found["foreign"] = f"Brown, dull object (b* {o.b:.0f}) - husk / chaff"
            elif rel < s.foreign_frac:
                found["foreign"] = f"Fragment {rel*100:.0f}% of whole length (< {s.foreign_frac*100:.1f}%)"
            if o.red_frac >= s.red_frac:
                found["red"] = f"{o.red_frac*100:.0f}% of surface red-brown (>= {s.red_frac*100:.0f}%)"
            if o.dark_frac >= s.damaged_frac and "foreign" not in found:
                found["damaged"] = f"{o.dark_frac*100:.1f}% dark spots (>= {s.damaged_frac*100:.1f}%)"
            if o.b >= med_b + s.disc_delta and "red" not in found and "foreign" not in found:
                found["discoloured"] = f"Yellowish tone (b* {o.b:.0f} vs sample {med_b:.0f})"
            if rel < s.small_broken_frac and "foreign" not in found:
                found["small_broken"] = f"Length {L:.2f} mm = {rel*100:.0f}% of whole ({ref:.2f} mm), < 1/4"
            elif rel < s.broken_frac:
                found["broken"] = f"Length {L:.2f} mm = {rel*100:.0f}% of whole ({ref:.2f} mm), < 3/4"
            if o.bright_frac >= s.chalky_frac and "red" not in found:
                found["chalky"] = f"{o.bright_frac*100:.0f}% opaque white area (>= {s.chalky_frac*100:.0f}%)"
            cls = next((c for c in precedence if c in found), "sound")
            # Borderline cases go to the officer.
            if abs(rel - s.broken_frac) < 0.04:
                review.append("length close to the broken limit")
            if abs(o.bright_frac - s.chalky_frac) < 0.06:
                review.append("opaque area close to the chalky limit")
            if abs(o.red_frac - s.red_frac) < 0.06:
                review.append("red area close to the limit")
            if o.split and o.solidity < 0.82:
                review.append("touching kernels were split - check outline")
        why = list(found.values()) if found else [f"Whole kernel ({rel*100:.0f}% of reference length), clear and uniform"]
        kernels.append({
            "id": o.id, "cls": cls, "also": [c for c in found if c != cls], "why": why, "review": review,
            "length_mm": round(L, 2), "breadth_mm": round(B, 2), "lb": round(L / B, 2) if B else 0,
            "area_mm2": round(area, 2), "rel_length": round(rel, 3),
            "L": round(o.L, 1), "a": round(o.a, 1), "b": round(o.b, 1),
            "opaque_pct": round(o.bright_frac * 100, 1), "red_pct": round(o.red_frac * 100, 1),
            "dark_pct": round(o.dark_frac * 100, 1),
            "bbox": [int(v) for v in o.bbox], "cx": round(o.centroid[0], 1), "cy": round(o.centroid[1], 1),
            # Ellipsoid volume with thickness ~ THICKNESS_RATIO x breadth; mass = density x volume.
            "_vol": math.pi / 6 * L * B * (THICKNESS_RATIO * B),
        })
    return {"kernels": kernels, "ref_length_mm": round(ref, 2)}


def summarise(kernels: list[dict], s: RiceSettings) -> dict:
    """Aggregate (possibly over several shots) into parameter %, verdict and statistics."""
    valid = [k for k in kernels if k["cls"] != "edge"]
    for k in valid:
        k["mass_g"] = round(DENSITY_G_CM3 * k["_vol"] / 1000, 5)
    imaged_g = sum(k["mass_g"] for k in valid)
    mass, count = {}, {}
    for k in valid:
        mass[k["cls"]] = mass.get(k["cls"], 0.0) + k["_vol"]
        count[k["cls"]] = count.get(k["cls"], 0) + 1
    rows, verdict = specmod.evaluate_rice(mass, count, s.grade)
    for r in rows:
        r["grams"] = round(s.sample_g * r["wt_pct"] / 100, 3) if s.sample_g else None   # projected to the sample
    whole = [k for k in valid if k["cls"] in ("sound", "chalky", "red", "discoloured", "damaged")]
    mean_L = float(np.mean([k["length_mm"] for k in whole])) if whole else 0.0
    mean_lb = float(np.mean([k["lb"] for k in whole])) if whole else 0.0
    return {
        "params": rows, "verdict": verdict,
        "counts": {c: count.get(c, 0) for c in CLASSES if c != "edge"},
        "n_kernels": len(valid), "n_review": sum(1 for k in valid if k["review"]),
        "imaged_g": round(imaged_g, 3), "sample_g": s.sample_g,
        "mean_length_mm": round(mean_L, 2), "mean_lb": round(mean_lb, 2),
        "size_class": size_class(mean_L, mean_lb) if whole else "-",
        "not_assessed": specmod.rice_spec()["not_assessed"],
    }
