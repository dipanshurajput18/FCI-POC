"""Relabel the IIT Indore rice dataset into FCI categories.

Rules: specs/label_map_iit_indore.yaml (+ FCI thresholds in specs/rice_kms_2026_27.yaml).
Traceability: docs/FCI_REQUIREMENTS_TRACE.md section 4.

Usage (from AI-anaylser/):
    .venv\\Scripts\\python -m training.relabel_iit            # length rules + chalky where images exist
    .venv\\Scripts\\python -m training.relabel_iit --no-images  # metadata only

Output: data/public/iit_indore/fci_labels.csv  (+ a summary printed to the console)
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "public" / "iit_indore" / "extracted" / "Data"
OUT = ROOT / "data" / "public" / "iit_indore" / "fci_labels.csv"


def load_rules() -> tuple[dict, dict]:
    lm = yaml.safe_load(open(ROOT / "specs" / "label_map_iit_indore.yaml", encoding="utf-8"))
    spec = yaml.safe_load(open(ROOT / "specs" / "rice_kms_2026_27.yaml", encoding="utf-8"))
    return lm, spec["thresholds"]


def fci_label(source_class: str, length_mm: float, whole_mm: float, chalky_fraction: float | None,
              lm: dict, thr: dict) -> tuple[str, str]:
    """Return (fci_class, rule) for one kernel, following the IS 4333 picking order."""
    entry = lm["labels"].get(source_class)
    if entry is None:
        return "unmapped", f"no mapping for source label '{source_class}'"
    cond = entry["fci"]
    # 1. damaged / discoloured outrank broken in the picking order
    if cond in ("damaged", "discoloured"):
        return cond, f"source '{source_class}' -> {cond}"
    # 2. FCI length rule (never the source's own broken label)
    rel = length_mm / whole_mm if whole_mm else 1.0
    if rel < thr["small_broken_frac"]:
        return "small_broken", f"length {length_mm:.2f} mm = {rel:.0%} of whole {whole_mm:.2f} mm (< 1/4)"
    if rel < thr["broken_frac"]:
        return "broken", f"length {length_mm:.2f} mm = {rel:.0%} of whole {whole_mm:.2f} mm (< 3/4)"
    # 3. condition class
    if cond == "chalky_if_measured":
        if chalky_fraction is None:
            return "chalky_unmeasured", "partial chalky - milky fraction not yet measured (image missing)"
        if chalky_fraction >= lm["chalky_min_fraction"]:
            return "chalky", f"measured milky fraction {chalky_fraction:.2f} >= {lm['chalky_min_fraction']}"
        return "sound", f"measured milky fraction {chalky_fraction:.2f} < {lm['chalky_min_fraction']} (not FCI chalky)"
    if source_class == "Broken":
        return "sound", f"source 'Broken' but length {rel:.0%} of whole >= 3/4 -> whole kernel under FCI rule"
    return cond, f"source '{source_class}' -> {cond}; length {rel:.0%} of whole"


# ---------------- chalky fraction measurement ----------------
def image_path(variety: str, source_class: str, image: str) -> Path:
    group = "BPT5204" if variety == "BPT5204" else "Basmati and non-basmati"
    folder = source_class.replace(" ", "_") if variety == "BPT5204" else variety
    return DATA / "Segmented image" / group / folder / image


def kernel_L(path: Path) -> np.ndarray | None:
    """L* (0-255) of kernel pixels, decoded at 1/8 scale (750 x 500) for speed."""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 12).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = cv2.erode((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8))  # drop the dim rim
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0][m > 0]


def calibrate_threshold(rows: list[dict], n: int = 150) -> tuple[float | None, dict]:
    """Milky-white pixel threshold: Otsu split between Healthy (translucent) and Full chalky pixels."""
    rng = np.random.default_rng(0)
    pools = {}
    for cls in ("Healthy", "Full chalky"):
        cand = [r for r in rows if r["class"] == cls and image_path(r["variety"], cls, r["image"]).exists()]
        if not cand:
            return None, {"reason": f"no {cls} images downloaded yet"}
        pick = rng.choice(len(cand), min(n, len(cand)), replace=False)
        px = [kernel_L(image_path(cand[i]["variety"], cls, cand[i]["image"])) for i in pick]
        pools[cls] = np.concatenate([p for p in px if p is not None])
    allpx = np.concatenate(list(pools.values())).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(allpx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    diag = {"threshold_L": float(t),
            "healthy_pixels_above": float(np.mean(pools["Healthy"] > t)),
            "full_chalky_pixels_above": float(np.mean(pools["Full chalky"] > t))}
    return float(t), diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true", help="skip chalky measurement")
    args = ap.parse_args()
    lm, thr = load_rules()
    rows = list(csv.DictReader(open(DATA / "Metadata" / "feature_values.csv", encoding="utf-8")))
    splits = {}
    for r in csv.DictReader(open(DATA / "Metadata" / "train_val_test_splits.csv", encoding="utf-8")):
        splits.setdefault(r["grain_id"], r["split"])

    # Whole-kernel reference length per variety.
    by_var = defaultdict(list)
    for r in rows:
        if r["class"] in lm["whole_length_from"] or r["variety"] != "BPT5204":
            by_var[r["variety"]].append(float(r["length_mm"]))
    whole = {v: statistics.median(ls) for v, ls in by_var.items()}

    t, diag = (None, {"reason": "--no-images"}) if args.no_images else calibrate_threshold(rows)
    print("whole-kernel length (mm):", {k: round(v, 2) for k, v in whole.items()})
    print("chalky calibration:", diag)

    out, counts, trans = [], Counter(), Counter()
    for r in rows:
        src, var, L = r["class"], r["variety"], float(r["length_mm"])
        frac = None
        if t is not None and lm["labels"].get(src, {}).get("fci") == "chalky_if_measured":
            p = image_path(var, src, r["image"])
            if p.exists():
                px = kernel_L(p)
                frac = float(np.mean(px > t)) if px is not None and len(px) else None
        fci, rule = fci_label(src, L, whole[var], frac, lm, thr)
        pending = lm["labels"].get(src, {}).get("pending", "")
        img = image_path(var, src, r["image"])
        out.append({"grain_id": r["grain_id"], "variety": var, "source_class": src, "fci_class": fci,
                    "rule": rule, "pending_fci_decision": pending, "length_mm": round(L, 3),
                    "whole_length_mm": round(whole[var], 3), "rel_length": round(L / whole[var], 3),
                    "chalky_fraction": "" if frac is None else round(frac, 3),
                    "split": splits.get(r["grain_id"], ""), "image_downloaded": img.exists(),
                    "image": str(img.relative_to(DATA.parent)).replace("\\", "/")})
        counts[fci] += 1
        trans[(src, fci)] += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {len(out)} rows -> {OUT.relative_to(ROOT)}")
    print("\nsource label -> FCI class:")
    for (s, c), n in sorted(trans.items()):
        print(f"  {s:15s} -> {c:18s} {n:6d}")
    print("\nFCI class totals:", dict(counts))


if __name__ == "__main__":
    main()
