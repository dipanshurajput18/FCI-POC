"""Trays built from REAL rice kernels (IIT Indore test split, never seen in training).

Each kernel photo is scaled to camera resolution, rotated at random and laid on the printed
calibration sheet, with ArUco markers, the black sample area, slight blur, noise and JPEG
compression. The full app pipeline then runs on the tray, as on a camera frame. Each detection is
matched to the kernel placed there and compared with its FCI label (data/public/iit_indore/fci_labels.csv).

Usage (from the repo root, needs the downloaded dataset):
    python -m tests.realtray                    # 3 trays x {12, 18} px/mm, rules vs AI model
    python -m tests.realtray --save             # also writes an annotated tray image
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from grain import calibration, pipeline, rice
from grain.crop import mask_from_black_bg

ROOT = Path(__file__).resolve().parent.parent
IIT = ROOT / "data" / "public" / "iit_indore"
FCI_CLASSES = ["sound", "broken", "small_broken", "chalky", "damaged", "discoloured", "red", "foreign"]
# kernels per tray from each source class (roughly a poor-quality lot, so every class is exercised)
MIX = {"Healthy": 40, "Broken": 20, "Full chalky": 15, "Partial chalky": 15, "Peck damage": 12, "Discolored": 18}


def available() -> bool:
    return (IIT / "fci_labels.csv").exists() and (IIT / "extracted" / "Data" / "Segmented image").exists()


def test_rows() -> list[dict]:
    return [r for r in csv.DictReader(open(IIT / "fci_labels.csv", encoding="utf-8"))
            if r["variety"] == "BPT5204" and r["split"] == "test" and r["image_downloaded"] == "True"]


def _kernel(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        return None
    m = mask_from_black_bg(img)
    if m is None:
        return None
    ys, xs = np.where(m > 0)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def native_ppm(rows: list[dict], n: int = 40) -> float:
    """px/mm of the 1/8-decoded dataset images, from the dataset's own length_mm."""
    r = []
    for row in rows[:n]:
        k = _kernel(IIT / "extracted" / row["image"])
        if k is None:
            continue
        pts = cv2.findNonZero(k[1])
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        r.append(max(w, h) / float(row["length_mm"]))
    return float(np.median(r))


def build_tray(rows: list[dict], ppm: float, nat_ppm: float, seed: int = 0,
               camera: bool = True) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["source_class"], []).append(r)
    pick = []
    for cls, n in MIX.items():
        pool = by.get(cls, [])
        pick += [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
    rng.shuffle(pick)

    sheet = cv2.cvtColor(calibration.calibration_sheet(dpi=int(round(ppm * 25.4))), cv2.COLOR_GRAY2BGR)
    mm = ppm
    H, W = sheet.shape[:2]
    margin, side = int(12 * mm), int(20 * mm)
    x0, y0 = margin + side + int(10 * mm), margin + int(3 * mm)
    x1, y1 = W - margin - side - int(10 * mm), H - margin - int(3 * mm)
    cell = int(8.0 * mm)
    slots = [(x, y) for y in range(y0, y1 - cell, cell) for x in range(x0, x1 - cell, cell)]
    order = rng.permutation(len(slots))
    s = ppm / nat_ppm
    truth = []
    for r, si in zip(pick, order):
        k = _kernel(IIT / "extracted" / r["image"])
        if k is None:
            continue
        img, m = k
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        ang = rng.uniform(0, 180)
        d = int(np.hypot(*img.shape[:2])) + 2
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, 1)
        M[0, 2] += d / 2 - img.shape[1] / 2
        M[1, 2] += d / 2 - img.shape[0] / 2
        img = cv2.warpAffine(img, M, (d, d), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(m, M, (d, d), flags=cv2.INTER_LINEAR)[..., None]
        cx, cy = slots[si]
        cx += cell // 2 + int(rng.integers(-cell // 10, cell // 10 + 1))
        cy += cell // 2 + int(rng.integers(-cell // 10, cell // 10 + 1))
        ys, xs = cy - d // 2, cx - d // 2
        roi = sheet[ys:ys + d, xs:xs + d].astype(np.float32)
        sheet[ys:ys + d, xs:xs + d] = (roi * (1 - a) + img.astype(np.float32) * a).astype(np.uint8)
        truth.append({"cx": cx, "cy": cy, "fci": r["fci_class"], "src": r["source_class"], "id": r["grain_id"]})
    if not camera:
        return sheet, truth
    # camera: slight blur, sensor noise, JPEG
    sheet = cv2.GaussianBlur(sheet, (0, 0), 0.6)
    sheet = np.clip(sheet.astype(np.int16) + rng.normal(0, 2.0, sheet.shape).astype(np.int16), 0, 255).astype(np.uint8)
    sheet = cv2.imdecode(cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
    return sheet, truth


def evaluate(img: np.ndarray, truth: list[dict], classifier: str) -> dict:
    s = rice.RiceSettings(classifier=classifier)
    res = pipeline.analyse(img, "rice", s, calibration.detect(img), real_camera=True)
    ks = [k for k in res["kernels"] if k["cls"] != "edge"]
    pairs, used = [], set()
    for t in truth:
        best, bd = None, 1e9
        for k in ks:
            d = np.hypot(k["cx"] - t["cx"], k["cy"] - t["cy"])
            if d < bd and k["id"] not in used:
                best, bd = k, d
        if best is not None and bd < 0.25 * np.hypot(*best["bbox"][2:]) + 4:
            used.add(best["id"])
            pairs.append((t, best))
    return {"res": res, "pairs": pairs, "n_truth": len(truth), "n_det": len(ks),
            "acc": float(np.mean([t["fci"] == k["cls"] for t, k in pairs])) if pairs else 0.0}


def report(name: str, pairs: list) -> None:
    acc = np.mean([t["fci"] == k["cls"] for t, k in pairs])
    print(f"  {name:7s} accuracy {acc*100:5.1f}%  ({len(pairs)} kernels)")
    per = {}
    for t, k in pairs:
        per.setdefault(t["fci"], Counter())[k["cls"]] += 1
    for c in FCI_CLASSES:
        if c in per:
            n = sum(per[c].values())
            wrong = ", ".join(f"{p} {v}" for p, v in per[c].most_common() if p != c)
            print(f"     {c:13s} {per[c][c]:3d}/{n:<3d} {per[c][c]/n*100:5.1f}%   {('-> ' + wrong) if wrong else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trays", type=int, default=3)
    ap.add_argument("--ppm", type=float, nargs="*", default=[12, 18])
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    rows = test_rows()
    nat = native_ppm(rows)
    print(f"{len(rows)} test-split kernels, dataset images at {nat:.1f} px/mm (1/8 decode)")
    for ppm in a.ppm:
        allp = {"rules": [], "model": []}
        for seed in range(a.trays):
            img, truth = build_tray(rows, ppm, nat, seed)
            for clf in allp:
                ev = evaluate(img, truth, clf)
                allp[clf] += ev["pairs"]
                if a.save and seed == 0 and clf == "model":
                    out = IIT / "preview" / f"realtray_{ppm:g}ppm_model.jpg"
                    cv2.imwrite(str(out), pipeline.draw(img, ev["res"]), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print("  saved", out.relative_to(ROOT))
            print(f"tray {seed} @ {ppm:g} px/mm: {len(truth)} placed, {ev['n_det']} detected, "
                  f"calibration {ev['res']['calibration']['px_per_mm']:.2f} px/mm")
        print(f"== {ppm:g} px/mm, {a.trays} trays")
        for clf, p in allp.items():
            report(clf, p)


if __name__ == "__main__":
    main()
