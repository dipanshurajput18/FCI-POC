"""Build the training cache for the rice condition classifier.

Reads data/public/iit_indore/fci_labels.csv, crops every downloaded BPT5204 kernel with
grain.crop.canonical_crop and writes data/cache/rice_condition_128.npz:
    X (N,128,128,3) uint8 BGR, y (N,) int, split (N,) str, grain_id (N,) str

Condition labels (the model's job; brokenness stays with the FCI length rule):
    sound       Healthy, Broken (source condition), Partial chalky measured < 1/2 milky
    chalky      Full chalky, Partial chalky measured >= 1/2 milky
    damaged     Peck damage
    discoloured Discolored (kept separate so FCI Q1 can be decided in the YAML, not by retraining)

Split: the official IIT Indore 6class split where a grain has one; other grains are
assigned 70/15/15 per class with a fixed seed.

Usage (from AI-anaylser/):  .venv\\Scripts\\python -m training.build_cache
"""
from __future__ import annotations

import csv
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from grain.crop import canonical_crop, mask_from_black_bg

ROOT = Path(__file__).resolve().parent.parent
IIT = ROOT / "data" / "public" / "iit_indore"
OUT = ROOT / "data" / "cache" / "rice_condition_128.npz"
CLASSES = ["sound", "chalky", "damaged", "discoloured"]


def condition(r: dict) -> str | None:
    src = r["source_class"]
    if src in ("Healthy", "Broken"):
        return "sound"
    if src == "Full chalky":
        return "chalky"
    if src == "Partial chalky":
        if r["chalky_fraction"] == "":
            return None
        return "chalky" if float(r["chalky_fraction"]) >= 0.5 else "sound"
    if src == "Peck damage":
        return "damaged"
    if src == "Discolored":
        return "discoloured"
    return None


def load(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        return None
    m = mask_from_black_bg(img)
    return None if m is None else canonical_crop(img, m)


def main() -> None:
    rows = [r for r in csv.DictReader(open(IIT / "fci_labels.csv", encoding="utf-8"))
            if r["variety"] == "BPT5204" and r["image_downloaded"] == "True" and condition(r)]
    official = {}
    for r in csv.DictReader(open(IIT / "extracted/Data/Metadata/train_val_test_splits.csv", encoding="utf-8")):
        if r["class_set"] == "6class":
            official[r["grain_id"]] = r["split"]
    rng = np.random.default_rng(42)
    split = {}
    for cls in {r["source_class"] for r in rows}:
        rest = [r["grain_id"] for r in rows if r["source_class"] == cls and r["grain_id"] not in official]
        rng.shuffle(rest)
        n = len(rest)
        for i, g in enumerate(rest):
            split[g] = "train" if i < 0.70 * n else ("val" if i < 0.85 * n else "test")
    split.update(official)

    paths = [str(IIT / "extracted" / r["image"]) for r in rows]
    with Pool(6) as p:
        crops = p.map(load, paths, chunksize=32)
    keep = [i for i, c in enumerate(crops) if c is not None]
    X = np.stack([crops[i] for i in keep])
    y = np.array([CLASSES.index(condition(rows[i])) for i in keep])
    sp = np.array([split[rows[i]["grain_id"]] for i in keep])
    gid = np.array([rows[i]["grain_id"] for i in keep])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, y=y, split=sp, grain_id=gid, classes=np.array(CLASSES))
    print(f"{len(keep)}/{len(rows)} kernels -> {OUT.relative_to(ROOT)}  {X.nbytes / 1e6:.0f} MB")
    for s in ("train", "val", "test"):
        print(s, {c: int(((sp == s) & (y == k)).sum()) for k, c in enumerate(CLASSES)})


if __name__ == "__main__":
    main()
