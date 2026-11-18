"""Contact sheet: random kernels per IIT Indore class, each labelled with its FCI class.

Usage (from the repo root, after training/relabel_iit.py):
    python scripts/make_contact_sheet.py
Output: data/public/iit_indore/preview/all_classes_fci_sheet.jpg
"""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IIT = ROOT / "data" / "public" / "iit_indore"
GROUPS = [("Healthy", None), ("Broken", None), ("Full chalky", None), ("Partial chalky", "chalky"),
          ("Partial chalky", "sound"), ("Peck damage", None), ("Discolored", None)]
N, T = 6, 220


def tile(r: dict) -> np.ndarray:
    img = cv2.imread(str(IIT / "extracted" / r["image"]), cv2.IMREAD_REDUCED_COLOR_8)
    ys, xs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 12)
    c = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = (T - 30) / max(c.shape[:2])
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = np.zeros((T, T, 3), np.uint8)
    y0, x0 = (T - 30 - c.shape[0]) // 2, (T - c.shape[1]) // 2
    t[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
    txt = f"{r['fci_class']}  {float(r['length_mm']):.1f}mm"
    if r["chalky_fraction"]:
        txt += f" milky {float(r['chalky_fraction']):.2f}"
    cv2.putText(t, txt, (4, T - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main() -> None:
    rows = [r for r in csv.DictReader(open(IIT / "fci_labels.csv", encoding="utf-8")) if r["image_downloaded"] == "True"]
    rng = np.random.default_rng(3)
    out = []
    for src, fci in GROUPS:
        cand = [r for r in rows if r["source_class"] == src and (fci is None or r["fci_class"] == fci)]
        if not cand:
            continue
        pick = [cand[i] for i in rng.choice(len(cand), min(N, len(cand)), replace=False)]
        tiles = [tile(r) for r in pick] + [np.zeros((T, T, 3), np.uint8)] * (N - len(pick))
        head = np.full((T, 260, 3), 40, np.uint8)
        cv2.putText(head, src, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(head, "-> FCI: " + (fci or pick[0]["fci_class"]), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 220, 255), 1, cv2.LINE_AA)
        out.append(np.hstack([head] + tiles))
    dst = IIT / "preview" / "all_classes_fci_sheet.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(dst)


if __name__ == "__main__":
    main()
