"""Download, verify (SHA-256) and safely extract the IIT Indore rice dataset (CC BY 4.0).

Zenodo DOI 10.5281/zenodo.22043091 - Sapkal, Anoop K R, Miglani, Kankar (IIT Indore).
Checksums: scripts/iit_indore_checksums.txt (published with the dataset).

Usage (from the repo root):
    python scripts/download_iit_indore.py                 # Metadata + all 6 BPT5204 segmented classes (~17 GB)
    python scripts/download_iit_indore.py --classes Healthy Broken
    python scripts/download_iit_indore.py --delete-zips   # remove each zip once it is extracted

Needs `curl` on PATH (Windows 10+ ships it). Downloads resume after connection resets.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "public" / "iit_indore"
DL, OUT = BASE / "download", BASE / "extracted"
URL = "https://zenodo.org/records/22043091/files/{}?download=1"
ALL = ["Healthy", "Broken", "Full_chalky", "Partial_chalky", "Peck_damage", "Discolored"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name: str, sums: dict, delete: bool) -> bool:
    path = DL / name
    print(f"== {name}", flush=True)
    for _ in range(3):
        subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(path), URL.format(name)])
        if path.exists() and sha256(path) == sums.get(name):
            print(f"   checksum OK  {path.stat().st_size / 1e9:.2f} GB", flush=True)
            break
        print("   checksum mismatch / incomplete - resuming", flush=True)
    else:
        print("   FAILED", flush=True)
        return False
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if any(n.startswith(("/", "\\")) or ".." in n.replace("\\", "/").split("/") for n in names):
            print("   unsafe paths in zip - not extracted", flush=True)
            return False
        zf.extractall(OUT)
    print(f"   extracted {sum(1 for n in names if not n.endswith('/'))} files", flush=True)
    if delete:
        path.unlink()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--delete-zips", action="store_true")
    a = ap.parse_args()
    DL.mkdir(parents=True, exist_ok=True)
    sums = {}
    for line in open(ROOT / "scripts" / "iit_indore_checksums.txt", encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    names = ["Metadata.zip"] + [f"Segmented_image_BPT5204_{c}.zip" for c in a.classes]
    ok = [fetch(n, sums, a.delete_zips) for n in names]
    print("ALL DONE" if all(ok) else "SOME FILES FAILED - re-run to resume", flush=True)


if __name__ == "__main__":
    main()
