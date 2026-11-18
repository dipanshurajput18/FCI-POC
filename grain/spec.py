"""Load season specifications (YAML) and evaluate parameters against limits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

SPEC_DIR = Path(__file__).resolve().parent.parent / "specs"


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    with open(SPEC_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rice_spec() -> dict:
    return load("rice_kms_2026_27.yaml")


def sugar_spec() -> dict:
    return load("sugar_is498.yaml")


def evaluate_rice(class_mass: dict[str, float], class_count: dict[str, int], grade: str) -> tuple[list[dict], dict]:
    spec = rice_spec()
    limits = spec["grades"][grade]
    tot_m = sum(class_mass.values()) or 1.0
    tot_n = sum(class_count.values()) or 1
    rows = []
    for p in spec["parameters"]:
        cls = p["classes"]
        m = sum(class_mass.get(c, 0.0) for c in cls)
        n = sum(class_count.get(c, 0) for c in cls)
        wt = 100.0 * m / tot_m
        lim = limits.get(p["key"])
        status = "na" if lim is None else ("within" if wt <= lim else "beyond")
        rows.append({"key": p["key"], "name": p["name"], "count": n, "count_pct": round(100.0 * n / tot_n, 2),
                     "wt_pct": round(wt, 2), "limit": lim, "status": status})
    beyond = [r for r in rows if r["status"] == "beyond"]
    if beyond:
        verdict = {"code": "BEYOND", "text": "Beyond specification - reject or refer for relaxation",
                   "params": [r["name"].strip() for r in beyond]}
    else:
        verdict = {"code": "WITHIN", "text": "Within specification - accept", "params": []}
    return rows, verdict
