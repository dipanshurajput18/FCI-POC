"""Train the rice condition classifier (sound / chalky / damaged / discoloured) on CPU.

Model: torchvision MobileNetV3-Large (BSD-3 code, ImageNet weights), fine-tuned at 128 px.
Data:  data/cache/rice_condition_128.npz (training/build_cache.py), IIT Indore CC BY 4.0.

Camera gap: the dataset is ~950 px/mm and noise-free; the tray camera is ~10-25 px/mm with
sensor noise and JPEG compression. v1 trained on clean down-sampled crops and collapsed on
realistic trays (tests/realtray.py). v2 (default) runs most training crops through
training/camera_aug.py, which reproduces the camera chain and the app's own crop code.
Model selection and the test report use camera-degraded images.

Usage (from the repo root):
    python -m training.train_condition --bench                      # time one batch
    python -m training.train_condition --epochs 14                  # -> models/rice_condition_v2.*
    python -m training.train_condition --epochs 14 --resume         # after an interruption
    python -m training.train_condition --eval-only                  # test + export the saved best
    python -m training.train_condition --name rice_condition_v1 --no-camera-aug   # v1 recipe
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from training.camera_aug import camera_degrade

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "rice_condition_128.npz"
LABELS = ROOT / "data" / "public" / "iit_indore" / "fci_labels.csv"
MODELS = ROOT / "models"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CAMERA_P = 0.85            # share of training crops passed through the camera simulation
# Fixed camera conditions used for validation / test (a typical phone or USB camera on the rig)
EVAL_CAMERAS = {
    "camera_18ppm": dict(ppm=18, noise=2.0, jpeg=90, blur=0.5, tray=18),
    "camera_12ppm": dict(ppm=12, noise=2.0, jpeg=90, blur=0.6, tray=18),
    "camera_12ppm_noisy": dict(ppm=12, noise=4.0, jpeg=70, blur=0.8, tray=25),
    "camera_8ppm": dict(ppm=8, noise=2.0, jpeg=90, blur=0.6, tray=18),
}
torch.manual_seed(0)
np.random.seed(0)


def to_tensor(x_u8_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_u8_bgr[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0


def normalise(t: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return (t - m) / s


def lowres(t: torch.Tensor, f: float) -> torch.Tensor:
    if f >= 0.999:
        return t
    n = t.shape[-1]
    k = max(8, int(round(n * f)))
    small = F.interpolate(t, size=(k, k), mode="area")
    return F.interpolate(small, size=(n, n), mode="bilinear", align_corners=False)


def augment(t: torch.Tensor, lowres_p: float) -> torch.Tensor:
    b = t.shape[0]
    if np.random.rand() < 0.5:          # kernel orientation is arbitrary
        t = t.flip(-1)
    if np.random.rand() < 0.5:
        t = t.flip(-2)
    ang = (torch.rand(b) - 0.5) * 2 * math.radians(10)
    sc = 0.9 + 0.15 * torch.rand(b)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, (torch.rand(b) - .5) * .08], 1),
                         torch.stack([sin, cos, (torch.rand(b) - .5) * .08], 1)], 1)
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    t = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if np.random.rand() < lowres_p:     # v1 recipe only
        t = lowres(t, float(np.random.uniform(0.2, 1.0)))
    gain = 1 + (torch.rand(b, 3, 1, 1) - 0.5) * 0.16          # white balance +-8%
    bright = 1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.4         # exposure +-20%
    t = t * gain * bright
    mean = t.mean(dim=(1, 2, 3), keepdim=True)
    t = (t - mean) * (1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.4) + mean
    return t.clamp(0, 1)


def build_model(n: int, pretrained: bool = True) -> nn.Module:
    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    return m


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(F.softmax(model(normalise(to_tensor(X[i:i + bs]))), 1).numpy())
    return np.concatenate(out)


def degrade_set(X: np.ndarray, lengths: np.ndarray, cam: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([camera_degrade(x, L, rng, **cam) for x, L in zip(X, lengths)])


def metrics(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"accuracy": float(cm.diagonal().sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "per_class": {c: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                              "f1": round(float(f1[i]), 4), "n": int(cm[i].sum())} for i, c in enumerate(classes)},
            "confusion": cm.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rice_condition_v2")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-camera-aug", action="store_true", help="v1 recipe (clean down-sampling only)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from models/<name>_state.pt")
    ap.add_argument("--eval-only", action="store_true", help="skip training: test + export the saved best")
    args = ap.parse_args()
    NAME, cam_aug = args.name, not args.no_camera_aug

    d = np.load(CACHE)
    X, y, split, gid = d["X"], d["y"], d["split"], d["grain_id"]
    classes = [str(c) for c in d["classes"]]
    lmap = {r["grain_id"]: float(r["length_mm"]) for r in csv.DictReader(open(LABELS, encoding="utf-8"))}
    lengths = np.array([lmap.get(g, 5.0) for g in gid], np.float32)
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  classes {classes}  camera_aug {cam_aug}", flush=True)

    model = build_model(len(classes))
    counts = np.bincount(y[tr], minlength=len(classes))
    w = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w / w.mean(), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(1)

    def batch(idx: np.ndarray) -> torch.Tensor:
        xb = X[idx]
        if cam_aug:
            xb = np.stack([camera_degrade(x, L, rng) if rng.random() < CAMERA_P else x
                           for x, L in zip(xb, lengths[idx])])
        return normalise(augment(to_tensor(xb), 0.0 if cam_aug else 0.7))

    if args.bench:
        model.train()
        idx = tr[:args.bs]
        t0 = time.time()
        for _ in range(3):
            loss = crit(model(batch(idx)), torch.from_numpy(y[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        dt = (time.time() - t0) / 3
        print(f"{dt:.2f} s/batch -> ~{dt * len(tr) / args.bs / 60:.1f} min/epoch", flush=True)
        return

    MODELS.mkdir(exist_ok=True)
    Xva_cam = degrade_set(X[va], lengths[va], EVAL_CAMERAS["camera_12ppm"], seed=7)
    state_path = MODELS / f"{NAME}_state.pt"
    best, hist, start = -1.0, [], 0
    if args.resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, hist, start = st["best"], st["hist"], st["epoch"]
        np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"]); rng.bit_generator.state = st["rng"]
        print(f"resumed after epoch {start} (best score {best:.3f})", flush=True)
    for ep in range(start, 0 if args.eval_only else args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            loss = crit(model(batch(idx)), torch.from_numpy(y[idx]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        mv = metrics(y[va], predict(model, X[va]).argmax(1), classes)
        mc = metrics(y[va], predict(model, Xva_cam).argmax(1), classes)
        score = (mv["macro_f1"] + 2 * mc["macro_f1"]) / 3        # the camera is what matters
        hist.append({"epoch": ep + 1, "loss": tot / nb, "val_f1": mv["macro_f1"], "val_f1_camera12": mc["macro_f1"],
                     "minutes": (time.time() - t0) / 60})
        tag = ""
        if score > best:
            best = score
            torch.save(model.state_dict(), MODELS / f"{NAME}.pt")
            tag = "  * saved"
        print(f"epoch {ep + 1:2d}/{args.epochs}  loss {tot / nb:.3f}  val acc {mv['accuracy']:.3f} "
              f"F1 {mv['macro_f1']:.3f} | camera 12 px/mm acc {mc['accuracy']:.3f} F1 {mc['macro_f1']:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min){tag}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "hist": hist, "epoch": ep + 1, "rng": rng.bit_generator.state,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, state_path)

    # ---- test on the held-out split with the best checkpoint
    model.load_state_dict(torch.load(MODELS / f"{NAME}.pt"))
    test = {}
    m3 = np.array([0, 1, 2, 2])                     # FCI Q1 default: discoloured -> damaged
    sets = {"dataset_clean": X[te]}
    sets.update({k: degrade_set(X[te], lengths[te], cam, seed=11) for k, cam in EVAL_CAMERAS.items()})
    for label, Xs in sets.items():
        p = predict(model, Xs).argmax(1)
        test[label] = metrics(y[te], p, classes)
        test[label]["fci_q1_default_accuracy"] = float((m3[p] == m3[y[te]]).mean())
        print(f"TEST {label:19s} acc {test[label]['accuracy']:.3f}  macroF1 {test[label]['macro_f1']:.3f}  "
              f"(discoloured->damaged acc {test[label]['fci_q1_default_accuracy']:.3f})", flush=True)

    # ---- ONNX export + parity check
    model.eval()
    onnx_path = MODELS / f"{NAME}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 128, 128), str(onnx_path), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xb = normalise(to_tensor(X[te[:32]])).numpy()
    diff = float(np.abs(sess.run(None, {"input": xb})[0] - model(torch.from_numpy(xb)).detach().numpy()).max())
    print(f"ONNX exported, max |torch - onnx| = {diff:.2e}", flush=True)

    meta = {"name": NAME, "task": "rice kernel condition (brokenness is decided by the FCI length rule, not this model)",
            "classes": classes, "fci_mapping": "specs/rice_kms_2026_27.yaml condition_model.class_map",
            "input": {"size": 128, "crop": "grain.crop.canonical_crop", "channel_order": "RGB",
                      "scale": "1/255", "mean": MEAN, "std": STD},
            "architecture": "torchvision MobileNetV3-Large, ImageNet IMAGENET1K_V2 init",
            "data": {"dataset": "IIT Indore rice benchmark, BPT5204 segmented images", "doi": "10.5281/zenodo.22043091",
                     "licence": "CC BY 4.0", "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))},
            "training": {"epochs": args.epochs, "batch": args.bs, "lr": args.lr, "camera_aug": cam_aug,
                         "camera_aug_share": CAMERA_P if cam_aug else 0, "history": hist},
            "eval_cameras": EVAL_CAMERAS, "test": test, "onnx_max_abs_diff": diff}
    json.dump(meta, open(MODELS / f"{NAME}.json", "w"), indent=2)
    print(f"wrote models/{NAME}.pt / .onnx / .json", flush=True)


if __name__ == "__main__":
    main()
