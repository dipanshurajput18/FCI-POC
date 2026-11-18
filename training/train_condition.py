"""Train the rice condition classifier (sound / chalky / damaged / discoloured) on CPU.

Model: torchvision MobileNetV3-Large (BSD-3 code, ImageNet weights), fine-tuned at 128 px.
Data:  data/cache/rice_condition_128.npz (training/build_cache.py), IIT Indore CC BY 4.0.

Camera gap: the dataset is ~950 px/mm, the tray camera is ~12-40 px/mm. Training blurs
and down/up-samples crops so the model also works at camera resolution; the test report
includes accuracy at simulated 24 and 12 px/mm.

Usage (from AI-anaylser/):
    .venv\\Scripts\\python -m training.train_condition --bench        # time one batch
    .venv\\Scripts\\python -m training.train_condition --epochs 14
Outputs: models/rice_condition_v1.pt / .onnx / .json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "rice_condition_128.npz"
MODELS = ROOT / "models"
NAME = "rice_condition_v1"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# Effective resolution of the cached crops: 1/8 decode of ~950 px/mm, then a ~5 mm kernel
# scaled to 112 px  ->  about 22 px/mm. Low-res factors below are relative to that.
torch.manual_seed(0)
np.random.seed(0)


def to_tensor(x_u8_bgr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x_u8_bgr[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
    return t


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


def augment(t: torch.Tensor) -> torch.Tensor:
    b = t.shape[0]
    # flips / 180 degree turn (kernel orientation is arbitrary)
    if np.random.rand() < 0.5:
        t = t.flip(-1)
    if np.random.rand() < 0.5:
        t = t.flip(-2)
    # small rotation + scale
    ang = (torch.rand(b) - 0.5) * 2 * math.radians(10)
    sc = 0.9 + 0.15 * torch.rand(b)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, (torch.rand(b) - .5) * .08], 1),
                         torch.stack([sin, cos, (torch.rand(b) - .5) * .08], 1)], 1)
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    t = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # camera resolution: 70% of batches at 0.2-1.0 of cached resolution (~4-22 px/mm)
    if np.random.rand() < 0.7:
        t = lowres(t, float(np.random.uniform(0.2, 1.0)))
    # lighting / white balance / contrast
    gain = 1 + (torch.rand(b, 3, 1, 1) - 0.5) * 0.16          # per-channel +-8%
    bright = 1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.4         # +-20%
    t = t * gain * bright
    mean = t.mean(dim=(1, 2, 3), keepdim=True)
    t = (t - mean) * (1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.4) + mean
    t = t + torch.randn_like(t) * 0.01 * np.random.rand()
    return t.clamp(0, 1)


def build_model(n: int, pretrained: bool = True) -> nn.Module:
    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    return m


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, f: float = 1.0, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        t = lowres(to_tensor(X[i:i + bs]), f)
        out.append(F.softmax(model(normalise(t)), 1).numpy())
    return np.concatenate(out)


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
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    d = np.load(CACHE)
    X, y, split = d["X"], d["y"], d["split"]
    classes = [str(c) for c in d["classes"]]
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  classes {classes}", flush=True)

    model = build_model(len(classes))
    counts = np.bincount(y[tr], minlength=len(classes))
    w = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32)
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    if args.bench:
        model.train()
        idx = tr[:args.bs]
        t0 = time.time()
        for _ in range(3):
            xb = normalise(augment(to_tensor(X[idx])))
            loss = crit(model(xb), torch.from_numpy(y[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        dt = (time.time() - t0) / 3
        print(f"{dt:.2f} s/batch -> ~{dt * len(tr) / args.bs / 60:.1f} min/epoch", flush=True)
        return

    MODELS.mkdir(exist_ok=True)
    best, hist = -1.0, []
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = normalise(augment(to_tensor(X[idx])))
            loss = crit(model(xb), torch.from_numpy(y[idx]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss); nb += 1
        pv = predict(model, X[va]).argmax(1)
        pv_lo = predict(model, X[va], 0.5).argmax(1)
        mv, mlo = metrics(y[va], pv, classes), metrics(y[va], pv_lo, classes)
        score = (mv["macro_f1"] + mlo["macro_f1"]) / 2
        hist.append({"epoch": ep + 1, "loss": tot / nb, "val_f1": mv["macro_f1"], "val_f1_12ppm": mlo["macro_f1"],
                     "minutes": (time.time() - t0) / 60})
        tag = ""
        if score > best:
            best = score
            torch.save(model.state_dict(), MODELS / f"{NAME}.pt")
            tag = "  * saved"
        print(f"epoch {ep + 1:2d}/{args.epochs}  loss {tot / nb:.3f}  val acc {mv['accuracy']:.3f} "
              f"F1 {mv['macro_f1']:.3f} | at 12 px/mm F1 {mlo['macro_f1']:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min){tag}", flush=True)

    # ---- test on the held-out split with the best checkpoint
    model.load_state_dict(torch.load(MODELS / f"{NAME}.pt"))
    test = {}
    for label, f in (("full_res_22ppm", 1.0), ("camera_12ppm", 0.55), ("camera_6ppm", 0.27)):
        p = predict(model, X[te], f).argmax(1)
        test[label] = metrics(y[te], p, classes)
        # FCI Q1 default: discoloured (fungal) counted as damaged
        m3 = np.array([0, 1, 2, 2])
        test[label]["fci_q1_default_accuracy"] = float((m3[p] == m3[y[te]]).mean())
        print(f"TEST {label:15s} acc {test[label]['accuracy']:.3f}  macroF1 {test[label]['macro_f1']:.3f}  "
              f"(discoloured->damaged acc {test[label]['fci_q1_default_accuracy']:.3f})", flush=True)

    # ---- ONNX export + parity check
    model.eval()
    dummy = torch.zeros(1, 3, 128, 128)
    onnx_path = MODELS / f"{NAME}.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xb = normalise(to_tensor(X[te[:32]])).numpy()
    diff = float(np.abs(sess.run(None, {"input": xb})[0] - model(torch.from_numpy(xb)).detach().numpy()).max())
    print(f"ONNX exported, max |torch - onnx| = {diff:.2e}", flush=True)

    meta = {"name": NAME, "task": "rice kernel condition (brokenness is decided by the FCI length rule, not this model)",
            "classes": classes, "fci_mapping": "specs/label_map_iit_indore.yaml (discoloured -> damaged pending FCI Q1)",
            "input": {"size": 128, "crop": "grain.crop.canonical_crop", "channel_order": "RGB",
                      "scale": "1/255", "mean": MEAN, "std": STD},
            "architecture": "torchvision MobileNetV3-Large, ImageNet IMAGENET1K_V2 init",
            "data": {"dataset": "IIT Indore rice benchmark, BPT5204 segmented images", "doi": "10.5281/zenodo.22043091",
                     "licence": "CC BY 4.0", "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))},
            "training": {"epochs": args.epochs, "batch": args.bs, "lr": args.lr, "history": hist},
            "test": test, "onnx_max_abs_diff": diff}
    json.dump(meta, open(MODELS / f"{NAME}.json", "w"), indent=2)
    print(f"wrote models/{NAME}.pt / .onnx / .json", flush=True)


if __name__ == "__main__":
    main()
