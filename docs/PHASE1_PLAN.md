# AI Grain Analyser: Phase 1 plan (rice + sugar, live video)

Status: draft for review · 17 Nov 2026 · Phase 2 (MoistureLink, hardware) is out of scope here.

---

## 1. Problem in one paragraph

FCI accepts or rejects each lot of rice (about 90 million tonnes a year) by hand. An officer picks through a 20 g sample with forceps and a magnifier, counts the broken, chalky, red, discoloured and damaged grains, and types the % into the ERP. Different officers get different results, the process can be influenced, no image evidence is kept, and the manual typing gives errors. FCI wants a scientific device instead: the same sample gives the same result at every depot, every grain is recorded as an image, and the result goes directly to the ERP.

**Lesson from the field.** FCI already deployed AI analysers in Punjab in 2024. Millers disputed the results, the models had to be retrained on more regions, and manual checks came back for borderline lots. So our system has to win on **accuracy on Indian samples, explainability (why each kernel was rejected) and evidence**, not just a nice UI.

## 2. What Phase 1 delivers

A working prototype where a **camera streams live video of a tray**. Every kernel or crystal is outlined and colour-coded on screen in real time. When the sample is still, the system runs a full analysis and produces:

- % by weight per quality parameter, compared with the official limits → **Accept / Accept with value cut / Reject**
- a kernel inspector (click any grain to see why it was classified that way)
- a signed digital report (SHA-256 hash and QR code) with annotated images
- a posting to a simulated FCI ERP (same flow as the existing demo console)

Commodities: **Rice** (raw and parboiled, Grade A / Common / the new "Improved Rice") and **Sugar** (plantation white, IS 498 grades).

## 3. Specifications we measure against

### 3.1 Rice: Uniform Specification KMS 2026-27 (DFPD), method IS 4333 (Part 1)

| Parameter | Raw max % | Parboiled max % | Definition | From camera? |
|---|---|---|---|---|
| Brokens | 25 (Improved Rice: 10) | 16 (Improved: 5) | Piece < 3/4 of mean whole-kernel length. Small brokens ≤ 1%. Piece < 1/8 counts as foreign matter. | **Yes** (length) |
| Foreign matter | 0.5 (mineral ≤ 0.2, animal ≤ 0.1) | 0.5 | Stones, mud, husk, chaff | **Partly** (visible items yes; tiny dust no) |
| Damaged / slightly damaged | 3 | 4 | Spots, pin-point damage, insect damage | **Partly** (needs ≥ 25 px/mm) |
| Discoloured | 3 | 5 | Yellow or brown tint | **Yes** (colour) |
| Chalky | 5 | n/a | ≥ ½ of the grain milky white | **Yes, needs backlight** |
| Red grains | 3 | 3 | Red pericarp | **Yes** (colour) |
| Admixture of lower class | 6 | 6 | Different length/breadth class | **Partly** (L/B ratio) |
| De-husked grains | 13 | 13 | > ¼ of surface still covered by bran | No (needs staining; Phase 1 gives indicative only) |
| Moisture | 14 (to 15 with cut) | 14 | | **No** (moisture meter, Phase 2) |

Rules taken from the official method:
- **Weight basis.** All parameters except de-husked are % by **weight**, so the image count must be converted to weight (see 5.4).
- **One category per grain, in the official picking order:** foreign matter → damaged → discoloured → broken → chalky → red → admixture. The classifier output follows this precedence.
- Limits live in a **config file per season** (`specs/*.yaml`), so a new circular never needs a code change.

### 3.2 Sugar: IS 498:2003 (grading) + IS 5982 (specification)

The grade name is size letter + colour number, e.g. **M-30**.

| Size group | ≥ 70% retained on | ≥ 95% on | ≥ 99% on |
|---|---|---|---|
| L | 1.70 mm | 0.85 mm | 0.60 mm |
| M | 1.18 mm | 0.60 mm | 0.425 mm |
| S | 0.60 mm | 0.30 mm | 0.212 mm |
| SS | 0.212 mm | 0.175 mm | 0.147 mm |

| Sugar parameter | From camera? | Phase 1 approach |
|---|---|---|
| Crystal size distribution → L/M/S/SS | **Yes** | Measure every crystal → equivalent sieve aperture → virtual sieve % retained |
| Colour grade 31/30/29 (officially by eye against NSI Kanpur reference bottles) | **Yes, as a proxy** | Calibrated whiteness (CIE L\*, b\*) against a grey card, mapped to grade using reference samples |
| Black specks, foreign particles, lumps | **Yes** | Dark-blob and size-outlier detection |
| ICUMSA colour, moisture, SO₂, ash, polarisation, reducing sugars | **No** | Lab tests; out of scope, shown as "lab" in the report |

> ⚠ **Open question.** We found no current FCI role in sugar: the levy was abolished in 2013 and states now buy PDS sugar on the open market. The EoI in the deck names rice, wheat and paddy only. Confirm who asked for sugar before we spend effort tuning it.

## 4. Physical set-up (Phase 1 rig, off-the-shelf, about ₹8–15k)

Chalkiness **cannot** be judged reliably from top light alone. Commercial analysers (e.g. Kett RN-700) use **light from below**. So:

| Item | Spec | Approx. cost |
|---|---|---|
| USB camera | 4K / 12 MP, **manual focus + manual exposure + manual white balance**, UVC (e.g. Logitech Brio 4K or an Arducam/ELP 12 MP USB board camera with an M12 lens) | ₹4–9k |
| Copy stand / phone stand | Rigid, camera 20–30 cm above the tray, perpendicular | ₹1–2k |
| Backlight | A4 LED tracing/light pad (flicker-free, 5000–6500 K) under a frosted acrylic sheet | ₹1–2k |
| Top light | Diffuse LED ring or two LED panels at 45° (for colour defects) | ₹1–2k |
| Tray background | Matte black card (used when the backlight is off) | ₹100 |
| Calibration card | Printed ArUco markers (gives mm per pixel and corrects tilt) + 18% grey patch (white balance) | print |
| Balance | 0.01 g pocket scale, to weigh the sample (needed for weight %) | ₹1k |

Resolution targets:
- **Rice:** ≥ 20–25 px/mm, so a 100 × 70 mm field needs about 2500 × 1750 px (a 4K camera covers it).
- **Sugar:** ≥ 50 px/mm, so use a smaller field (about 50 × 35 mm) or a macro lens.

Sample handling: about **20 g of rice spread in a single layer** (as in the official 20 g method). Sugar: a small spoonful spread thinly, with several frames per sample.

The laptop has no GPU and no camera. **Inference runs on the CPU** (ONNX Runtime). **Training runs on a free cloud GPU** (Kaggle or Colab).

## 5. Software architecture

```
 USB camera ──► Capture service (OpenCV)
                  │  live frames (~720p, 10–15 fps)            full-res still when tray is steady
                  ▼                                              ▼
           Live overlay engine (fast)                 Full analysis pipeline (accurate)
           segmentation + quick colour tags            calibrate → segment → measure →
           ~5–10 fps on CPU                            classify → weight % → spec rules
                  │                                              │
                  └────────────── FastAPI server ────────────────┘
                         WebSocket (overlay frames + live stats) · REST (analyse, report, ERP)
                                         ▼
                Browser UI (same look as the FCI Demo Console)
                live video + outlines · parameter table · verdict · kernel inspector
                report with QR + hash · "Post to ERP" · simulated ERP inbox
```

### 5.1 Video handling
- **Live mode:** downscaled frames → fast segmentation → coloured outlines and a running estimate. This is what the officer and the audience see.
- **Stability gate:** when frame-to-frame motion stays below a threshold for about 1 s, the system grabs a **full-resolution still** and runs the full analysis. Results freeze until the tray moves again. This avoids double-counting when the tray is shaken.
- **Multi-shot:** the officer can shake or re-spread the tray and capture 2–5 shots. Results are aggregated, which gives larger counts and both faces of the grains (a cheap substitute for "360° imaging").

### 5.2 Calibration (each session)
ArUco markers → homography (removes tilt) and **mm per pixel**. Grey patch → white balance and exposure lock. Without this, length and colour measurements are not trustworthy.

### 5.3 Pipeline stages

| # | Stage | Phase 1A (classical, works day 1) | Phase 1B (trained AI) |
|---|---|---|---|
| 1 | Segmentation | Backlight threshold (Otsu) + distance transform + **watershed** to split touching kernels | Instance segmentation model (**RTMDet-Ins**, MMDetection, Apache-2.0); classical result used to bootstrap labels |
| 2 | Morphometry | `minAreaRect` / ellipse fit → length, breadth, L/B, area in mm | same |
| 3 | Classification | Rules: length < ¾ of the median whole kernel → broken; backlight opacity → chalky; HSV/Lab colour → red / discoloured; dark blobs → damaged | Per-kernel CNN (**EfficientNet-B0 / MobileNetV3**, torchvision BSD) on a two-channel crop (backlit + top-lit) |
| 4 | Weight % | mass ≈ k · area × breadth (rice) or area^1.5 (sugar); k scaled so the total equals the weighed sample | same, k calibrated per variety |
| 5 | Spec rules | YAML spec + picking-order precedence → within / value cut / reject | same |
| 6 | Evidence | Annotated image, JSON result, SHA-256, QR, ERP payload | + signing key, model version in the record |

**Low-confidence kernels** are flagged in yellow for the officer to confirm. Each correction is saved as a new training label (human-in-the-loop).

**Licensing rule.** Use **no Ultralytics YOLOv8/YOLO11** (AGPL; needs a paid licence) and **no YOLO-NAS weights** (non-commercial). Use only Apache-2.0 / MIT / BSD components, matching the promise in the deck.

### 5.4 Sugar specifics
- Segment crystals: watershed plus a convexity split for clusters, as in published massecuite methods.
- Crystal "sieve size" = the minor axis of its min-area rectangle (the dimension that decides whether it passes a square mesh), cross-calibrated with real sieves.
- Weighted cumulative distribution → % retained at each IS 498 aperture → size group.
- Report the mean aperture (MA) and coefficient of variation (CV), as sugar labs do.

## 6. Data plan

**Public data** is only a starting point:

| Dataset | Use | Licence |
|---|---|---|
| RiceCC (Hugging Face, ~21k images: whole/broken + chalkiness subset) | Pretrain the broken/chalky classifier | MIT tag, but the paper is NC. **Verify.** |
| Roboflow `bits/rice_object_detection` and similar (broken, chalky, damaged, discoloured, immature, FM, red) | Classes close to FCI's; detection pretraining | CC BY 4.0 (unverified) |
| GrainSet rice (31k single kernels with defect labels) | Classifier pretraining | Unknown. **Verify.** |
| Kaggle Koklu Rice (75k single kernels, black background) | Segmentation / shape pretraining | CC0 |
| Sugar | **None exist.** Must collect our own. | n/a |

**No public Indian quality-labelled rice dataset exists. Our own data is the core asset.**

### Own data collection
- Rice: about 30 samples covering raw and parboiled, 3–5 Indian varieties (e.g. common/IR-64 type, Sona Masuri, PR-126, a parboiled lot).
- To create defect classes on demand, make **spiked samples**: start from clean rice and add known counts of brokens, chalky grains (picked from a lower-grade lot), red grains, etc. This gives ground truth without needing an FCI analyst.
- Sugar: buy L, M and S grade sugar (retail S-30/M-30 packs, bakery fine sugar, coarse sugar). **Sieve** each one (₹2–3k for a test-sieve set at 1.70, 1.18, 0.85, 0.60, 0.425 and 0.30 mm) to create ground truth.
- Labelling: CVAT or Label Studio. Auto-label with the classical pipeline + SAM2 masks, then correct by hand. Target ≥ 5,000 labelled rice kernels for 1B.
- Each image saves its metadata: variety, lighting, camera settings, px/mm, and the true counts or weights.

## 7. Accuracy targets and validation

| Metric | Target for Phase 1 demo |
|---|---|
| Kernel count vs. hand count | ± 2% |
| Length measurement vs. vernier/micrometer | ± 0.15 mm |
| Broken % (weight) vs. manual picking on spiked samples | ± 2 percentage points |
| Chalky / red / discoloured classification | ≥ 90% per-class recall on held-out samples |
| Sugar size group (L/M/S) vs. sieve | Correct group on ≥ 9/10 samples; MA within 10% |
| Repeatability (same sample, 5 re-spreads) | Std-dev of each % ≤ 1 point |
| Speed | Live overlay ≥ 5 fps; full analysis < 5 s on the i3 laptop |

Every model release comes with a validation report (confusion matrix, per-class accuracy, test samples used). This is what answers the Punjab-style disputes.

## 8. Build sequence

| Step | Output | Rough effort |
|---|---|---|
| **M0 Skeleton** | Repo layout, FastAPI server, camera capture, browser page with live video over WebSocket, spec YAML loader | 2–3 days |
| **M1 Classical rice, live** | ArUco calibration, watershed segmentation, morphometry, rule classifier, live outlines, stability gate, parameter table + verdict, kernel inspector. Works with a webcam or a video file / still images (for testing without a camera) | 1–1.5 weeks |
| **M2 Evidence & ERP** | Report page (annotated image, QR, hash), multi-shot aggregation, simulated ERP posting (reuse the demo console flow) | 3–4 days |
| **M3 Sugar** | Crystal segmentation, virtual sieve, size grade, whiteness proxy, specks | 1 week |
| **M4 Data** | Capture tool built into the app (saves images + metadata), labelling set-up, first own dataset | 1–2 weeks, parallel with M3 |
| **M5 Trained models** | Kernel classifier (then instance segmentation) trained on Kaggle/Colab, exported to ONNX, plugged in behind the same interface; officer corrections fed back | 2 weeks |
| **M6 Validation** | Accuracy report on spiked rice + sieved sugar; repeatability; demo script | 3–4 days |

M1 alone is already a convincing live demo. M5 turns it into a credible product.

### Proposed folder layout
```
AI-anaylser/
  server/            FastAPI app, camera service, websocket
  vision/            calibration, segmentation, morphometry, classify, sugar, weight
  specs/             rice_kms_2026_27.yaml, sugar_is498.yaml
  web/               UI (live view, results, report, ERP inbox)
  models/            ONNX models + model cards
  training/          notebooks/scripts for Kaggle/Colab
  data/              captured samples (not in git), labels
  tests/             unit tests + golden images
```
Tech: Python 3.13, OpenCV, NumPy, FastAPI + Uvicorn, ONNX Runtime (CPU); PyTorch only for training; vanilla JS or React for the UI.

## 9. Risks

| Risk | Mitigation |
|---|---|
| A cheap webcam's auto-exposure, compression and focus make measurements drift | Lock exposure, white balance and focus; calibrate each session; prefer a UVC camera with manual controls |
| Touching or overlapping kernels | Single-layer spreading guide on screen; watershed; trained segmenter later; reject frames with too many clusters |
| Chalky vs. translucent errors | Backlight is mandatory for the chalky parameter; the report shows "chalky not assessed" without it |
| Count → weight conversion error | Weigh the sample; calibrate the factor per variety on known samples |
| Variety drift (the Punjab problem) | Collect data from many varieties; officer correction loop; per-variety calibration profile |
| Sugar crystals too small for the lens | Smaller field of view, or a ₹1–2k clip-on macro lens |
| Dataset licences | Use only CC0 / CC-BY / MIT / own data for the shipped model; keep a licence register |

## 10. Decisions needed from the team

1. **Camera:** which camera will we use? A 4K UVC camera with manual controls is recommended. Until it arrives we develop on recorded video and photos.
2. **Samples:** can we get Indian rice samples (including lower-grade lots with chalky/red/broken) and three sugar grades + a sieve set?
3. **Sugar scope:** confirm that FCI (or another client) wants sugar, since FCI's EoI lists rice/wheat/paddy.
4. **Spec baseline:** use **KMS 2026-27 Uniform Specs** (incl. the new "Improved Rice" 10% brokens column) as default? *Proposed: yes, selectable per lot.*
5. **UI:** build on the look of the existing *FCI Demo Console* (proposed), or a fresh design?
