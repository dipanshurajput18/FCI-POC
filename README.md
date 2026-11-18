# FCI POC: AI Grain Analyser (Phase 1) and MoistureLink (Phase 2)

This is a proof of concept for the Food Corporation of India.

**Phase 1: AI Grain Analyser** (rice and sugar, live camera; this code). A camera watches a tray of rice or sugar. Every grain or crystal is outlined live by class. When the tray is still, a full-resolution analysis gives:
- % by weight per FCI parameter, or the IS 498 grade for sugar
- a verdict
- a clickable explanation for every grain

**Phase 2: MoistureLink** (connected moisture meters). This is planned; see [phase2_moisturelink/](phase2_moisturelink/).

| Document | Purpose |
|---|---|
| [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md) | Phase 1 plan, rig, architecture, milestones |
| [docs/FCI_REQUIREMENTS_TRACE.md](docs/FCI_REQUIREMENTS_TRACE.md) | Every FCI requirement → where it is met; open questions for FCI |
| [docs/DATASETS.md](docs/DATASETS.md) | Datasets surveyed, with licences |
| [data/README.md](data/README.md) | How to download and prepare the training data |
| [models/README.md](models/README.md) | Trained models and their test results |

## Project structure

```
grain/                 vision core (pure Python + OpenCV, no web code)
  calibration.py         ArUco markers -> px/mm, calibration sheet
  segment.py             Otsu + watershed segmentation, per-object measurements
  crop.py                canonical kernel crop shared by training and the live app
  rice.py / sugar.py     FCI KMS rice rules / IS 498 sugar rules, weight-% maths
  spec.py, pipeline.py   spec loading, end-to-end analyse + draw
  synth.py               simulated trays with ground truth
server/                FastAPI app: camera sources, live engine, REST + WebSocket
web/                   browser UI (vanilla HTML/CSS/JS)
specs/                 FCI / BIS limits and dataset label maps (YAML: edit these, not code)
training/              relabel IIT Indore -> FCI classes, crop cache, CNN training
scripts/               dataset download (SHA-256 verified), contact sheets
tests/                 pytest: simulated-tray accuracy + FCI relabel rules
models/                exported ONNX models + model cards (checkpoints not committed)
data/                  datasets and caches (not committed; see data/README.md)
docs/                  plan, FCI requirement trace, dataset survey
phase2_moisturelink/   Phase 2 placeholder
```

## Run

```powershell
git clone https://github.com/dipanshurajput18/FCI-POC.git
cd FCI-POC
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn server.app:app --port 8000
```

Open http://localhost:8000. It starts on a **simulated tray**, so no camera is needed. Press **Shake tray** to re-spread the sample. The tray settles, a new shot is analysed automatically, and the shots add up into one sample.

### With a real camera

1. Print the calibration sheet, `http://localhost:8000/api/calibration-sheet.png`, at **100% scale on A4**. It has four 20 mm ArUco markers, an 18% grey patch, and a matte-black sample area.
2. Spread about 20 g of rice in a **single layer** on the black area. Mount the camera 20–30 cm above it, looking straight down. Use diffuse light with no hard shadows.
3. Choose the source:
   - **USB camera:** pick the camera number; *Find cameras* lists them.
   - **Phone:** install an IP-camera app (e.g. Android "IP Webcam"), start its server, and enter `http://<phone-ip>:8080/video`. The phone and PC must be on the same Wi-Fi.
   - **Video file / photo:** for offline testing.
4. The calibration line must show *"from 4 marker(s) ✓"*. If markers are not found, sizes fall back to a manual px/mm and are not trustworthy.
5. For sugar, use a closer view (≥ 40 px/mm) and print the markers smaller (set *marker size* to match).

## How it works

| Stage | File | Method |
|---|---|---|
| Calibration | `grain/calibration.py` | ArUco markers → px/mm; marker areas excluded |
| Segmentation | `grain/segment.py` | Otsu threshold + distance-transform watershed (splits touching grains) |
| Measurement | `grain/segment.py` | min-area rectangle (length, breadth), Lab colour, opaque / red / dark-spot fractions |
| Condition model | `grain/condition_model.py` | Trained CNN (ONNX, CPU) judges sound / chalky / damaged / discoloured on each kernel crop (`grain/crop.py`). Used on full-resolution captures from real cameras at ≥ 10 px/mm. Selector in the UI: *Auto / AI model / Rules*. |
| Rice rules | `grain/rice.py` | broken < ¾ and small broken < ¼ of the whole-kernel length; chalky, red, discoloured, damaged, foreign; FCI picking order (one class per kernel); borderline grains flagged for review |
| Sugar rules | `grain/sugar.py` | crystal sieve size = minor dimension → virtual sieve → IS 498 L/M/S/SS; whiteness proxy → 31/30/29; dark inclusions → specks |
| Weight % | `rice.py` / `sugar.py` | volume from size × density (rice 1.45, sucrose 1.587 g/cm³) |
| Limits | `specs/*.yaml` | KMS 2026-27 Uniform Specs (incl. "Improved Rice"); IS 498. **Edit the YAML, not the code**, when a new circular comes. |
| Live engine | `server/engine.py` | live overlay on a 960 px preview; motion gate → full analysis on the still frame; multi-shot aggregation |
| Simulator | `grain/synth.py` | renders trays with known truth: rice presets (FAQ, broken, chalky, red, mixed, foreign) and sugar presets (L-31, M-30, S-30, M-29, specks, mixed) |

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

There are 38 tests:
- 17 for the pipeline on simulated trays
- 15 for the FCI relabelling rules in `tests/test_relabel.py`
- 6 for the model integration in `tests/test_model_integration.py`, covering:
  - the printed A4 sheet in view
  - how the classifier is chosen
  - the model end to end
  - a tray of real IIT test kernels (skipped when the dataset is not downloaded)

`python -m tests.realtray` builds camera-like trays from real IIT Indore test kernels and compares rules against the AI model.

On simulated trays the pipeline reaches:
- rice: 99.5–100% per-kernel class accuracy, with length error under 0.15 mm
- sugar: the correct IS 498 size group, sieve-size error under 0.03 mm, and no false specks on clean samples

**Simulated accuracy is not field accuracy.** Real-image validation against manual FCI analysis is milestone M6.

## Training the rice condition model

The classical pipeline finds each kernel and the FCI length rule decides broken / small broken. A CNN (MobileNetV3-Large, transfer learning) judges each kernel's **condition**: sound / chalky / damaged / discoloured. It is trained on the IIT Indore dataset (CC BY 4.0) after relabelling into FCI classes. CPU only, about 1 hour on a laptop i3.

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python scripts\download_iit_indore.py --delete-zips   # ~17 GB download
.venv\Scripts\python -m training.relabel_iit                         # -> fci_labels.csv
.venv\Scripts\python -m training.build_cache                         # -> 128 px crops
.venv\Scripts\python -m training.train_condition --epochs 14         # -> models/rice_condition_v1.onnx
```

## Known limits (Phase 1A, rule-based)

- **Chalkiness from top light is approximate.** Use a backlit tray for chalky grading.
- **Loose dark particles on a black tray are invisible.** Use a light or backlit background for foreign-matter checks.
- **Thresholds and colour limits are untuned for real cameras.** The thresholds and sugar colour limits are placeholders until calibrated on real samples. Use the *Thresholds* panel to tune them.
- **Not assessed by camera:** moisture, de-husked grains (staining), chemical sugar tests.
- **The AI model needs a black background, a single layer of grains, the camera straight above, and ≥ 10 px/mm.** On white paper, heaped grain or oblique views, results are not valid.
