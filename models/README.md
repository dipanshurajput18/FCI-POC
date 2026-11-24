# models/

| Model | Status | Real-kernel tray, 12 px/mm with camera noise | Files |
|---|---|---|---|
| **`rice_condition_v2`** | **in use** (`specs/rice_kms_2026_27.yaml`) | **94.9%** (rules alone: 62.9%) | `.onnx` (16.8 MB), `.json` |
| `rice_condition_v1` | superseded: not robust to camera noise | 61% | `.onnx`, `.json` |

PyTorch checkpoints (`*.pt`) are not committed. Rebuild them with the commands in the main README.

---

## Model card: `rice_condition_v2`

**Job.** The model judges the *condition* of one rice kernel: sound, chalky, damaged or discoloured. It does **not** decide broken or small broken. Those come from the FCI length rule (< ¾ and < ¼ of the whole-kernel length) in `grain/rice.py`, because length is measured, not guessed. Foreign matter and red kernels also stay rule-based, because they are not in the training data.

| | |
|---|---|
| Architecture | MobileNetV3-Large (torchvision, BSD-3), ImageNet `IMAGENET1K_V2` initialisation, last layer → 4 classes |
| Input | 128 × 128 RGB crop from `grain.crop.canonical_crop`: kernel upright, soft-edged, centred on black, long side = 112 px. Scale 1/255, ImageNet mean/std. |
| Output | logits for `[sound, chalky, damaged, discoloured]`; FCI mapping in `specs/rice_kms_2026_27.yaml` → `condition_model.class_map` |
| Runtime | ONNX opset 17, CPU (onnxruntime, or OpenCV DNN); about 2 ms per kernel on an i3-1215U |
| Training | 14 epochs, AdamW, one-cycle LR 1e-3, class-weighted cross-entropy + label smoothing 0.05, about 1.2 h on CPU |
| **Camera simulation** | 85% of training crops pass through `training/camera_aug.py`. The chain is: kernel on a dark tray (level 0–35) → down-sample to 7–24 px/mm → blur → sensor noise (σ 0–4) → JPEG (quality 60–95) → cut out again with the app's own crop code. Plus flips, rotation, scale, white balance, exposure and contrast. |
| Model selection | best of (clean F1 + 2 × camera-12 px/mm F1) / 3 on the validation split |

### Data

The data is the IIT Indore rice benchmark, BPT5204 segmented images: DOI [10.5281/zenodo.22043091](https://doi.org/10.5281/zenodo.22043091), **CC BY 4.0**, by Sapkal, Anoop K R, Miglani and Kankar of IIT Indore. It was relabelled into FCI terms by `training/relabel_iit.py` and `specs/label_map_iit_indore.yaml`.

| Model class | Source labels | Train / val / test |
|---|---|---|
| sound | Healthy, Broken (its condition), Partial chalky with < ½ milky white | 3656 / 799 / 814 |
| chalky | Full chalky, Partial chalky with ≥ ½ milky white (FCI definition) | 1977 / 413 / 384 |
| damaged | Peck damage | 1504 / 322 / 302 |
| discoloured | Discolored (fungal) | 2824 / 602 / 631 |

The split follows the official IIT Indore `6class` split where a grain has one; other grains are assigned 70/15/15 per class (seed 42). That gives **9,961 training images**, about 139,000 training views over 14 epochs.

### Results

**1. Held-out test kernels (2,131), single-kernel crops.** Each condition is simulated with a fixed seed:

| Images | Accuracy | Macro-F1 | With FCI Q1 default (discoloured → damaged) |
|---|---|---|---|
| Dataset, clean | 94.3% | 0.943 | 95.1% |
| Camera 18 px/mm, noise σ 2, JPEG 90 | 94.1% | 0.942 | 94.9% |
| **Camera 12 px/mm, noise σ 2, JPEG 90** | **93.5%** | **0.935** | **94.6%** |
| Camera 12 px/mm, noise σ 4, JPEG 70 | 92.7% | 0.928 | 93.7% |
| Camera 8 px/mm, noise σ 2, JPEG 90 | 91.7% | 0.918 | 93.0% |

Per class at 12 px/mm:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| sound | 0.934 | 0.935 | 0.934 |
| chalky | 0.903 | 0.919 | 0.911 |
| damaged | 0.942 | 0.964 | 0.953 |
| discoloured | 0.955 | 0.932 | 0.943 |

**2. Whole pipeline on real-kernel trays** (`python -m tests.realtray`). Each tray is 120 IIT **test** kernels laid on the printed calibration sheet at camera resolution, with blur, noise and JPEG. The full app pipeline runs on it: markers → segmentation → FCI length rule → model → FCI picking order. Scores are final FCI classes over 3 trays.

| | Rules only | **v2 model** | v1 model |
|---|---|---|---|
| 12 px/mm | 62.9% | **94.9%** | 61% |
| 18 px/mm | 63.0% | **96.9%** | – |

At 12 px/mm, by FCI class:
- sound 94%
- broken 100%
- chalky 100%
- damaged 90%

The rules alone miss almost all chalky and damaged kernels on real photos.

### Rig requirements

- Camera straight down, grains in a **single layer** on a **black** background (the calibration sheet), with diffuse light.
- **At least 10 px/mm** (the app switches to rules below this; the setting is `min_px_per_mm`). The model still scores 91.7% at 8 px/mm, so the limit could be lowered after field validation.
- A phone photo of 1920 px across a whole A4 sheet is only about 6.5 px/mm. Use the phone's highest photo resolution, or move the camera closer.

### Known limits

- **One variety, lab-quality source photos.** The data is BPT5204 only, photographed under lab lighting. Field accuracy must be validated against manual FCI analysis on our rig (milestone M6), and our own labelled photos (milestone M4) should be added.
- **Chalky labels are partly derived.** Partial-chalky kernels were split at the FCI "≥ ½ milky white" line using a measured milky fraction.
- **FCI Q1 is open.** Whether fungal discolouration counts as *damaged* or *discoloured* is set in the spec YAML, so it can change without retraining.
- **Touching kernels.** On real-kernel trays, about 2–7% extra objects come from the splitter cutting one kernel in two. A trained segmenter is the planned fix.

### History

- **v1** was trained with clean down-sampling only. It scored 92% on its own test images, but on realistic trays camera noise and JPEG dropped it to 61%: sound kernels were read as damaged or chalky. This was found by `tests/realtray.py` and fixed in v2 by the camera simulation.
