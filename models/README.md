# models/

| Model | Task | Test accuracy at camera resolution (12 px/mm) | Files |
|---|---|---|---|
| `rice_condition_v1` | rice kernel condition: sound / chalky / damaged / discoloured | 92.6% on clean down-sampled crops, **but about 61–66% with real camera noise and JPEG** (see below) | `.onnx` (16.8 MB), `.json` (full metrics and training history) |
| `rice_condition_v2` | same task, trained with camera simulation (`training/camera_aug.py`) | in training; validation ~93% on noisy 12 px/mm camera images | added when training finishes |

> **v1 is not robust to camera noise.** On a tray of real IIT test kernels with realistic sensor noise and JPEG compression (`tests/realtray.py`), v1 fell from 90% accuracy (clean) to 61%: sound kernels were read as damaged or chalky. v2 fixes this by training through a camera simulator that uses the app's own crop code.

PyTorch checkpoints (`*.pt`) are not committed. Rebuild them with the commands in the main README.

---

## Model card: `rice_condition_v1`

**Job.** The model judges the *condition* of one rice kernel. It does **not** decide broken or small broken. Those come from the FCI length rule (< ¾ and < ¼ of the whole-kernel length) in `grain/rice.py`, because length is measured, not guessed.

| | |
|---|---|
| Architecture | MobileNetV3-Large (torchvision, BSD-3), ImageNet `IMAGENET1K_V2` initialisation, last layer → 4 classes |
| Input | 128 × 128 RGB crop from `grain.crop.canonical_crop`: kernel upright, centred on black, long side = 112 px. Scale 1/255, ImageNet mean/std. |
| Output | logits for `[sound, chalky, damaged, discoloured]` |
| Runtime | ONNX opset 17, CPU (onnxruntime or OpenCV DNN); max \|torch − onnx\| = 4e-6 |
| Training | 14 epochs, AdamW, one-cycle LR 1e-3, class-weighted cross-entropy + label smoothing 0.05, about 1 h on an i3-1215U CPU |
| Augmentation | flips, ±10° rotation, scale, white balance / brightness / contrast, noise, **down-and-up sampling to 4–22 px/mm** (camera resolution) |

### Data

The data is the IIT Indore rice benchmark, BPT5204 segmented images: DOI [10.5281/zenodo.22043091](https://doi.org/10.5281/zenodo.22043091), **CC BY 4.0**, by Sapkal, Anoop K R, Miglani and Kankar of IIT Indore. It was relabelled into FCI terms by `training/relabel_iit.py` and `specs/label_map_iit_indore.yaml`.

| Model class | Source labels | Train / val / test |
|---|---|---|
| sound | Healthy, Broken (its condition), Partial chalky with < ½ milky white | 3656 / 799 / 814 |
| chalky | Full chalky, Partial chalky with ≥ ½ milky white (FCI definition) | 1977 / 413 / 384 |
| damaged | Peck damage | 1504 / 322 / 302 |
| discoloured | Discolored (fungal) | 2824 / 602 / 631 |

The split follows the official IIT Indore `6class` split where a grain has one; other grains are assigned 70/15/15 per class (seed 42). The test set is 2,131 held-out kernels.

### Test results (held-out kernels)

| Resolution | Accuracy | Macro-F1 | With FCI Q1 default (discoloured → damaged) |
|---|---|---|---|
| Dataset, about 22 px/mm | 92.0% | 0.916 | 92.8% |
| **Camera, about 12 px/mm** | **92.6%** | **0.927** | **93.6%** |
| Low camera, about 6 px/mm | 77.8% | 0.799 | 79.3% |

Per class at 12 px/mm:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| sound | 0.966 | 0.880 | 0.921 |
| chalky | 0.787 | 0.982 | 0.874 |
| damaged | 0.936 | 0.977 | 0.956 |
| discoloured | 0.983 | 0.929 | 0.955 |

### What this means for the rig and for FCI

- **Damaged and discoloured are reliable at every resolution.** F1 is above 0.94 even at 6 px/mm.
- **Sound versus chalky is the weak spot, and it depends on resolution.**
  - At high resolution the model under-calls chalky: 86 chalky kernels were called sound.
  - At low resolution blur makes translucent kernels look milky, so it over-calls chalky: 89 sound kernels were called chalky at 12 px/mm and 298 at 6 px/mm.
- **Rig requirement: at least 12 px/mm on the tray, and 15–20 px/mm is better.** At 6 px/mm, chalky cannot be trusted.
- **Chalky labels are partly derived.** Partial-chalky kernels were split at the FCI "≥ ½ milky white" line using a measured milky fraction, so the chalky/sound boundary in the data is itself approximate. FCI-annotated kernels from our own rig (milestone M4) are the fix.

### Known limits

- **Dataset accuracy is not field accuracy.** The data is one variety (BPT5204), single kernels photographed under lab lighting. It must be validated against manual FCI analysis on our rig (milestone M6).
- **Foreign matter, red kernels and de-husked grains are not in the training data.** These stay with the rule-based checks.
- **FCI Q1 is open.** Whether fungal discolouration counts as *damaged* or *discoloured* is set in the YAML label map, so it can change without retraining.
