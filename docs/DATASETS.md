# Dataset register: AI Grain Analyser (rice + sugar)

Compiled 17 Nov 2026 from searches of Kaggle, Hugging Face, Zenodo, Mendeley Data, Figshare, Roboflow Universe, GitHub, arXiv/MDPI/Elsevier, Zindi and Indian sources (data.gov.in, ICAR-IIRR, NRRI, Shodhganga).

**Licence key**
- ✅ = commercial use allowed (CC0, CC BY, MIT, Apache)
- 🔁 = commercial use allowed, but anything we build from it must keep the same licence (CC BY-SA)
- ⛔ = non-commercial only, or no licence given (all rights reserved). Research and pretraining only, unless the author gives written permission.
- ❓ = licence not verified. Roboflow project pages are behind a bot check, so a person must open each page and confirm before we use it.

**Annotation types:** *class* = one label per image · *box* = rectangle around each grain · *mask* = exact outline of each grain · *count* = per-image totals measured in a lab.

---

## Bottom line

1. **No public dataset matches FCI's definitions on Indian rice under a fixed rig.** Public data is for **pretraining**. The model we ship must be fine-tuned and validated on **our own images**.
2. Three new **Indian** sources were found and are worth getting first: **IIT Indore damage benchmark** (✅), **FRK fortified-rice images from North Indian mills** (⛔, ask the author), and Roboflow **"rice grain 2"** with FCI-style classes including PB-1121 (✅ per snippet).
3. **Sugar:** only small Roboflow sets (≈30–250 images) and sieve-labelled sediment data as a proxy. Plan on **our own sieve-graded samples + synthetic images**.
4. **No public data exists for parboiled rice, de-husked grains, inorganic foreign matter or pin-point damage** under FCI definitions.

---

## A. Rice: priority list

| # | Dataset | Size | Classes | Image type | Annotation | Origin | Licence | Use for |
|---|---|---|---|---|---|---|---|---|
| 1 | **IIT Indore high-res rice damage benchmark**: zenodo.org/records/22043091 | 151 GB | BPT-5204: full chalky, partial chalky, healthy, discoloured, peck damage, broken; also PB-1121, PB-1401, PR-14 | High-res originals + segmented foreground | class + manual segmentation subset | **India** | CC BY 4.0 ✅ | Core classifier pretraining (chalky, discoloured, damaged, broken) |
| 2 | **Roboflow "rice grain 2"** (someone-project): universe.roboflow.com/someone-project/rice-grain-2-inveb | 250 imgs | Blacktip, Broken, Chalky, Damaged, Dehusked, Healthy, **Healthy-1121**, Immature, Inorganic FM, Organic FM, **Red Grain** | Bulk | box | Likely India | ❓ (snippet says CC BY 4.0 for a sibling project) | Closest class list to FCI spec |
| 3 | **Roboflow "Rice Quality Detection"** (madhav-tiwary): /madhav-tiwary-24bce2576/rice-quality-detection and /madhav-tiwary/rice-quality-detection-v3 | 4.75k / 4.72k imgs | Broken, Chalky, Clean, Damaged, Discolored, Immature, Organic FM | Bulk | box | Likely India | ❓ | Largest FCI-style detection set |
| 4 | **AfricaRice (UNIDO/Zindi)**: kaggle.com/datasets/idsalifou/dataset-africarice | 1,341 imgs, 16 GB | Per-image counts: broken, chalky, red, yellow, green/immature, black; length, width, L/W; CIELAB | **Bulk trays** (paddy, brown, white), blue background, controlled light | **count** (lab equipment) | Africa | CC BY-SA 4.0 🔁 | **Benchmark for whole-sample % accuracy** |
| 5 | **RiceCC**: huggingface.co/datasets/xwk25/RiceCC | 21,053 imgs | whole / large broken / tiny broken; chalkiness | 960×960, grains on carrier | class (verify) | China | MIT ✅ (paper is NC; confirm) | Broken + chalky classifier |
| 6 | **NED Univ. rice tray set**: data.mendeley.com/datasets/jcgmmpbv6g/1 | 407 tray imgs (~70 grains each) + 9.9k crops | 3 varieties | **Bulk tray** | **mask** | Pakistan | CC BY 4.0 ✅ | **Touching-kernel segmentation training** |
| 7 | **Peruvian rice**: kaggle.com/datasets/cristhiansempertegui/dataset-de-arroz-peruano | 6,599 imgs | whole, stained, broken, chalky | Single grain, uniform background | class | Peru | MIT ✅ | Classifier pretraining |
| 8 | **Roboflow FCI-like small sets**: rice-grain-sorting (datasetproject-1056x, CC BY 4.0), aspire/rice-sren0 & school-b7udi/rice-6rl78 (CC BY 4.0), project-klj99/rice-quality-3 (144 imgs: sound/unsound/broken-chalky/whole-chalky/stone/plastic), darshan-dunna/rice-grain-quality-hlcdq (219 imgs, **instance seg**: good/broken/damaged/foreign), someone-project/rice-grain-3-f0eh8 (336 imgs: chalky/damaged/normal/yellow), rice-grain-wmeky/rice-grain-defects (2k imgs), object-detection-ooe3t/grain-detection (~2.8k imgs: head/broken/red), pradeep-oq764/rice_quality (240 imgs; broken size bins) | 50–2.8k each | Mixed | Bulk | box / some mask | Mixed | ❓ / CC BY 4.0 | Merge and remap to FCI classes after checking licences |
| 9 | **rice_quality_parameter**: kaggle.com/datasets/andiadityaa/rice-quality-parameter | 620 MB | head, broken, immature, red, damaged, chalky, unhulled, foreign objects | Bulk, varied backgrounds | box (likely) | Indonesia? | None given ⛔ | Ask the author; research only |
| 10 | **Fortified rice + FRK images**: kaggle.com/datasets/anuragjain14/fortified-rice-and-fortified-rice-kernel-images | 500 + 500 imgs, 3.6 GB | fortified rice, FRK | Industrial, 5 North Indian mills | class | **India** | None given ⛔ | **Only FRK source.** Ask the author for permission |
| 11 | **GrainSet (rice) + GrainSet-raw**: github.com/hellodfan/GrainSet · doi 10.6084/m9.figshare.22987292.v3 | 31k kernels + 15k raw frames | normal; broken, immature, moldy, pest, sprouted, heated, shriveled; impurities | Single kernel (top + bottom) + raw plate frames | class + **per-kernel mask** + weight/size XML | China, Cambodia | **Conflict:** figshare CC BY 4.0, GitHub CC BY-NC-SA ⛔ | Research pretraining. **Weight XML helps the count→weight model.** |
| 12 | **75k Rice seg (Koklu-based)**: github.com/AliKHaliliT/75000-Rice-Image-Dataset-for-Object-Detection-and-Object-Segmentation | 75k | 5 varieties | Single kernel, black background | **polygon mask** (auto) | Turkey | CC0 imgs / CC BY ann. ✅ | Segmentation pretraining |
| 13 | **Koklu Rice Image Dataset**: kaggle.com/datasets/muratkokludataset/rice-image-dataset | 75k | 5 varieties | Single kernel, black background | class | Turkey | CC0 ✅ | Shape/variety pretraining |
| 14 | **Milled Rice Grain (Sambalpur Univ.)**: data.mendeley.com/datasets/c5y6gjwdzh/1 | ~112k | 8 varieties incl. Jhili, HMT/Sona Masuri, Masuri | Black background, daylight | class | **India** | CC BY 4.0 ✅ | Indian variety / admixture |
| 15 | **Indian Basmati paddy seed (IARI)**: data.mendeley.com/datasets/byms2fjzpw/2 | 3,210 | 10 basmati paddy varieties + 1121/1509 mix | Single grain, rig | class | **India** | CC BY 4.0 ✅ | Admixture + paddy classes |
| 16 | **C4 Rice object detection**: kaggle.com/datasets/kevinkarl/c4-rice-dataset-for-object-detection | ~3.6k boxes | premium / medium grades | Bulk | box | Indonesia | CC BY-SA 4.0 🔁 | Detection pretraining |
| 17 | **Rice Images (broken/full/mixed)**: kaggle.com/datasets/shubhamcodez/rice-images-dataset | 64 MB | broken, full, mixed | ? | class | ? | CC0 ✅ | Broken classifier |
| 18 | Bangladesh sets: f6nw4hhznr (4.9k), Aruzz22.5K 3mn9843tz2 (23.6k bulk), 2fgv99854n (19k) | | varieties | phone / microscope | class | Bangladesh | CC BY 4.0 ✅ (2fgv… unverified) | Variety pretraining |
| 19 | **CORN-KERNEL (ESPOL)**: cidis.espol.edu.ec | 23 seg imgs + 10k crops | good / defective / impurity | **Touching kernels** | mask | Ecuador (maize) | ⛔ unverified | Touching-kernel segmentation reference |
| 20 | **GrainSpace**: github.com/hellodfan/GrainSpace | 5.25M | wheat/maize/rice damage | Single kernel | class | 5 countries | CC BY-NC-SA ⛔ | Research only; email the authors |

**Papers with no public data, where we could ask the authors:** the 8,048-image high-magnification rice damage set (Computers & Electronics in Agriculture 2022, S0168169922001284); arXiv 2210.02665 (chalky, yellow, spotted, broken + weight estimation); Kurade et al., Foods 2023 (IIT, 3,081 images, 8 Indian milled varieties).

## B. Sugar

| # | Dataset | Size | Content | Annotation | Licence | Use for |
|---|---|---|---|---|---|---|
| 1 | **Roboflow workspace "ai-sugar-crystal-size-estimation"**: universe.roboflow.com/ai-sugar-crystal-size-estimation (sugar-1/9-box, sugarbox_detectionv2, sugar_detection_segment_user) | ~30–220 imgs per project | Sugar crystals for size estimation | box + some mask | ❓ | Best direct sugar data found |
| 2 | **Roboflow "Sugar Crystal"**: /rams-workspace-6cnug/sugar-crystal | 31 imgs | Sugar crystals | **instance mask** | ❓ | Crystal segmentation seed |
| 3 | **Roboflow "Sugar" (sugar-crystal class)**: /object-detection-1zxjn/sugar-sl8wr | 250 imgs | Sugar crystals | **instance mask** | ❓ | Crystal segmentation |
| 4 | **Roboflow "sugar" (foreign particle)**: /test-kgv1f/sugar-kyfn1 | 55 imgs | Foreign particles in sugar | box | ❓ | Speck detection seed |
| 5 | **Roboflow "sugar" (d1pl0m)**: /d1pl0m/sugar-1mdnu | 1.19k imgs | sugar + "musor" (debris) classes | box | ❓ | Possibly foreign matter in sugar. Inspect. |
| 6 | **Roboflow "crystal" agglomeration**: /dlut-zzz/crystal-z96pq | 171 imgs | single vs agglomerated crystals | mask | ❓ | Touching-crystal split |
| 7 | **SediNet / SandSnap / segmenteverygrain**: github.com/DigitalGrainSize/SediNet · zenodo.org/records/7063226 · zsylvester.github.io/segmenteverygrain | 409+ imgs | Sediment photos with **sieve-measured grain size** | sieve size labels; SAM-based masks | code Apache-2.0; images ❓ | **Closest analogue to IS 498 sieve grading** |
| 8 | **OpenCrystalData (Strathclyde/CMAC)**: kaggle.com/opencrystaldata | 4 sets | In-situ crystallisation, size distribution, anomalous particles | size labels | likely NC ⛔ | Research pretraining |
| 9 | **Visual Granulometry (LUH)**: service.tib.eu/ldmservice/dataset/luh-visual-granulometry | 900 imgs | Aggregate grading curves, 8 px/mm | class per grading | CC BY-NC 3.0 ⛔ | Research reference for "grade from image" |

**Sugar authors to contact for data:** Wu et al., *ASSugarNet* (cane sugar crystal amodal segmentation, J. Food Eng. 2023); Purdue (F. Zhu / Z. Nagy) food-crystal masks; Masoudi et al. 2024 (massecuite, Shahid Chamran Univ.); UKZN sugar-crystal-size thesis (SMRI). **NSI Kanpur** prepares the official L/M/S/SS and 29/30/31 reference standards and is the ideal partner for ground truth.

---

## C. Plan for our own data (the part that makes the product)

**Rice**
- Fixed rig (backlit + top-lit), both images captured per tray.
- Spiked samples: clean base rice + a known number of brokens, chalky, red, discoloured, damaged, stones/husk added, giving exact ground truth.
- Cover raw + **parboiled**, ≥ 5 Indian varieties, ≥ 3 regions.
- FRK: blend 1:100 with purchased FRK.
- De-husked: stained reference (methylene blue) for ground truth only.

**Sugar**
- Buy L/M/S grades and sieve each one (IS 498 Annex C apertures) to get its true size distribution.
- Spike with specks: bagasse, carbon, sand, fibre.
- Capture with a grey card for colour.

**Synthetic data**
- Copy-paste kernels and crystals (cut out with SAM2) onto real tray backgrounds to create dense touching layouts with free masks.
- Paste real speck crops onto clean sugar.

**Labelling**
- Auto-label with the classical pipeline + SAM2 → correct in CVAT / Label Studio → export as COCO masks.

**Targets:** ≥ 5,000 labelled Indian rice kernels across all FCI classes; ≥ 30 sieve-graded sugar samples.

---

## D. Licence actions before training a shippable model

- [ ] Open each ❓ Roboflow page and record its licence. The pages show a Cloudflare check that a person must pass.
- [ ] RiceCC: confirm MIT applies to the data, not only the code.
- [ ] GrainSet: email the authors to resolve the CC BY vs CC BY-NC-SA conflict.
- [ ] FRK Kaggle set and rice_quality_parameter: request written permission from the uploaders.
- [ ] Keep a licence register in `data/LICENSES.csv`. Exclude ⛔ data from any model shipped to FCI unless permission is granted.
