# FCI requirements traceability: AI Grain Analyser

**Rule for this project:** FCI's requirement is the reference. Every feature, threshold, dataset label and model output must map to an FCI requirement or definition below. Outside data and definitions are converted to FCI's rules, never the other way round.

Sources:
- FCI Expression of Interest, Auto Grain Analyser, as summarised in the Uneecops demonstration deck
- DFPD Uniform Specification KMS 2026-27
- IS 4333 (Part 1) (method of analysis)
- IS 2813 (terminology)
- IS 498 / IS 5982 (sugar)

Status key: ✅ done · 🟡 partial · ❌ not yet · ➖ out of Phase 1 scope

Last updated: 18 Nov 2026

---

## 1. EoI requirements

| ID | FCI requirement | Status | How it is met / gap | Where |
|---|---|---|---|---|
| EOI-1 | Assess quality by a scientific device, not by eye; same sample gives the same result | ✅ | Deterministic pipeline; same image gives the same result; tests in `tests/` | `grain/` |
| EOI-2 | All parameters of the specification **except moisture** | 🟡 | See section 2. Gaps: admixture, de-husked, fine foreign matter | `grain/rice.py` |
| EOI-3 | **360° images** of the grain sample | 🟡 | Multi-shot (re-spread and re-capture) approximates both faces. True 360° needs the multi-camera device (hardware phase). | `server/engine.py` |
| EOI-4 | On-device edge **or** remote edge processing | 🟡 | Runs on CPU (laptop). Same code targets Jetson (on-device) or an edge server (remote). | `grain/` is UI-independent |
| EOI-5 | Practical method for **de-husked grains** | ❌ | FCI method is staining (methylene blue / metanil yellow, count basis). Camera method to be agreed with FCI. | - |
| EOI-6 | Practical method for **foreign matter** | 🟡 | Visible stones, mud and husk by image. Dust and fine mineral matter need the sieve + balance step. | `grain/rice.py` |
| EOI-7 | Image record for disputes (evidence) | ✅ | Every shot stored raw and annotated, with a per-grain "why" | `server/engine.py`, UI inspector |
| EOI-8 | Results posted to the **FCI ERP** without manual entry | ❌ | Next milestone: signed report (SHA-256 + QR) and ERP posting | - |
| EOI-9 | Tamper-evident results | ❌ | Next milestone: hash and signature per report | - |

## 2. Rice parameters: Uniform Specification KMS 2026-27

All limits are % **by weight** (except de-husked, which is by count). Each grain counts under **one** refraction, in the IS 4333 picking order: foreign → damaged → discoloured → small broken → broken → chalky → red.

| ID | Parameter | FCI definition | Limit raw / parboiled | Status | Method |
|---|---|---|---|---|---|
| SPEC-1 | Brokens | Piece **< ¾** of the average whole-kernel length | 25 / 16 (Improved Rice: 10 / 5) | ✅ | Length vs sample whole-kernel length |
| SPEC-2 | Small brokens | Piece **< ¼** of the whole kernel | ≤ 1 | ✅ | Length |
| SPEC-3 | Foreign matter | Inorganic (stones, mud) and organic (husk, chaff, seeds, pieces < ⅛ kernel) | 0.5 (mineral ≤ 0.2) | 🟡 | Colour and shape; fine matter not covered |
| SPEC-4 | Damaged / slightly damaged | Damaged by insects, fungus, heat, etc.; includes pin-point damage | 3 / 4 | 🟡 | Interior dark spots; trained model planned |
| SPEC-5 | Discoloured | Colour changed but not materially damaged | 3 / 5 | ✅ | Yellowness (b*) vs the sample |
| SPEC-6 | Chalky | **≥ ½** of the kernel milky white | 5 / n.a. | 🟡 | Opaque fraction; **backlight required** for reliable results |
| SPEC-7 | Red grains | Red pericarp | 3 / 3 | ✅ | Red fraction (a*) |
| SPEC-8 | Admixture of lower class | Different size/shape class | 6 | ❌ | L/B class rules + variety data (PB1121 / PB1401 / PR14) |
| SPEC-9 | De-husked grains | > ¼ of surface covered by bran (count basis) | 13 | ❌ | See EOI-5 |
| SPEC-10 | Moisture | - | 14 (15 with cut) | ➖ | Phase 2, MoistureLink |
| SPEC-11 | Weight basis | % by weight | - | ✅ | Volume × density; balance reconciliation planned |
| SPEC-12 | Spec limits configurable per season | - | - | ✅ | `specs/rice_kms_2026_27.yaml` |

## 3. Sugar (IS 498 / IS 5982), pending scope confirmation

FCI's EoI names rice, wheat and paddy only. **Sugar stays in the prototype until FCI confirms whether it is in scope.**

| ID | Parameter | Status | Method |
|---|---|---|---|
| SUG-1 | Size group L / M / S / SS (sieve test) | ✅ | Virtual sieve on crystal minor dimension |
| SUG-2 | Colour grade 31 / 30 / 29 (NSI standards) | 🟡 | Whiteness proxy. Thresholds are placeholders until calibrated against NSI standards. |
| SUG-3 | Black specks / foreign particles | 🟡 | Interior dark inclusions |
| SUG-4 | ICUMSA colour, moisture, SO₂, ash, polarisation | ➖ | Lab tests |

## 4. Training data must use FCI categories

- **Length-based categories (broken, small broken) are always decided by measurement against FCI's ¾ / ¼ rule.** They are never learned from another dataset's labels.
- External labels are mapped to FCI categories by `specs/label_map_iit_indore.yaml`. The script `training/relabel_iit.py` applies the map.
- Each mapped kernel records which rule decided its FCI class, so the mapping can be audited.
- Mappings marked **pending FCI decision** are flagged in the output and must be confirmed before a model trained on them is used for official results.

### IIT Indore (Zenodo 10.5281/zenodo.22043091, CC BY 4.0) → FCI

| Source label | FCI category | Rule |
|---|---|---|
| Healthy | sound | Then the length rule (< ¾ → broken, < ¼ → small broken) |
| Broken | broken / small broken / **sound** | **Re-judged by length only.** 114 of 1,001 are ≥ ¾ of whole length, so FCI calls them whole. |
| Full chalky | chalky | Matches "≥ ½ milky white"; length rule applies first |
| Partial chalky | chalky **or** sound | Measured chalky fraction of the kernel ≥ 0.5 → chalky, else sound |
| Peck damage | damaged | Insect damage is "damaged" |
| Discolored (fungal pigmentation) | **damaged** (default) | ⚠️ **Pending FCI decision Q1** |
| PB1121 / PB1401 / PR14 | sound + variety tag | For admixture work (SPEC-8) |

## 5. Open questions for FCI's QC team

| # | Question | Affects |
|---|---|---|
| Q1 | Is a fungus-discoloured kernel **damaged** or **discoloured**? | SPEC-4, SPEC-5, IIT label map |
| Q2 | Chalky "½ of the kernel": judged from one face or both? | SPEC-6 |
| Q3 | Validation method: 20 g in duplicate (IS 4333) or the FCI depot procedure? Which institute validates? | Accuracy targets |
| Q4 | Is sugar in scope? | Section 3 |
| Q5 | Accepted camera method for de-husked grains, or keep staining? | EOI-5, SPEC-9 |
| Q6 | ERP API documentation or sandbox | EOI-8 |
