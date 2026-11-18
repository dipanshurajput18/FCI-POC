# data/

Nothing in this folder is committed except this file: the datasets are tens of GB.

| Path | What | How to get it |
|---|---|---|
| `public/iit_indore/download/` | Zenodo zips | `python scripts/download_iit_indore.py` (add `--delete-zips` to save ~17 GB) |
| `public/iit_indore/extracted/Data/` | Metadata CSVs + segmented kernel images | created by the download script |
| `public/iit_indore/fci_labels.csv` | every kernel relabelled into FCI classes, with the rule used | `python -m training.relabel_iit` |
| `public/iit_indore/preview/` | contact sheets | `python scripts/make_contact_sheet.py` |
| `cache/rice_condition_128.npz` | 128 px canonical kernel crops for training (~700 MB) | `python -m training.build_cache` |

**Dataset:** IIT Indore rice grain benchmark, DOI [10.5281/zenodo.22043091](https://doi.org/10.5281/zenodo.22043091), licence **CC BY 4.0**. Its authors are Sapkal, Anoop K R, Miglani and Kankar of IIT Indore. Any model or report built from it must credit them.

Other candidate datasets and their licences are listed in [docs/DATASETS.md](../docs/DATASETS.md).
