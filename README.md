# Foot arch type recognition from podoscope footprints

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/venetel/BMEVITMAV45_hazi/blob/main/notebooks/01_data_exploration.ipynb)

Homework project for *Deep Learning in Practice with Python and LUA* (BME VITMAV45, autumn 2026).

## Team

**Team name: Levi_es_a_biomechasok**

| Name | Neptun code |
| --- | --- |
| Veibli-Magyari Kristóf | UF5793 |
| Faragó-Pethő Luca | JFSCLX |
| Pintér Levente | GHHKWS |

## Project

We train a convolutional network on plantar footprint images of the PEDIS podoscope system
(~44,000 images, 1997–2011). The main task is to recognise the **foot arch type** (flat / normal /
high) of each foot; its labels are computed from an arch index of the foot outline. Age, body weight
and shoe size are learned as auxiliary regression targets.

**Status:** milestone I (data source, exploration, train/validation/test sets) is complete.

## Data source

The images and the patient database were exported from the PEDIS podoscope software and provided to
the team for this course. The raw data holds personal data (names,
national health insurance numbers, addresses, photos of people), so it is **not public**.

We publish a **de-identified release** instead: the preprocessed foot crops, the train / validation /
test split files and the targets, with pseudonymous patient ids and exam years only
(`pedis_feet_release_v1.zip`, built by `src/export_dataset.py`). `src/download_data.py` downloads and
unpacks it (the link is built into the script).

## Repository files

| Path | Function |
| --- | --- |
| `src/parse_pedis_db.py` | Reads the legacy PEDIS Access database (`PEDI3.MDB`) and extracts one record per image plus the free-text patient notes |
| `src/make_labels.py` | Builds the image-level label table: age, weight, shoe size and weak condition labels |
| `src/preprocess.py` | Image cleaning: colour decoding, foot separation, straightening, cropping, quality control, patient-level split |
| `src/export_dataset.py` | Packs the de-identified release (crops, split files, targets) into one zip |
| `src/download_data.py` | Downloads and unpacks the de-identified release (Google Drive link or any URL) |
| `src/dataset.py` | Loads the training, validation and test inputs (crops) and outputs (targets) |
| `notebooks/01_data_exploration.ipynb` | Data exploration and visualisation: images, cleaning, labels, split, training inputs/outputs |
| `requirements.txt` | Python dependencies |
| `data/` | Generated and downloaded data; not in the repository |

## How to run

**On Google Colab:** open `notebooks/01_data_exploration.ipynb` with the *Open in Colab* button above
and run all cells; the first code cell clones the repository and downloads the dataset (~1 GB). The sections that
need the raw files are skipped; their outputs are stored in the notebook.

**Locally, with the de-identified release:**

```bash
pip install -r requirements.txt
python src/download_data.py
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Locally, from the raw data** (team members only): see *Reproducing the data pipeline* below.

## Environment

Python 3.11+ (developed with Python 3.14 on Windows).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the data pipeline

Team members only: put the raw PEDIS folder next to the repository root, e.g.
`PEDIS - talpnyomás adatok/PEDIS/`. Everything under `data/` is generated and is ignored by git.

Run the steps from the repository root (`RAW` = the raw PEDIS folder):

```bash
python src/parse_pedis_db.py --db "RAW/PEDI3.MDB" --images "RAW" --out-dir data
python src/make_labels.py --data-dir data
python src/preprocess.py --raw "RAW" --data-dir data
python src/export_dataset.py --data-dir data
```

| Step | Output | Notes |
| --- | --- | --- |
| `parse_pedis_db.py` | `data/pedis_records.csv`, `data/pedis_ekorlap_notes.csv` | Direct identifiers are dropped; patients get pseudonymous ids. The notes file is free text and must stay local. |
| `make_labels.py` | `data/labels.csv` | One row per image: age, weight, shoe size, 10 keyword-based condition labels |
| `preprocess.py` | `data/processed/feet/*.png`, `data/processed/feet.csv`, `data/processed/dropped_images.csv` | Two 96×224 crops per image (each foot straightened, left foot mirrored), QC metrics, foot tilt, `qc_ok` flag, `split` column. Takes ~6 minutes on 8 cores. |
| `export_dataset.py` | `data/release/`, `data/pedis_feet_release_v1.zip` | De-identified release: exact dates reduced to the year, no free text. Upload the zip and share its link. |

Then open `notebooks/01_data_exploration.ipynb` and run all cells.

### Key preprocessing facts

* The BMP files are palette images, but the palette index is **not** an intensity. The PEDIS
  software rendered the camera image with a 16-colour false-colour scale, and the colours were
  re-quantised into arbitrary palette slots. `preprocess.decode` maps every colour back to its
  level (0–15) on that scale.
* The image is inverted, so the sole is bright on a dark background.
* Each foot is straightened: the image is rotated about the foot so that the principal axis of
  the foot mask is vertical. The foot is not rescaled, because its length carries information.
* Quality control removes overexposed, empty and wrongly coloured frames, and failed foot
  localisations. 75,988 of 88,430 foot crops (85.9%) pass.
* The split is 70/15/15% **by patient**, so no person appears in two subsets.

## License

Course project; the data is not redistributable.
