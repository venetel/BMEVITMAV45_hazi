# Foot condition recognition from podoscope footprints

Homework project for *Deep Learning in Practice with Python and LUA* (BME VITMAV45).

The goal is to train a neural network that recognises foot conditions from plantar
footprint images recorded with the PEDIS podoscope system (~44k images, 1997–2011).

**Status:** milestone I (data acquisition, exploration, train/validation/test sets) is in progress.

## Repository layout

| Path | Content |
| --- | --- |
| `src/parse_pedis_db.py` | Reads the legacy PEDIS Access database (`PEDI3.MDB`) and extracts one record per image plus the free-text patient notes |
| `src/make_labels.py` | Builds the image-level label table: age, weight, shoe size and weak condition labels |
| `src/preprocess.py` | Image cleaning: colour decoding, foot separation, cropping, quality control, patient-level split |
| `notebooks/01_data_exploration.ipynb` | Exploratory data analysis of the images, the cleaning step and the labels |
| `data/` | Generated files; not in the repository (see *Data*) |

## Environment

Python 3.11+ (developed with Python 3.14 on Windows).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The raw data (the PEDIS image folder and its Access databases) is **not part of this repository**.
It contains personal data: patient names, national health insurance (TAJ) numbers, addresses,
and photos of people. Ask the team for access. Put the folder next to the repository root, e.g.
`PEDIS - talpnyomás adatok/PEDIS/`. Everything under `data/` is generated and is ignored by git.

## Reproducing the data pipeline

Run the three steps from the repository root (`RAW` = the raw PEDIS folder):

```bash
python src/parse_pedis_db.py --db "RAW/PEDI3.MDB" --images "RAW" --out-dir data
python src/make_labels.py --data-dir data
python src/preprocess.py --raw "RAW" --data-dir data
```

| Step | Output | Notes |
| --- | --- | --- |
| `parse_pedis_db.py` | `data/pedis_records.csv`, `data/pedis_ekorlap_notes.csv` | Direct identifiers are dropped; patients get pseudonymous ids. The notes file is free text and must stay local. |
| `make_labels.py` | `data/labels.csv` | One row per image: age, weight, shoe size, 10 keyword-based condition labels |
| `preprocess.py` | `data/processed/feet/*.png`, `data/processed/feet.csv`, `data/processed/dropped_images.csv` | Two 96×224 crops per image (left foot mirrored), QC metrics, `qc_ok` flag, `split` column. Takes ~6 minutes on 8 cores. |

Then open `notebooks/01_data_exploration.ipynb` and run all cells.

### Key preprocessing facts

* The BMP files are palette images, but the palette index is **not** an intensity. The PEDIS
  software rendered the camera image with a 16-colour false-colour scale, and the colours were
  re-quantised into arbitrary palette slots. `preprocess.decode` maps every colour back to its
  level (0–15) on that scale.
* The image is inverted, so the sole is bright on a dark background.
* Quality control removes overexposed, empty and wrongly coloured frames, and failed foot
  localisations. 76,575 of 88,430 foot crops (86.6%) pass.
* The split is 70/15/15% **by patient**, so no person appears in two subsets.

## Team

<!-- names and roles to be filled in -->

## License

Course project; the data is not redistributable.
