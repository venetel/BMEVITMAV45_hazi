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
| `tools/review_app.py` | Browser app for the manual review of the quality-control decisions |
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
| `preprocess.py` | `data/processed/feet/*.png`, `data/processed/feet.csv`, `data/processed/dropped_images.csv` | Two 96×224 crops per image (each foot straightened, left foot mirrored), QC metrics, foot tilt, `qc_ok` flag, `split` column. Takes ~6 minutes on 8 cores. |

Then open `notebooks/01_data_exploration.ipynb` and run all cells.

### Manual review of the quality control

`data/processed/feet/` holds every crop, including the ones the quality control rejected; `qc_ok` in
`feet.csv` decides. `tools/review_app.py` is a small browser app (standard library only) to check
those decisions by hand. It shows a grid of crops with the automatic decision pre-filled
(green = usable, red = rejected). Click a crop to flip it, Shift+click to mark it unsure, and press
Enter to save the page. Press O over a crop to open the full original image.

```bash
python tools/review_app.py --reviewer <name> --queue borderline --shard 1/3
```

Then open http://localhost:8765.

| Queue | Crops | Purpose |
| --- | --- | --- |
| `borderline` | ~4,400 | QC metrics within 3% of a threshold; most mistakes are here |
| `random` | all, shuffled | a sample of a few hundred measures how accurate the QC is |
| `rejected` / `passed` | 12,442 / 75,988 | browse one side of the decision |

`--shard i/n` splits a queue between `n` reviewers. Each reviewer writes
`data/review/review_<name>.csv`. Crops that are already in any file under `data/review/` are skipped,
so without `--shard` reviewers only avoid duplicates when they share one machine; on separate
machines use `--shard`, then copy everyone's file into one `data/review/` folder.
`python tools/review_app.py --summary` merges the files into `data/review/review_merged.csv`
(majority vote) and prints how often the reviewers agree with the QC.

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

## Team

<!-- names and roles to be filled in -->

## License

Course project; the data is not redistributable.
