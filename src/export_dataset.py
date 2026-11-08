"""Build the de-identified dataset release used for training and for Google Colab.

The raw PEDIS data cannot be shared: the database holds names, national health
insurance (TAJ) numbers and addresses, and some raw files are photos of people.
This script packs only what training needs:

    feet/*.png          every preprocessed foot crop (96x224, 8-bit grey)
    feet.csv            per crop: QC metrics, qc_ok, split, pseudonymous patient id, targets
    images.csv          per raw image: pseudonymous patient id, exam year, targets
    splits/{train,val,test}.csv
                        the training inputs (foot_file) and outputs (targets) per split,
                        quality-controlled crops only
    dropped_images.csv  raw files that could not be used, with the reason
    DATA_README.md      provenance, columns and usage restrictions

Exact exam dates are reduced to the year, and the free-text patient notes are
never included. The result is written to <data-dir>/release/ and zipped to
<data-dir>/pedis_feet_release_v1.zip (upload that file; see src/download_data.py).

Usage:
    python src/export_dataset.py --data-dir data
"""

import argparse
import hashlib
import os
import shutil
import sys
import zipfile

import pandas as pd

RELEASE_NAME = "pedis_feet_release_v1"
TARGETS = ["age", "weight_kg", "shoe_size"]
CONDITIONS = ["diabetes", "neuropathy", "hallux_valgus", "flat_foot", "calcaneovalgus", "heel_spur",
              "toe_deformity", "spine", "leg_length_diff", "injury_surgery"]
SPLIT_COLUMNS = ["foot_file", "image_file", "patient_id", "side", "mirrored", *TARGETS, "has_note", *CONDITIONS]

DATA_README = """# PEDIS foot crop dataset (de-identified release v1)

Source: PEDIS podoscope system, ~44,000 plantar images recorded between 1997 and 2011, provided to
the team for the BME VITMAV45 course project. **Not for redistribution**; use only for this course.

Created by `src/parse_pedis_db.py`, `src/make_labels.py`, `src/preprocess.py` and
`src/export_dataset.py`. Patients are pseudonymous integers; names, insurance numbers, addresses,
exact dates and free-text notes are not included.

| File | Content |
| --- | --- |
| `feet/<image>_L.png`, `feet/<image>_R.png` | Foot crops, 96x224, 8-bit grey; the sole is bright, every foot is straightened and the left one is mirrored |
| `feet.csv` | One row per crop: QC metrics, `qc_ok`, `split`, `patient_id`, targets |
| `images.csv` | One row per raw image: `patient_id`, `exam_date` (year only), targets |
| `splits/train.csv`, `splits/val.csv`, `splits/test.csv` | Quality-controlled crops of each split: input file + targets |
| `dropped_images.csv` | Raw files that are not palette footprint images (photos, corrupt files) |

Targets: `age` (years), `weight_kg`, `shoe_size` (EU) are regression targets; the 10 condition
columns are weak, keyword-based labels from free-text notes and are only defined where `has_note == 1`.
The split is 70/15/15% by patient (no patient appears in two splits).
"""


def year_only(series):
    return series.astype(str).str[:4].where(series.notna(), "")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    processed = os.path.join(args.data_dir, "processed")
    release = os.path.join(args.data_dir, "release")
    os.makedirs(os.path.join(release, "splits"), exist_ok=True)

    feet = pd.read_csv(os.path.join(processed, "feet.csv"))
    feet["exam_date"] = year_only(feet["exam_date"])
    feet.to_csv(os.path.join(release, "feet.csv"), index=False)

    images = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    images["exam_date"] = year_only(images["exam_date"])
    images.to_csv(os.path.join(release, "images.csv"), index=False)

    shutil.copy(os.path.join(processed, "dropped_images.csv"), os.path.join(release, "dropped_images.csv"))
    with open(os.path.join(release, "DATA_README.md"), "w", encoding="utf-8") as f:
        f.write(DATA_README)

    usable = feet[(feet.qc_ok == 1) & feet.split.isin(["train", "val", "test"])]
    for split in ["train", "val", "test"]:
        part = usable[usable.split == split][SPLIT_COLUMNS]
        part.to_csv(os.path.join(release, "splits", f"{split}.csv"), index=False)
        print(f"{split:5s}: {len(part):6d} crops, {part.patient_id.nunique():5d} patients", file=sys.stderr)

    zip_path = os.path.join(args.data_dir, f"{RELEASE_NAME}.zip")
    feet_dir = os.path.join(processed, "feet")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for name in ["feet.csv", "images.csv", "dropped_images.csv", "DATA_README.md"]:
            z.write(os.path.join(release, name), f"{RELEASE_NAME}/{name}")
        for split in ["train", "val", "test"]:
            z.write(os.path.join(release, "splits", f"{split}.csv"), f"{RELEASE_NAME}/splits/{split}.csv")
        for name in sorted(feet.foot_file):
            z.write(os.path.join(feet_dir, name), f"{RELEASE_NAME}/feet/{name}")

    size_mb = os.path.getsize(zip_path) / 2 ** 20
    print(f"{zip_path}: {size_mb:.0f} MB, sha256 {sha256(zip_path)}", file=sys.stderr)


if __name__ == "__main__":
    main()
