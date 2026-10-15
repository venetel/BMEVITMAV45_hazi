"""Turn the parsed PEDIS records and e-record notes into an image-level label table.

Inputs (produced by parse_pedis_db.py):
    pedis_records.csv        one row per footprint image, no direct identifiers
    pedis_ekorlap_notes.csv  free-text anamnesis per patient (keep local)

Output:
    labels.csv  one row per existing image file with metadata targets
                (age, weight, shoe size) and multi-hot condition labels.

Condition labels come from keyword matching on the Hungarian/Latin notes, so
they are patient-level and weak: a note is not dated, and a missing keyword
does not prove the condition is absent. Only patients that have a note get
condition labels; for everyone else they are left empty (unknown), not 0.
"""

import argparse
import collections
import csv
import os
import re
import sys

# Condition -> regex over the lower-cased note text.
CONDITIONS = {
    "diabetes": r"\bn?iddm\b|diab|cukor",
    "neuropathy": r"neuropath",
    "hallux_valgus": r"hall?a?ux|bütyök",
    "flat_foot": r"planus|transversopl|lúdtalp|süllyed",
    "calcaneovalgus": r"calcaneo-?valgus|sarokcsont bed",
    "heel_spur": r"sarkanty",
    "toe_deformity": r"digit(us|i) mall|kalapácsujj",
    "spine": r"scol|skol|gerinc|csigoly|cobb|porckorong|lumb",
    "leg_length_diff": r"rövidebb|rövidül|\d+\s*(mm|cm)-?rel",
    "injury_surgery": r"törés|fract|baleset|sérül|szakad|műtét|mütét|protézis|prot\.",
}


# Plausible ranges; values outside are typos in the original data entry.
VALID_RANGE = {"age": (1, 105), "weight_kg": (8, 200), "shoe_size": (15, 52)}


def to_number(text, field):
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        return ""
    low, high = VALID_RANGE[field]
    return value if low <= value <= high else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data", help="folder with the parser outputs")
    parser.add_argument("--out", default=None, help="output CSV (default: <data-dir>/labels.csv)")
    args = parser.parse_args()
    out_path = args.out or os.path.join(args.data_dir, "labels.csv")

    with open(os.path.join(args.data_dir, "pedis_records.csv"), encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    with open(os.path.join(args.data_dir, "pedis_ekorlap_notes.csv"), encoding="utf-8") as f:
        notes = list(csv.DictReader(f))

    patterns = {name: re.compile(regex) for name, regex in CONDITIONS.items()}
    patient_labels = {}
    for note in notes:
        text = note["note"].lower()
        labels = patient_labels.setdefault(note["patient_id"], dict.fromkeys(CONDITIONS, 0))
        for name, pattern in patterns.items():
            if pattern.search(text):
                labels[name] = 1

    rows, seen = [], set()
    for record in records:
        if not record.get("image_file") or record["image_file"] in seen:
            continue
        seen.add(record["image_file"])
        labels = patient_labels.get(record["patient_id"])
        row = {
            "image_file": record["image_file"],
            "patient_id": record["patient_id"],
            "exam_date": record["exam_date"],
            "age": to_number(record["kor"], "age"),
            "weight_kg": to_number(record["suly"], "weight_kg"),
            "shoe_size": to_number(record["cmeret"], "shoe_size"),
            "has_note": int(labels is not None),
        }
        # Diabetes is also recorded in structured fields of the image record.
        structured_diabetes = record.get("diabetes") == "1" or bool(record.get("dtip"))
        for name in CONDITIONS:
            row[name] = labels[name] if labels else ""
        if structured_diabetes:
            row["diabetes"] = 1
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    patients = {r["patient_id"] for r in rows}
    print(f"{len(rows)} images of {len(patients)} patients written to {out_path}", file=sys.stderr)
    print(f"images with a note: {sum(r['has_note'] for r in rows)}", file=sys.stderr)
    for name in CONDITIONS:
        positive = [r for r in rows if r[name] == 1]
        print(f"  {name:16s} images={len(positive):5d} patients={len({r['patient_id'] for r in positive}):4d}",
              file=sys.stderr)
    for name in ["age", "weight_kg", "shoe_size"]:
        values = [r[name] for r in rows if r[name] != ""]
        print(f"  {name:16s} known for {len(values)} images", file=sys.stderr)


if __name__ == "__main__":
    main()
