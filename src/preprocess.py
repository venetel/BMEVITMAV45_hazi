"""Clean and preprocess the PEDIS footprint images into per-foot training crops.

Pipeline for every raw BMP image:

1. Decode. The BMPs are 8-bit palette images, but the palette index is NOT an
   intensity: the PEDIS software rendered the grey camera image with a 16-colour
   false-colour scale and the colours were re-quantised into arbitrary palette
   slots. The intensity is recovered by mapping every colour back to its
   position on that scale (FALSE_COLOUR_SCALE, 0..15). Images whose palette is
   already grey are converted by luminance. RGB files (photos, screenshots,
   sometimes of people) and unreadable files are dropped.
2. Invert, so that the sole (dark on the podoscope) is bright on a dark
   background.
3. Split the image into the two feet at the darkest column between them.
4. Locate each foot (thresholded, smoothed mask; largest blob plus aligned
   blobs such as toes and a detached heel), and compute quality metrics.
5. Crop a fixed-aspect window (full image height) centred on the foot, mirror
   the foot on the left of the image so all crops have the same orientation,
   resize to CROP_SIZE and save as 8-bit PNG.
6. Write a manifest with quality flags and a patient-level train/val/test split.

Usage:
    python src/preprocess.py --raw "<raw image folder>" --data-dir data [--workers 8] [--limit N]
"""

import argparse
import multiprocessing as mp
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

# PEDIS false-colour scale, from darkest to brightest camera level.
FALSE_COLOUR_SCALE = [
    (0, 0, 0), (128, 0, 128), (0, 0, 128), (0, 0, 255), (0, 128, 128), (0, 128, 0), (128, 0, 0), (255, 0, 0),
    (255, 0, 255), (128, 128, 0), (128, 128, 128), (192, 192, 192), (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (255, 255, 255),
]
COLOUR_TO_LEVEL = {r << 16 | g << 8 | b: level for level, (r, g, b) in enumerate(FALSE_COLOUR_SCALE)}

CROP_SIZE = (96, 224)            # (width, height) of the saved foot crops
CROP_ASPECT = CROP_SIZE[1] / CROP_SIZE[0]
SIDE_MARGIN = 0.06               # fraction of the width ignored at both sides (platform frame)
FOOT_QUANTILE = 0.75             # brightest 25% of a half image is treated as foot candidate

# Quality thresholds (see notebooks/01_data_exploration.ipynb for the distributions).
MIN_FOOT_HEIGHT = 0.45           # foot blob height / image height
FOOT_ASPECT_RANGE = (1.3, 4.5)   # foot blob height / width
MIN_FILL = 0.25                  # blob pixels / bounding-box area
MAX_BORDER_BRIGHTNESS = 0.55     # mean of the top/bottom rows; high = overexposed platform
MIN_CONTRAST = 0.08              # std of the inverted image
MAX_ROUGHNESS = 0.20             # mean |image - blur| / contrast; banded noise is far above


def decode(path):
    """Return the inverted intensity image in [0, 1] (float32), or None if unusable."""
    try:
        with Image.open(path) as im:
            if im.mode not in ("P", "L"):
                return None
            rgb = np.asarray(im.convert("RGB"))
    except Exception:
        return None
    key = rgb[..., 0].astype(np.int32) << 16 | rgb[..., 1].astype(np.int32) << 8 | rgb[..., 2]
    colours, inverse = np.unique(key, return_inverse=True)
    levels = np.array([COLOUR_TO_LEVEL.get(int(c), -1) for c in colours])
    if (levels >= 0).all():
        intensity = levels[inverse].reshape(key.shape) / 15.0
    else:  # grey palette or greyscale image
        intensity = rgb.mean(axis=-1) / 255.0
    return (1.0 - intensity).astype(np.float32)


def locate_foot(half):
    """Return (x_centre, metrics) of the foot in one half of the (smoothed) image."""
    h, w = half.shape
    mask = half > np.quantile(half, FOOT_QUANTILE)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 7)))  # removes thin vertical streaks
    mask = ndi.binary_opening(mask, iterations=1)
    labels, n = ndi.label(mask)
    if n == 0:
        return w / 2, {"foot_height": 0.0, "foot_aspect": 0.0, "foot_fill": 0.0}
    areas = ndi.sum(mask, labels, range(1, n + 1))
    objects = ndi.find_objects(labels)
    main = int(np.argmax(areas))
    y0, y1 = objects[main][0].start, objects[main][0].stop
    x0, x1 = objects[main][1].start, objects[main][1].stop
    keep = labels == main + 1
    # Toes and a detached heel are separate blobs in the same column as the main blob.
    min_area = 0.002 * h * w
    for i, (ys, xs) in enumerate(objects):
        if i == main or areas[i] < min_area:
            continue
        cx = (xs.start + xs.stop) / 2
        if x0 - 8 <= cx <= x1 + 8:
            y0, y1, x0, x1 = min(y0, ys.start), max(y1, ys.stop), min(x0, xs.start), max(x1, xs.stop)
            keep |= labels == i + 1
    box_h, box_w = y1 - y0, x1 - x0
    x_centre = float(np.average(np.nonzero(keep)[1]))
    return x_centre, {"foot_height": box_h / h, "foot_aspect": box_h / max(box_w, 1),
                      "foot_fill": float(keep.sum()) / max(box_h * box_w, 1)}


def crop_foot(image, x_centre):
    """Fixed-aspect window of full image height centred on x_centre, resized to CROP_SIZE."""
    h, w = image.shape
    win_w = int(round(h / CROP_ASPECT))
    x0 = int(round(x_centre - win_w / 2))
    padded = np.pad(image, ((0, 0), (win_w, win_w)))  # zero padding = background after inversion
    window = padded[:, x0 + win_w:x0 + 2 * win_w]
    resized = Image.fromarray((window * 255).astype(np.uint8)).resize(CROP_SIZE, Image.BILINEAR)
    return np.asarray(resized)


def process_image(image_file, raw_dir, out_dir):
    """Process one raw image; return a list of manifest rows (one per foot) or a drop reason."""
    image = decode(os.path.join(raw_dir, image_file))
    if image is None:
        return [{"image_file": image_file, "drop_reason": "not a palette/grey image or unreadable"}]
    h, w = image.shape
    margin = int(w * SIDE_MARGIN)
    smooth = ndi.gaussian_filter(image, 2)
    smooth[:, :margin] = 0
    smooth[:, w - margin:] = 0
    columns = ndi.uniform_filter1d(smooth.mean(axis=0), 15)
    lo, hi = int(w * 0.35), int(w * 0.65)
    split = lo + int(np.argmin(columns[lo:hi]))

    rows_border = max(1, int(h * 0.08))
    border_brightness = float(np.concatenate([image[:rows_border], image[-rows_border:]]).mean())
    contrast = float(image.std())
    # Images rendered with another colour scale decode into banded noise: high pixel-level roughness.
    roughness = float(np.abs(image - ndi.gaussian_filter(image, 1)).mean() / max(contrast, 1e-6))
    image_ok = (border_brightness <= MAX_BORDER_BRIGHTNESS and contrast >= MIN_CONTRAST
                and roughness <= MAX_ROUGHNESS)

    stem = os.path.splitext(image_file)[0]
    rows = []
    for side, (x_start, x_stop) in [("image_left", (margin, split)), ("image_right", (split, w - margin))]:
        x_centre, metrics = locate_foot(smooth[:, x_start:x_stop])
        crop = crop_foot(image, x_start + x_centre)
        mirrored = side == "image_left"
        if mirrored:
            crop = crop[:, ::-1]
        foot_file = f"{stem}_{'L' if side == 'image_left' else 'R'}.png"
        Image.fromarray(np.ascontiguousarray(crop)).save(os.path.join(out_dir, foot_file))
        foot_ok = (metrics["foot_height"] >= MIN_FOOT_HEIGHT
                   and FOOT_ASPECT_RANGE[0] <= metrics["foot_aspect"] <= FOOT_ASPECT_RANGE[1]
                   and metrics["foot_fill"] >= MIN_FILL)
        rows.append({"image_file": image_file, "foot_file": foot_file, "side": side, "mirrored": int(mirrored),
                     "width": w, "height": h, "border_brightness": round(border_brightness, 4),
                     "contrast": round(contrast, 4), "roughness": round(roughness, 4), **{k: round(v, 4) for k, v in metrics.items()},
                     "qc_ok": int(image_ok and foot_ok), "drop_reason": ""})
    return rows


def patient_split(patients, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Assign every patient to train/val/test (deterministic, grouped by patient)."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(patients)))
    rng.shuffle(unique)
    n_train = int(round(fractions[0] * len(unique)))
    n_val = int(round(fractions[1] * len(unique)))
    names = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(unique) - n_train - n_val)
    return dict(zip(unique, names))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", required=True, help="folder with the raw PEDIS BMP images")
    parser.add_argument("--data-dir", default="data", help="folder with labels.csv; outputs go to <data-dir>/processed")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=0, help="process only the first N images (for testing)")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "processed")
    feet_dir = os.path.join(out_dir, "feet")
    os.makedirs(feet_dir, exist_ok=True)

    image_files = sorted(f for f in os.listdir(args.raw) if f.lower().endswith(".bmp"))
    if args.limit:
        image_files = image_files[:args.limit]
    worker = partial(process_image, raw_dir=args.raw, out_dir=feet_dir)
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, image_files, chunksize=64)
    manifest = pd.DataFrame([row for rows in results for row in rows])

    dropped = manifest[manifest.drop_reason != ""]
    feet = manifest[manifest.drop_reason == ""].drop(columns="drop_reason")

    # Attach patient ids and labels; images without a database record get no patient and are kept apart.
    labels_path = os.path.join(args.data_dir, "labels.csv")
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path)
        feet = feet.merge(labels, on="image_file", how="left")
        has_patient = feet.patient_id.notna()
        split_of = patient_split(feet.loc[has_patient, "patient_id"].astype(int))
        feet["split"] = feet.patient_id.map(lambda p: split_of.get(int(p)) if pd.notna(p) else "unassigned")

    feet.to_csv(os.path.join(out_dir, "feet.csv"), index=False)
    dropped[["image_file", "drop_reason"]].to_csv(os.path.join(out_dir, "dropped_images.csv"), index=False)

    print(f"{len(image_files)} raw images -> {len(feet)} foot crops in {feet_dir}", file=sys.stderr)
    print(f"dropped images: {len(dropped)}; foot crops passing QC: {feet.qc_ok.sum()} ({feet.qc_ok.mean():.1%})",
          file=sys.stderr)
    if "split" in feet:
        summary = feet[feet.qc_ok == 1].groupby("split").agg(crops=("foot_file", "size"),
                                                             patients=("patient_id", "nunique"))
        print(summary.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
