"""Load the training, validation and test sets (inputs and outputs) of the foot crop dataset.

Works with both layouts:
    downloaded release   data/pedis_feet_release_v1/{splits,feet}   (src/download_data.py)
    local pipeline       data/release/splits + data/processed/feet  (src/export_dataset.py)

Example:
    from dataset import load_split
    X, y = load_split("train", limit=1000)   # X: (N, 224, 96) uint8, y: DataFrame of targets
"""

import os

import numpy as np
import pandas as pd
from PIL import Image

RELEASE_NAME = "pedis_feet_release_v1"
TARGETS = ["age", "weight_kg", "shoe_size"]
# Per-pixel statistics of the quality-controlled crops (scaled to [0, 1]), for input normalisation.
PIXEL_MEAN, PIXEL_STD = 0.407, 0.292


def locate(data_dir="data"):
    """Return (splits_dir, feet_dir) of whichever dataset layout is present."""
    candidates = [
        (os.path.join(data_dir, RELEASE_NAME, "splits"), os.path.join(data_dir, RELEASE_NAME, "feet")),
        (os.path.join(data_dir, "release", "splits"), os.path.join(data_dir, "processed", "feet")),
    ]
    for splits_dir, feet_dir in candidates:
        if os.path.isdir(splits_dir) and os.path.isdir(feet_dir):
            return splits_dir, feet_dir
    raise FileNotFoundError(f"No dataset under {data_dir!r}: run src/download_data.py first "
                            "(or the local pipeline, see README.md).")


def load_table(split, data_dir="data"):
    """The split table: one row per foot crop with its file name and targets."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val' or 'test'")
    splits_dir, _ = locate(data_dir)
    return pd.read_csv(os.path.join(splits_dir, f"{split}.csv"))


def load_split(split, data_dir="data", targets=TARGETS, limit=None, seed=0):
    """Return (X, y): the crops as a uint8 array (N, 224, 96) and the target columns.

    limit picks a reproducible random subset (useful for quick experiments).
    Missing target values are kept as NaN; mask them in the loss.
    """
    table = load_table(split, data_dir)
    if limit and limit < len(table):
        table = table.sample(limit, random_state=seed).reset_index(drop=True)
    _, feet_dir = locate(data_dir)
    X = np.stack([np.asarray(Image.open(os.path.join(feet_dir, name)), dtype=np.uint8)
                  for name in table.foot_file])
    return X, table[list(targets)].reset_index(drop=True)


def normalise(X):
    """uint8 crops -> float32 network input with zero mean and unit variance."""
    return ((X.astype(np.float32) / 255.0) - PIXEL_MEAN) / PIXEL_STD
