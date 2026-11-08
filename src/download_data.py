"""Download and unpack the de-identified PEDIS foot crop dataset.

The raw PEDIS data contains personal data and is not public, so this project
distributes a de-identified release (built by src/export_dataset.py): the
preprocessed foot crops, the train/validation/test split files and the targets.

The release is a single zip file hosted on Google Drive (or any HTTP URL).
Pass its share link with --url, or set the PEDIS_DATA_URL environment variable,
or fill in DEFAULT_URL below.

Usage:
    python src/download_data.py --url "<share link>" [--out data]

After it finishes, the data is in data/pedis_feet_release_v1/:
    feet/*.png, feet.csv, images.csv, splits/{train,val,test}.csv, DATA_README.md
"""

import argparse
import hashlib
import os
import re
import sys
import urllib.request
import zipfile

RELEASE_NAME = "pedis_feet_release_v1"
DEFAULT_URL = "https://drive.google.com/file/d/1ncvbMDye32V2arcB1tQaY-E4ekOCoNGw/view?usp=drive_link"      # Google Drive share link of pedis_feet_release_v1.zip
EXPECTED_SHA256 = "2e8fd9157aecbb73ea67d35ea8c3988864302dfbf40730f5e6d869b6ec8e0fa6"  # of release v1, printed by src/export_dataset.py


def drive_file_id(url):
    """Return the Google Drive file id of a share link, or None for other URLs."""
    match = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=download&)?id=)([\w-]+)", url)
    return match.group(1) if match else None


def download(url, path):
    file_id = drive_file_id(url)
    if file_id:
        try:
            import gdown  # preinstalled on Colab; handles the large-file confirmation page
        except ImportError:
            sys.exit("Google Drive links need gdown: pip install gdown")
        gdown.download(id=file_id, output=path, quiet=False)
    else:
        print(f"downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=os.environ.get("PEDIS_DATA_URL", DEFAULT_URL),
                        help="share link of the release zip (default: $PEDIS_DATA_URL)")
    parser.add_argument("--out", default="data", help="folder to unpack into")
    parser.add_argument("--keep-zip", action="store_true", help="do not delete the zip after unpacking")
    args = parser.parse_args()

    target = os.path.join(args.out, RELEASE_NAME)
    if os.path.exists(os.path.join(target, "splits", "train.csv")):
        print(f"already present: {target}", file=sys.stderr)
        return
    if not args.url:
        sys.exit("No download link. Ask the team for the link of pedis_feet_release_v1.zip and pass it "
                 "with --url or the PEDIS_DATA_URL environment variable.")

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, f"{RELEASE_NAME}.zip")
    download(args.url, zip_path)
    if EXPECTED_SHA256 and sha256(zip_path) != EXPECTED_SHA256:
        sys.exit(f"checksum mismatch for {zip_path}; delete it and download again")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    if not args.keep_zip:
        os.remove(zip_path)
    print(f"dataset ready in {target}", file=sys.stderr)


if __name__ == "__main__":
    main()
