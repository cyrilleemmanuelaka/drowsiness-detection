"""
Organize MRL Eye Dataset into train/val/test splits.

Usage:
    python src/organize_dataset.py --source "C:/path/to/extracted/mrl" --dest data

The script:
  - Walks the source folder recursively
  - Finds image files (.png, .jpg) and reads the label from either:
      * the parent folder name ("open"/"closed", case-insensitive)
      * OR the MRL filename convention (s0001_NNNNN_X_Y_Z_W_V_S_C.png where Z = eye state, 0=closed, 1=open)
  - Splits 70% train / 15% val / 15% test
  - Copies into data/{train,val,test}/{open,closed}/
"""

import argparse
import os
import random
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# MRL convention: subjectID_imageID_gender_glasses_eyeState_reflections_lighting_sensorID
# Eye state is the 5th underscore-separated field (index 4).
MRL_PATTERN = re.compile(r"^s\d+_\d+_\d+_\d+_(\d+)_\d+_\d+_\d+", re.IGNORECASE)


def infer_label(path: Path) -> str | None:
    """Return 'open', 'closed', or None if we can't tell."""
    # First try: parent folder name
    parent = path.parent.name.lower()
    if "close" in parent:
        return "closed"
    if "open" in parent:
        return "open"

    # Second try: MRL filename pattern
    m = MRL_PATTERN.match(path.stem)
    if m:
        state = m.group(1)
        if state == "0":
            return "closed"
        if state == "1":
            return "open"
    return None


def collect_images(source: Path):
    """Return dict {label: [paths]} for all labelable images in source."""
    buckets = {"open": [], "closed": []}
    skipped = 0
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            label = infer_label(p)
            if label is None:
                skipped += 1
                continue
            buckets[label].append(p)
    return buckets, skipped


def split_and_copy(buckets, dest: Path, train_pct=0.70, val_pct=0.15, seed=42):
    rng = random.Random(seed)
    summary = {}
    for label, files in buckets.items():
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:],
        }
        for split, items in splits.items():
            target = dest / split / label
            target.mkdir(parents=True, exist_ok=True)
            for src in items:
                shutil.copy2(src, target / src.name)
            summary[(split, label)] = len(items)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Folder containing the extracted MRL dataset")
    p.add_argument("--dest", default="data", help="Destination folder (default: data)")
    p.add_argument("--clean", action="store_true",
                   help="If set, deletes the destination folder first.")
    args = p.parse_args()

    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()

    if not source.exists():
        raise SystemExit(f"❌ Source folder does not exist: {source}")

    if args.clean and dest.exists():
        print(f"[INFO] Cleaning {dest}")
        shutil.rmtree(dest)

    print(f"[INFO] Scanning {source} for eye images...")
    buckets, skipped = collect_images(source)
    print(f"[INFO] Found {len(buckets['open']):>6} open  eye images")
    print(f"[INFO] Found {len(buckets['closed']):>6} closed eye images")
    print(f"[INFO] Skipped {skipped} files (could not determine label)")

    if min(len(buckets['open']), len(buckets['closed'])) == 0:
        raise SystemExit(
            "❌ No labeled images found. Check that --source points to the extracted dataset.\n"
            "   Expected: parent folders named 'open'/'closed', OR MRL-style filenames."
        )

    print(f"\n[INFO] Splitting 70% train / 15% val / 15% test...")
    summary = split_and_copy(buckets, dest)

    print("\n=== Split summary ===")
    print(f"{'Split':<8} {'Open':>8} {'Closed':>8}")
    print("-" * 28)
    for split in ("train", "val", "test"):
        print(f"{split:<8} {summary[(split,'open')]:>8} {summary[(split,'closed')]:>8}")
    print(f"\n✅ Done. Dataset ready at: {dest}")
    print("   Next step:  python src/train_cnn.py --epochs 20")


if __name__ == "__main__":
    main()
