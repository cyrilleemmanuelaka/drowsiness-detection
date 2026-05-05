"""
Generate Synthetic Demo Data
============================
Creates a small synthetic dataset of "open" and "closed" eye-like images
so the training and evaluation pipelines can be exercised end-to-end
WITHOUT downloading the MRL dataset.

This is for plumbing tests only - the real model should be trained
on real eye images. Synthetic patterns won't generalize to webcams.

Generates:
    data/{train,val,test}/{open,closed}/*.png
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw


def make_open_eye(size=24, rng=None):
    """A bright sclera with a dark iris/pupil."""
    rng = rng or np.random.default_rng()
    img = np.full((size, size), 220, dtype=np.uint8)
    img += rng.integers(-15, 15, img.shape, dtype=np.int8).astype(np.uint8)
    pil = Image.fromarray(img).convert("L")
    d = ImageDraw.Draw(pil)
    cx = size // 2 + rng.integers(-2, 3)
    cy = size // 2 + rng.integers(-1, 2)
    r_iris = rng.integers(5, 7)
    r_pupil = rng.integers(2, 4)
    d.ellipse((cx - r_iris, cy - r_iris, cx + r_iris, cy + r_iris), fill=70)
    d.ellipse((cx - r_pupil, cy - r_pupil, cx + r_pupil, cy + r_pupil), fill=10)
    # Eyelid lines
    d.arc((1, 0, size - 1, size - 2), start=200, end=340, fill=60, width=1)
    d.arc((1, 1, size - 1, size - 1), start=20, end=160, fill=60, width=1)
    return np.array(pil)


def make_closed_eye(size=24, rng=None):
    """A horizontal dark crease - no iris visible."""
    rng = rng or np.random.default_rng()
    img = np.full((size, size), 215, dtype=np.uint8)
    img += rng.integers(-15, 15, img.shape, dtype=np.int8).astype(np.uint8)
    pil = Image.fromarray(img).convert("L")
    d = ImageDraw.Draw(pil)
    y = size // 2 + rng.integers(-2, 3)
    # Dark slit
    d.line((2, y, size - 3, y + rng.integers(-1, 2)), fill=40, width=2)
    # Eyelashes hint
    for x in range(3, size - 3, 3):
        d.line((x, y - 1, x + rng.integers(-1, 2), y - 3), fill=30, width=1)
    return np.array(pil)


def make_split(out_root, n_per_class=200, size=24, seed=0):
    rng = np.random.default_rng(seed)
    splits = {"train": int(n_per_class * 0.7),
              "val":   int(n_per_class * 0.15),
              "test":  int(n_per_class * 0.15)}
    for split, n in splits.items():
        for label in ("open", "closed"):
            d = os.path.join(out_root, split, label)
            os.makedirs(d, exist_ok=True)
            maker = make_open_eye if label == "open" else make_closed_eye
            for i in range(n):
                img = maker(size=size, rng=rng)
                Image.fromarray(img).save(os.path.join(d, f"{label}_{i:04d}.png"))
        print(f"[OK] Built '{split}' split with {n} images per class.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--n", type=int, default=200, help="Images per class (before split)")
    args = p.parse_args()
    make_split(args.out, n_per_class=args.n)
    print(f"\nDone. Synthetic dataset is in: {os.path.abspath(args.out)}")
    print("⚠️  Reminder: this is plumbing-only. Train on MRL Eye Dataset for real results.")
