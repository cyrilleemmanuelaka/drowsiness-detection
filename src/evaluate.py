"""
Evaluate the trained eye-state CNN on a held-out test set.

Outputs:
  - Classification report (precision, recall, F1)
  - Confusion matrix figure
  - ROC curve & AUC
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(PROJECT_ROOT / "models" / "eye_state_cnn.h5"))
    p.add_argument("--test_dir", default=str(PROJECT_ROOT / "data" / "test"))
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "docs"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading {args.model}")
    model = load_model(args.model)

    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        args.test_dir,
        target_size=(24, 24),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=64,
        shuffle=False,
        classes=["closed", "open"],
    )

    print("[INFO] Predicting...")
    preds = model.predict(gen, verbose=1)
    y_true = gen.classes
    y_pred = np.argmax(preds, axis=1)
    y_score = preds[:, 1]  # P(open)

    # ---- Report
    target_names = ["closed", "open"]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n=== Classification Report ===")
    print(report)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # ---- Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - Eye State CNN")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    print(f"[INFO] Saved {cm_path}")

    # ---- ROC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title("ROC Curve - Eye State CNN")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.out_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=120)
    print(f"[INFO] Saved {roc_path}")
    print(f"[INFO] AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
