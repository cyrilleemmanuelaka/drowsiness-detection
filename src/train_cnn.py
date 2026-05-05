"""
CNN Model Definition & Training for Eye-State Classification
=============================================================
Trains a lightweight CNN to classify a 24x24 grayscale eye crop as
either OPEN or CLOSED. The model is then loaded by `detector.py`
to provide a deep-learning signal that complements the EAR heuristic.

Recommended dataset:
  - MRL Eye Dataset: http://mrl.cs.vsb.cz/eyedataset
  - Closed Eyes In The Wild (CEW): http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html

Expected directory layout:
    data/
      train/
        open/    *.png
        closed/  *.png
      val/
        open/    *.png
        closed/  *.png
"""

import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_cnn(input_shape=(24, 24, 1), num_classes=2) -> tf.keras.Model:
    """
    Lightweight CNN inspired by LeNet, tuned for tiny eye crops.
    ~80k parameters - runs at >100 FPS on CPU.
    """
    inputs = layers.Input(shape=input_shape, name="eye_input")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Convention: index 0 = closed, index 1 = open
    outputs = layers.Dense(num_classes, activation="softmax", name="eye_state")(x)

    model = models.Model(inputs, outputs, name="EyeStateCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_generators(data_dir: str, img_size=(24, 24), batch_size=64):
    """Train/val data generators with light augmentation."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        classes=["closed", "open"],  # forces label order
    )
    train_gen = train_aug.flow_from_directory(os.path.join(data_dir, "train"), shuffle=True, **common)
    val_gen = val_aug.flow_from_directory(os.path.join(data_dir, "val"), shuffle=False, **common)
    return train_gen, val_gen


def plot_history(history, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[INFO] Saved training curves -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the eye-state CNN")
    parser.add_argument("--data_dir", default="data", help="Root data folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="models/eye_state_cnn.h5")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("[INFO] Building model...")
    model = build_cnn()
    model.summary()

    print("[INFO] Loading data...")
    train_gen, val_gen = make_generators(args.data_dir, batch_size=args.batch_size)
    print(f"[INFO] Class indices: {train_gen.class_indices}")

    cb = [
        callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor="val_accuracy", mode="max"),
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("[INFO] Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=cb,
    )

    plot_history(history, "docs/training_curves.png")
    print(f"[INFO] Best model saved to {args.out}")


if __name__ == "__main__":
    main()
