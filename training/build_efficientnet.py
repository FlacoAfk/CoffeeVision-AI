#!/usr/bin/env python3
"""
Build and train an EfficientNetB0 transfer learning model for coffee leaf/grain classification.

Architecture:
    Base: EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    → GlobalAveragePooling2D
    → Dense(256, relu) → Dropout(0.5)
    → Dense(num_classes, softmax)

Uses tf.keras.applications.efficientnet.preprocess_input for data preprocessing.

Usage:
    python build_efficientnet.py --num-classes 15 --dataset leaf
    python build_efficientnet.py --num-classes 3 --dataset grain
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications


def build_efficientnet(num_classes: int) -> keras.Model:
    """Build EfficientNetB0 transfer learning model with frozen base."""
    base = applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False  # Freeze base layers

    inputs = keras.Input(shape=(224, 224, 3))
    # EfficientNet's preprocess_input will scale pixels appropriately
    x = applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="efficientnetb0"
    )
    return model


def load_datasets(data_dir: str, dataset: str, batch_size: int):
    """Load training and validation datasets from split CSVs.

    Keeps images as uint8 [0, 255] for efficientnet.preprocess_input.
    """
    splits_dir = Path(data_dir).parent / "splits" / dataset

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    weights_json = splits_dir / "class_weights.json"

    for required in [train_csv, val_csv]:
        if not required.exists():
            print(
                f"Error: required file not found: {required}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load CSVs
    train_df = pd.read_csv(str(train_csv))
    val_df = pd.read_csv(str(val_csv))

    # Build label index
    all_labels = sorted(
        set(pd.concat([train_df["label"], val_df["label"]]))
    )
    num_classes = len(all_labels)
    label_to_idx = {name: i for i, name in enumerate(all_labels)}

    # Convert to arrays
    train_files = train_df["filepath"].values
    train_labels = np.array(
        [label_to_idx[l] for l in train_df["label"]]
    )
    val_files = val_df["filepath"].values
    val_labels = np.array([label_to_idx[l] for l in val_df["label"]])

    # Load class weights
    class_weights = None
    if weights_json.exists():
        with open(str(weights_json)) as f:
            class_weights_dict = json.load(f)
        class_weights = {
            label_to_idx[name]: weight
            for name, weight in class_weights_dict.items()
        }
        print(f"Loaded class weights for {len(class_weights)} classes")

    # tf.data pipeline
    # EfficientNet preprocess_input expects [0, 255] uint8 images
    def load_image(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        # Keep as uint8 [0, 255] for preprocess_input
        img = tf.cast(img, tf.uint8)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_files, train_labels)
    )
    train_ds = train_ds.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.shuffle(buffer_size=min(1000, len(train_files)))
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_weights


def main():
    parser = argparse.ArgumentParser(
        description="Build and train EfficientNetB0 transfer learning for leaf/grain classification."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=15,
        help="Number of output classes (default: 15 for leaf, 3 for grain)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="leaf",
        choices=["leaf", "grain"],
        help="Dataset type: leaf or grain (default: leaf)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./training/data",
        help="Path to the dataset class-folder directory (default: ./training/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training/models",
        help="Directory to save trained models (default: ./training/models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4)",
    )
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"EfficientNetB0 Transfer Learning — {args.dataset}")
    print(f"{'='*60}")
    print(f"Num classes:     {args.num_classes}")
    print(f"Dataset:         {args.dataset}")
    print(f"Data dir:        {data_dir.resolve()}")
    print(f"Output dir:      {output_dir.resolve()}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Epochs:          {args.epochs}")
    print(f"Learning rate:   {args.learning_rate}")
    print()

    # Build model
    print("Building EfficientNetB0 model...")
    strategy = tf.distribute.MirroredStrategy() if tf.config.list_physical_devices('GPU') else tf.distribute.get_strategy()
    with strategy.scope():
        model = build_efficientnet(args.num_classes)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    model.summary()

    # Load data
    splits_dir = data_dir.parent / "splits" / args.dataset
    print(f"\nLoading datasets from {splits_dir}...")
    train_ds, val_ds, class_weights = load_datasets(
        str(data_dir), args.dataset, args.batch_size
    )

    # Callbacks
    model_name = f"{args.dataset}_efficientnetb0"
    checkpoint_path = output_dir / f"{model_name}_checkpoint.keras"
    best_model_path = output_dir / f"{model_name}.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(output_dir / f"{model_name}_training_log.csv")
        ),
    ]

    # Train
    print(f"\nStarting training ({args.epochs} epochs max)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(str(best_model_path))
    print(f"\nModel saved to: {best_model_path}")

    # Save training metrics
    val_acc = max(history.history["val_accuracy"])
    val_loss = min(history.history["val_loss"])
    metrics = {
        "dataset": args.dataset,
        "model": "efficientnetb0",
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "epochs_trained": len(history.history["val_accuracy"]),
    }
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Best val_accuracy: {val_acc:.4f}")
    print(f"Best val_loss:     {val_loss:.4f}")


if __name__ == "__main__":
    main()
