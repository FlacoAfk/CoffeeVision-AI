#!/usr/bin/env python3
"""
Build a weighted soft-voting ensemble from three trained models.

Loads the three trained models (custom_cnn, resnet50, efficientnetb0) for
the specified dataset, computes ensemble weights from validation metrics,
and saves a single end-to-end ensemble model in .keras format.

Usage:
    python build_ensemble.py --dataset leaf
    python build_ensemble.py --dataset grain
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class WeightedSoftVoting(layers.Layer):
    """Custom layer that computes weighted soft-voting of multiple model outputs.

    Weights are pre-computed from validation metrics and frozen during inference.
    """

    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.num_models = len(weights)
        # Store weights as a non-trainable constant
        self.w = self.add_weight(
            name="ensemble_weights",
            shape=(self.num_models,),
            initializer=lambda shape, dtype: tf.constant(
                weights, dtype=dtype
            ),
            trainable=False,
        )

    def call(self, inputs):
        """Compute weighted sum of model predictions.

        Args:
            inputs: List of Keras tensors, each shape (batch, num_classes)

        Returns:
            Weighted sum of probabilities, shape (batch, num_classes)
        """
        weighted = tf.math.accumulate_n(
            [inputs[i] * self.w[i] for i in range(self.num_models)]
        )
        return weighted

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_models": self.num_models,
                "weights": self.w.numpy().tolist(),
            }
        )
        return config


def compute_ensemble_weights(metrics_dir: str, dataset: str) -> list:
    """Compute ensemble weights from per-model validation metrics.

    Uses softmax-normalized validation accuracy as weights, giving more
    weight to models with better validation performance.

    Returns:
        List of three float weights (one per model, sum = 1.0)
    """
    metrics_path = Path(metrics_dir)
    model_keys = ["custom_cnn", "resnet50", "efficientnetb0"]
    accuracies = []

    for model_key in model_keys:
        metric_file = metrics_path / f"{dataset}_{model_key}_metrics.json"
        if metric_file.exists():
            with open(str(metric_file)) as f:
                metrics = json.load(f)
            acc = metrics.get("val_accuracy", 0.0)
            accuracies.append(acc)
            print(f"  {model_key}: val_accuracy = {acc:.4f}")
        else:
            print(
                f"  Warning: metrics file not found for {model_key}, using default weight",
                file=sys.stderr,
            )
            accuracies.append(0.5)

    # Softmax normalization
    acc_arr = np.array(accuracies, dtype=np.float32)
    # Subtract max for numerical stability
    exp_acc = np.exp(acc_arr - np.max(acc_arr))
    weights = exp_acc / np.sum(exp_acc)

    return weights.tolist()


def build_ensemble_model(
    cnn_path: str,
    resnet_path: str,
    effnet_path: str,
    weights: list,
    num_classes: int,
) -> keras.Model:
    """Build an end-to-end ensemble model wrapping the three sub-models.

    The ensemble takes a single image input, runs it through each of the
    three models with appropriate preprocessing, and produces a weighted
    soft-voting output.

    Args:
        cnn_path: Path to trained Custom CNN .keras file
        resnet_path: Path to trained ResNet50 .keras file
        effnet_path: Path to trained EfficientNetB0 .keras file
        weights: List of 3 float weights for soft-voting
        num_classes: Number of output classes

    Returns:
        Ensemble Keras model
    """
    print("Loading trained models...")

    # Load models with custom objects scope for WeightedSoftVoting
    # (in case they were saved as part of a previous ensemble)
    custom_objects = {"WeightedSoftVoting": WeightedSoftVoting}

    # Load sub-models
    cnn_model = keras.models.load_model(
        str(cnn_path), custom_objects=custom_objects
    )
    resnet_model = keras.models.load_model(
        str(resnet_path), custom_objects=custom_objects
    )
    effnet_model = keras.models.load_model(
        str(effnet_path), custom_objects=custom_objects
    )

    # Freeze all sub-model weights
    cnn_model.trainable = False
    resnet_model.trainable = False
    effnet_model.trainable = False

    # Build ensemble with Functional API
    image_input = keras.Input(
        shape=(224, 224, 3), name="ensemble_input"
    )

    # Each model uses its own preprocessing (built into the model graph)
    p1 = cnn_model(image_input, training=False)
    p2 = resnet_model(image_input, training=False)
    p3 = effnet_model(image_input, training=False)

    # Weighted soft-voting
    weighted = WeightedSoftVoting(weights)([p1, p2, p3])
    output = layers.Softmax(name="ensemble_output")(weighted)

    ensemble = keras.Model(
        inputs=image_input, outputs=output, name=f"ensemble_soft_voting"
    )

    return ensemble


def main():
    parser = argparse.ArgumentParser(
        description="Build weighted soft-voting ensemble from trained models."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["leaf", "grain"],
        help="Dataset type: leaf or grain",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./training/models",
        help="Directory containing trained model files (default: ./training/models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training/models",
        help="Directory to save the ensemble model (default: ./training/models)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected from models if not specified)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Ensemble Builder — {args.dataset}")
    print(f"{'='*60}")
    print(f"Models dir:  {models_dir.resolve()}")
    print(f"Output dir:  {output_dir.resolve()}")
    print()

    # Locate trained model files
    cnn_path = models_dir / f"{args.dataset}_custom_cnn.keras"
    resnet_path = models_dir / f"{args.dataset}_resnet50.keras"
    effnet_path = models_dir / f"{args.dataset}_efficientnetb0.keras"

    missing = []
    for name, path in [
        ("Custom CNN", cnn_path),
        ("ResNet50", resnet_path),
        ("EfficientNetB0", effnet_path),
    ]:
        if not path.exists():
            missing.append(f"  {name}: {path}")

    if missing:
        print(
            "Error: The following trained model files are missing:",
            file=sys.stderr,
        )
        for m in missing:
            print(m, file=sys.stderr)
        print(
            "\nPlease train all three models before building the ensemble.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compute ensemble weights from validation metrics
    print("Computing ensemble weights from validation metrics...")
    weights = compute_ensemble_weights(str(models_dir), args.dataset)
    print(f"\nEnsemble weights (softmax-normalized val_accuracy):")
    for model_key, w in zip(
        ["custom_cnn", "resnet50", "efficientnetb0"], weights
    ):
        print(f"  {model_key}: {w:.4f}")
    print(f"  Sum: {sum(weights):.4f}")

    # Determine num_classes from the first model's output layer
    if args.num_classes is None:
        temp_model = keras.models.load_model(str(cnn_path))
        num_classes = temp_model.output_shape[-1]
        # Clean up to free memory
        del temp_model
        keras.backend.clear_session()
        print(f"\nAuto-detected num_classes = {num_classes}")
    else:
        num_classes = args.num_classes

    # Build ensemble
    print("\nBuilding ensemble model...")
    strategy = tf.distribute.MirroredStrategy() if tf.config.list_physical_devices('GPU') else tf.distribute.get_strategy()
    with strategy.scope():
        ensemble = build_ensemble_model(
            str(cnn_path),
            str(resnet_path),
            str(effnet_path),
            weights,
            num_classes,
        )
        ensemble.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    ensemble.summary()

    # Save ensemble
    ensemble_path = output_dir / f"{args.dataset}_ensemble.keras"
    ensemble.save(str(ensemble_path))
    print(f"\nEnsemble saved to: {ensemble_path}")

    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "models": {
            "custom_cnn": str(cnn_path),
            "resnet50": str(resnet_path),
            "efficientnetb0": str(effnet_path),
        },
        "weights": {
            "custom_cnn": weights[0],
            "resnet50": weights[1],
            "efficientnetb0": weights[2],
        },
        "weight_source": "softmax(val_accuracy) from training metrics",
    }
    metadata_path = output_dir / f"{args.dataset}_ensemble_metadata.json"
    with open(str(metadata_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print(f"\n{'='*60}")
    print(f"Ensemble Build Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
