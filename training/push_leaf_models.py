#!/usr/bin/env python3
"""
Generate Kaggle notebooks for leaf model training and push them to Kaggle.

Creates .ipynb notebooks from each training script, configures them for
Kaggle GPU execution, and pushes them to the julianmedinamonje45 account.

Three notebooks are generated:
    1. coffevision-leaf-cnn-train-v1        — Custom CNN training
    2. coffevision-leaf-resnet50-train-v1   — ResNet50 transfer learning
    3. coffevision-leaf-efficientnet-train-v1 — EfficientNetB0 transfer learning

Usage:
    python push_leaf_models.py
    python push_leaf_models.py --source-dir ./training --kaggle-account julianmedinamonje45
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Notebook template for Kaggle training
NOTEBOOK_TEMPLATE = """\
{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# CoffeeVision AI — {model_name}\\n",
    "\\n",
    "Training {model_type} for leaf disease classification ({num_classes} classes).\\n",
    "Dataset: Roboflow Coffee Leaf Disease (Roya dataset, ~5000 images).\\n",
    "\\n",
    "**Notebook Slug**: `{slug}`\\n",
    "**Author**: `{kaggle_account}`"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{
    "id": "install-deps"
   }},
   "source": [
    "# Cell 1: Setup — Install dependencies\\n",
    "!pip install --quiet tensorflow pandas numpy scikit-learn pillow opencv-python\\n",
    "\\n",
    "import os\\n",
    "import sys\\n",
    "import json\\n",
    "from pathlib import Path\\n",
    "\\n",
    "import tensorflow as tf\\n",
    "from tensorflow import keras\\n",
    "\\n",
    "# Verify GPU\\n",
    "gpus = tf.config.list_physical_devices('GPU')\\n",
    "print(f'GPU available: {{len(gpus) > 0}}')\\n",
    "if gpus:\\n",
    "    for gpu in gpus:\\n",
    "        print(f'  {{gpu}}')"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{
    "id": "download-data"
   }},
   "source": [
    "# Cell 2: Download dataset from Roboflow\\n",
    "# Replace with actual Roboflow download URL for the Roya dataset\\n",
    "# Dataset source: Coffee Leaf Disease (Roya) — ~5000 images, 15 classes\\n",
    "\\n",
    "ROBOFLOW_URL = \\"{roboflow_url}\\"\\n",
    "DATA_DIR = Path('/kaggle/working/data')\\n",
    "\\n",
    "if not DATA_DIR.exists():\\n",
    "    print('Downloading dataset from Roboflow...')\\n",
    "    os.makedirs(str(DATA_DIR), exist_ok=True)\\n",
    "    !curl -L \\\"$ROBOFLOW_URL\\\" -o /kaggle/working/roboflow.zip\\n",
    "    !unzip -q /kaggle/working/roboflow.zip -d /kaggle/working/roboflow_raw/\\n",
    "    print('Dataset downloaded and extracted.')"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{
    "id": "convert-data"
   }},
   "source": [
    "# Cell 3: Convert OD to classification format\\n",
    "import sys\\n",
    "sys.path.insert(0, '/kaggle/working')\\n",
    "\\n",
    "# Run convert_roboflow.py\\n",
    "!python /kaggle/working/convert_roboflow.py \\\\\\n",
    "    --zip /kaggle/working/roboflow.zip \\\\\\n",
    "    --output /kaggle/working/data/leaf\\n",
    "\\n",
    "# Run data_pipeline.py\\n",
    "!python /kaggle/working/data_pipeline.py \\\\\\n",
    "    --data-dir /kaggle/working/data/leaf \\\\\\n",
    "    --dataset leaf \\\\\\n",
    "    --output-dir /kaggle/working/splits/leaf"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{
    "id": "train-model"
   }},
   "source": [
    "# Cell 4: Train the model\\n",
    "print('Starting training...')\\n",
    "\\n",
    "!python /kaggle/working/{script_name} \\\\\\n",
    "    --num-classes {num_classes} \\\\\\n",
    "    --dataset leaf \\\\\\n",
    "    --data-dir /kaggle/working/data/leaf \\\\\\n",
    "    --output-dir /kaggle/working/models/leaf\\n",
    "\\n",
    "print('Training complete.')"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{
    "id": "verify"
   }},
   "source": [
    "# Cell 5: Verify trained model\\n",
    "MODEL_PATH = f'/kaggle/working/models/leaf/leaf_{model_key}.keras'\\n",
    "\\n",
    "if os.path.exists(MODEL_PATH):\\n",
    "    model = keras.models.load_model(MODEL_PATH)\\n",
    "    model.summary()\\n",
    "    print(f'Model loaded successfully: {{MODEL_PATH}}')\\n",
    "    print(f'Input shape: {{model.input_shape}}')\\n",
    "    print(f'Output shape: {{model.output_shape}}')\\n",
    "else:\\n",
    "    print(f'Error: Model not found at {{MODEL_PATH}}')\\n",
    "    print('Check training logs for errors.')"
   ]
  }}
 ],
 "metadata": {{
  "kaggle": {{
   "accelerator": "{accelerator}",
   "dataset_sources": [],
   "enable_gpu": {enable_gpu},
   "enable_internet": true,
   "kernel_sources": [],
   "language": "python",
   "language_info": {{"name": "python", "version": "{python_version}"}}
  }},
  "kaggle_sessions": [],
  "language_info": {{
   "name": "python",
   "version": "{python_version}"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
"""


def generate_notebook(
    script_name: str,
    model_name: str,
    model_type: str,
    model_key: str,
    slug: str,
    num_classes: int,
    roboflow_url: str,
    kaggle_account: str,
    python_version: str = "3.10",
    accelerator: str = "GPU",
):
    """Generate a Kaggle-compatible .ipynb notebook from a training script.

    Args:
        script_name: The Python training script filename (e.g., 'build_custom_cnn.py')
        model_name: Display name for the notebook (e.g., 'Custom CNN')
        model_type: Short description (e.g., 'Custom CNN')
        model_key: Key used in model filenames (e.g., 'custom_cnn')
        slug: Kaggle notebook slug (e.g., 'coffevision-leaf-cnn-train-v1')
        num_classes: Number of output classes
        roboflow_url: URL to download the Roboflow dataset
        kaggle_account: Kaggle account username
        python_version: Python version for Kaggle environment
        accelerator: Kaggle accelerator type ('GPU' or 'TPU')

    Returns:
            Notebook content as a JSON string
    """
    enable_gpu = accelerator.upper() == "GPU"
    content = NOTEBOOK_TEMPLATE.format(
        model_name=model_name,
        model_type=model_type,
        model_key=model_key,
        slug=slug,
        num_classes=num_classes,
        roboflow_url=roboflow_url,
        script_name=script_name,
        kaggle_account=kaggle_account,
        python_version=python_version,
        accelerator=accelerator,
        enable_gpu=str(enable_gpu).lower(),
    )
    return content


def push_to_kaggle(
    notebook_path: str,
    slug: str,
    kaggle_account: str,
    kgat_token: str = None,
):
    """Push a notebook to Kaggle using the Kaggle API.

    Args:
        notebook_path: Path to the .ipynb notebook file
        slug: Kaggle notebook slug (without account prefix)
        kaggle_account: Kaggle account username
        kgat_token: KGAT token for authentication (optional, uses env var if not set)
    """
    # Set KGAT token if provided
    env = os.environ.copy()
    if kgat_token:
        env["KAGGLE_KEY"] = kgat_token

    full_slug = f"{kaggle_account}/{slug}"

    # Create metadata JSON required by Kaggle API
    metadata = {
        "id": full_slug,
        "title": slug.replace("-", " ").title(),
        "subtitle": f"CoffeeVision AI — {slug}",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "language": "python",
        "kernel_type": "notebook",
    }

    metadata_path = notebook_path.replace(".ipynb", "-kernel-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Push via Kaggle API
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "kernels",
                "push",
                "--kernel",
                notebook_path,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"  ✓ Successfully pushed: {full_slug}")
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    print(f"    {line}")
        else:
            print(f"  ✗ Failed to push {full_slug}", file=sys.stderr)
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"    {line}", file=sys.stderr)
            return False
    except FileNotFoundError:
        print(
            "  ✗ Kaggle CLI not found. Install with: pip install kaggle",
            file=sys.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        print(
            f"  ✗ Timeout pushing {full_slug} (may still be processing)",
            file=sys.stderr,
        )
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate and push leaf model training notebooks to Kaggle."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="./training",
        help="Directory containing training scripts (default: ./training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training/notebooks",
        help="Directory to save generated notebooks (default: ./training/notebooks)",
    )
    parser.add_argument(
        "--kaggle-account",
        type=str,
        default="julianmedinamonje45",
        help="Kaggle account username (default: julianmedinamonje45)",
    )
    parser.add_argument(
        "--kgat-token",
        type=str,
        default=None,
        help="KGAT token for Kaggle API authentication (default: KAGGLE_KEY env var)",
    )
    parser.add_argument(
        "--roboflow-url",
        type=str,
        default="https://universe.roboflow.com/ds/coffee-leaf-disease?key=REPLACE_ME",
        help="Roboflow dataset download URL (default: placeholder)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=15,
        help="Number of leaf disease classes (default: 15)",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default="3.10",
        choices=["3.10", "3.12"],
        help="Python version for Kaggle environment (default: 3.10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate notebooks but skip Kaggle push",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Push Leaf Models to Kaggle")
    print(f"{'='*60}")
    print(f"Account:       {args.kaggle_account}")
    print(f"Source dir:    {source_dir.resolve()}")
    print(f"Output dir:    {output_dir.resolve()}")
    print(f"Num classes:   {args.num_classes}")
    print(f"Python:        {args.python_version}")
    print(f"Dry run:       {args.dry_run}")
    print()

    # Define the three leaf model notebooks
    notebooks = [
        {
            "script": "build_custom_cnn.py",
            "name": "CoffeeVision Leaf Custom CNN",
            "type": "Custom CNN Architecture",
            "key": "custom_cnn",
            "slug": "coffevision-leaf-cnn-train-v1",
        },
        {
            "script": "build_resnet50.py",
            "name": "CoffeeVision Leaf ResNet50",
            "type": "ResNet50 Transfer Learning",
            "key": "resnet50",
            "slug": "coffevision-leaf-resnet50-train-v1",
        },
        {
            "script": "build_efficientnet.py",
            "name": "CoffeeVision Leaf EfficientNet",
            "type": "EfficientNetB0 Transfer Learning",
            "key": "efficientnetb0",
            "slug": "coffevision-leaf-efficientnet-train-v1",
        },
    ]

    # Verify source scripts exist
    print("Checking source scripts...")
    for nb in notebooks:
        script_path = source_dir / nb["script"]
        if not script_path.exists():
            print(
                f"  Warning: {script_path} not found. Notebook may not work.",
                file=sys.stderr,
            )
        else:
            print(f"  ✓ {nb['script']} found")
    print()

    # Generate notebooks
    generated = []
    print("Generating notebooks...")
    for nb in notebooks:
        notebook_content = generate_notebook(
            script_name=nb["script"],
            model_name=nb["name"],
            model_type=nb["type"],
            model_key=nb["key"],
            slug=nb["slug"],
            num_classes=args.num_classes,
            roboflow_url=args.roboflow_url,
            kaggle_account=args.kaggle_account,
            python_version=args.python_version,
            accelerator="GPU",
        )

        notebook_path = output_dir / f"{nb['slug']}.ipynb"
        with open(str(notebook_path), "w") as f:
            f.write(notebook_content)

        generated.append(notebook_path)
        print(f"  ✓ {notebook_path.name}")

    print(f"\n{len(generated)} notebooks generated in {output_dir.resolve()}")

    # Push to Kaggle
    if args.dry_run:
        print(f"\n{'='*60}")
        print("Dry run — skipping Kaggle push.")
        print("Run without --dry-run to push to Kaggle.")
        print(f"{'='*60}")
        return

    print(f"\nPushing to Kaggle ({args.kaggle_account})...")
    kgat_token = args.kgat_token or os.environ.get("KAGGLE_KEY")
    if not kgat_token:
        print(
            "Warning: No KGAT token provided. Set KAGGLE_KEY env var or use --kgat-token.",
            file=sys.stderr,
        )
        print("Continuing with any available Kaggle credentials...")
        print()

    success_count = 0
    for nb in notebooks:
        notebook_path = output_dir / f"{nb['slug']}.ipynb"
        print(f"Pushing {nb['slug']}...")
        ok = push_to_kaggle(
            str(notebook_path),
            nb["slug"],
            args.kaggle_account,
            kgat_token,
        )
        if ok:
            success_count += 1
        print()

    print(f"{'='*60}")
    print(f"Push Results")
    print(f"{'='*60}")
    print(f"Successful:  {success_count}/{len(notebooks)}")
    print(f"Account:     {args.kaggle_account}")
    print(f"Dataset:     Leaf (Roya, {args.num_classes} classes)")
    if success_count < len(notebooks):
        print(
            f"Failed:      {len(notebooks) - success_count} — check errors above",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
