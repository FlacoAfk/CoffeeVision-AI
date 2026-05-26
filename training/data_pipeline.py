#!/usr/bin/env python3
"""
Data Pipeline — Load class-folder data, perform 70-15-15 split, compute class weights.

Accepts a --dataset flag (leaf|grain) to parameterize all output filenames.
Uses GroupShuffleSplit on the filename prefix (before UUID) to ensure all
augmentations of the same source image stay in the same split.

Usage:
    python data_pipeline.py --data-dir data/leaf --dataset leaf
    python data_pipeline.py --data-dir data/grain --dataset grain
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


def extract_group_key(filename: str) -> str:
    """
    Extract the group key from a filename.

    Removes the extension and UUID suffix so that all variations of the same
    source image (e.g. 'leaf_001_aug0.jpg', 'leaf_001_aug1.jpg') stay in the
    same split. UUID is identified as a 32-character hex string (with optional
    hyphens), or by splitting on the last underscore or hyphen before the UUID.
    """
    stem = Path(filename).stem  # remove extension

    # Pattern 1: trailing UUID (32 hex chars, possibly hyphenated)
    # e.g. "leaf_001_abc123def456abc123def456abc12345" -> "leaf_001"
    match = re.search(
        r"^(.+)_[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
        r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$",
        stem,
    )
    if match:
        return match.group(1)

    # Pattern 2: trailing short hash (8+ hex chars)
    match = re.search(r"^(.+)_[0-9a-fA-F]{8,}$", stem)
    if match:
        return match.group(1)

    # Pattern 3: split on last underscore, treat everything before as group
    if "_" in stem:
        return stem.rsplit("_", 1)[0]

    # Fallback: use the full stem as the group
    return stem


def load_dataset(data_dir: str) -> tuple:
    """
    Load images from class-folder structure.

    Expected structure:
        data_dir/
            class_1/
                img1.jpg
                img2.jpg
            class_2/
                img3.jpg
                ...

    Returns:
        filepaths: list of absolute image paths
        labels: list of class label strings (same order as filepaths)
        groups: list of group keys (same order)
        class_names: sorted list of unique class names
    """
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Find class folders (subdirectories containing images)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    class_folders = sorted(
        [d for d in data_path.iterdir() if d.is_dir()]
    )

    if not class_folders:
        print(
            f"Error: no class subdirectories found in {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    filepaths = []
    labels = []
    groups = []

    for class_folder in class_folders:
        class_name = class_folder.name
        image_files = sorted(
            [
                p
                for p in class_folder.iterdir()
                if p.suffix.lower() in image_extensions
            ]
        )
        for img_path in image_files:
            filepaths.append(str(img_path.resolve()))
            labels.append(class_name)
            groups.append(extract_group_key(img_path.name))

    if not filepaths:
        print(f"Error: no images found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    class_names = sorted(set(labels))
    return filepaths, labels, groups, class_names


def main():
    parser = argparse.ArgumentParser(
        description="Load class-folder data, perform 70-15-15 split, compute class weights."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the dataset directory with class subfolders",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["leaf", "grain"],
        help="Dataset type: 'leaf' or 'grain'",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for split files (default: <data-dir>/../splits/<dataset>)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation set size as fraction (default: 0.15)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set size as fraction (default: 0.15)",
    )
    args = parser.parse_args()

    # Setup output directory
    data_path = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent / "splits" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    filepaths, labels, groups, class_names = load_dataset(args.data_dir)
    total_images = len(filepaths)
    print(f"Total images:     {total_images}")
    print(f"Classes found:    {len(class_names)}")
    print(f"  {', '.join(class_names)}")

    # Compute initial class distribution
    class_dist = Counter(labels)
    print(f"\nClass distribution:")
    for cls in sorted(class_dist.keys()):
        print(f"  {cls}: {class_dist[cls]} ({class_dist[cls]/total_images*100:.1f}%)")

    # Calculate split sizes
    n_total = len(filepaths)
    # We split sequentially: first separate test (15%), then split remaining into train/val
    test_ratio = args.test_size
    val_ratio = args.val_size
    train_ratio = 1.0 - test_ratio - val_ratio

    print(f"\nPerforming {train_ratio:.0f}-{val_ratio:.0f}-{test_ratio:.0f} split...")

    # Step 1: Split off test set (15%)
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=args.random_state,
    )
    trainval_idx, test_idx = next(
        gss_test.split(filepaths, labels, groups)
    )

    # Step 2: Split remaining into train (70% of total = ~82.35% of trainval) and val (15% of total)
    trainval_filepaths = [filepaths[i] for i in trainval_idx]
    trainval_labels = [labels[i] for i in trainval_idx]
    trainval_groups = [groups[i] for i in trainval_idx]

    # The test size for the second split should make val = 15% of total
    # Since trainval = 85% of total: test_size_in_split = val_ratio / (train_ratio + val_ratio)
    split2_test_size = val_ratio / (train_ratio + val_ratio)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=split2_test_size,
        random_state=args.random_state,
    )
    train_idx, val_idx = next(
        gss_val.split(trainval_filepaths, trainval_labels, trainval_groups)
    )

    # Map indices back to original filepaths
    train_files = [trainval_filepaths[i] for i in train_idx]
    train_labels = [trainval_labels[i] for i in train_idx]
    val_files = [trainval_filepaths[i] for i in val_idx]
    val_labels = [trainval_labels[i] for i in val_idx]
    test_files = [filepaths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # Verify no overlap in groups
    train_groups = set(trainval_groups[i] for i in train_idx)
    val_groups = set(trainval_groups[i] for i in val_idx)
    test_groups = set(groups[i] for i in test_idx)
    assert len(train_groups & val_groups) == 0, "Group leakage between train and val!"
    assert len(train_groups & test_groups) == 0, "Group leakage between train and test!"
    assert len(val_groups & test_groups) == 0, "Group leakage between val and test!"

    # Compute class weights (using training set only)
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_label_indices = [label_to_idx[l] for l in train_labels]
    class_weight_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.array(train_label_indices),
    )
    class_weights = {
        class_names[i]: float(class_weight_array[i])
        for i in range(len(class_names))
    }

    # Verify split sizes
    n_train = len(train_files)
    n_val = len(val_files)
    n_test = len(test_files)
    total_split = n_train + n_val + n_test

    print(f"\n{'='*50}")
    print(f"Split Results — {args.dataset}")
    print(f"{'='*50}")
    print(f"  Train:  {n_train:5d} ({n_train/total_split*100:.1f}%)")
    print(f"  Val:    {n_val:5d} ({n_val/total_split*100:.1f}%)")
    print(f"  Test:   {n_test:5d} ({n_test/total_split*100:.1f}%)")
    print(f"  Total:  {total_split:5d}")
    print(f"\nClass weights (inverse frequency):")
    for cls in class_names:
        print(f"  {cls}: {class_weights[cls]:.4f}")

    # Save split CSVs
    def save_csv(file_list, label_list, output_path):
        with open(output_path, "w", newline="") as f:
            f.write("filepath,label\n")
            for fp, lbl in zip(file_list, label_list):
                f.write(f"{fp},{lbl}\n")

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    weights_json = output_dir / "class_weights.json"

    save_csv(train_files, train_labels, str(train_csv))
    save_csv(val_files, val_labels, str(val_csv))
    save_csv(test_files, test_labels, str(test_csv))

    with open(str(weights_json), "w") as f:
        json.dump(class_weights, f, indent=2)

    print(f"\nSaved splits to:")
    print(f"  Train CSV:      {train_csv}")
    print(f"  Val CSV:        {val_csv}")
    print(f"  Test CSV:       {test_csv}")
    print(f"  Class weights:  {weights_json}")

    # Summary per split
    print(f"\nPer-split class distribution:")
    for split_name, split_labels in [
        ("Train", train_labels),
        ("Val", val_labels),
        ("Test", test_labels),
    ]:
        split_dist = Counter(split_labels)
        dist_str = ", ".join(
            f"{cls}: {split_dist[cls]}" for cls in sorted(class_names)
        )
        print(f"  {split_name}: {dist_str}")


if __name__ == "__main__":
    main()
