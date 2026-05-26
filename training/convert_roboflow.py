#!/usr/bin/env python3
"""
Convert Roboflow Object Detection ZIP to Classification Folder Structure.

Parses a Roboflow OD zip archive containing _annotations.csv, identifies
the dominant bounding box per image (largest area), and copies each image
into output/<class_name>/ folders for use as a classification dataset.

Usage:
    python convert_roboflow.py --zip path/to/roboflow.zip --output data/leaf
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_annotations(csv_path: str) -> dict:
    """
    Parse _annotations.csv and return mapping of filename -> class_name for
    the dominant (largest area) bounding box per image.

    Handles the common Roboflow YOLO OD CSV format where columns may have
    duplicate names (e.g. "width,height" for both image and bbox dimensions).
    """
    # Read raw header to detect format
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        raw_header = next(reader)

    raw_header = [h.strip() for h in raw_header]
    num_cols = len(raw_header)

    # Map column indices to canonical names
    # Common Roboflow YOLO OD format:
    #   filename, width, height, class, x_center, y_center, bbox_width, bbox_height
    # (first width/height are image dimensions, second pair are bbox)
    col_index = {}

    if num_cols >= 4:
        col_index["filename"] = 0
        # Columns 1,2 = image width/height (skip for dominant bbox logic)
        col_index["class"] = 3

    if num_cols == 8:
        # YOLO format: filename, img_w, img_h, class, x_center, y_center, bbox_w, bbox_h
        col_index["bbox_width"] = 6
        col_index["bbox_height"] = 7
    elif num_cols == 9:
        # With confidence: filename, img_w, img_h, class, x_center, y_center, bbox_w, bbox_h, conf
        col_index["bbox_width"] = 6
        col_index["bbox_height"] = 7
        col_index["confidence"] = 8
    else:
        # Fallback: try to detect by column name patterns with duplicate handling
        seen = {}
        for i, name in enumerate(raw_header):
            name_lower = name.lower()
            if name_lower in seen:
                seen[name_lower] += 1
                suffix = seen[name_lower]
                if name_lower in ("width", "height"):
                    col_index[f"bbox_{name_lower}"] = i
                else:
                    col_index[f"{name_lower}_{suffix}"] = i
            else:
                seen[name_lower] = 0
                col_index[name_lower] = i

    # Read all rows into a structured list
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or not row[0].strip():
                continue
            entry = {
                "filename": row[col_index["filename"]].strip(),
                "class": row[col_index["class"]].strip(),
            }
            if "bbox_width" in col_index and "bbox_height" in col_index:
                entry["bbox_width"] = float(row[col_index["bbox_width"]])
                entry["bbox_height"] = float(row[col_index["bbox_height"]])
            else:
                # Default area = 1.0 if no bbox dimensions (all equal weight)
                entry["bbox_width"] = 1.0
                entry["bbox_height"] = 1.0

            if "confidence" in col_index:
                entry["confidence"] = float(row[col_index["confidence"]])
            else:
                entry["confidence"] = 1.0

            rows.append(entry)

    # Find dominant bbox per image by area
    dominant = {}  # filename -> (class_name, area)
    for entry in rows:
        filename = entry["filename"]
        area = entry["bbox_width"] * entry["bbox_height"]
        if filename not in dominant or area > dominant[filename][1]:
            dominant[filename] = (entry["class"], area)

    return {fname: cls for fname, (cls, _) in dominant.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Convert Roboflow OD ZIP to classification folder structure."
    )
    parser.add_argument(
        "--zip", required=True, help="Path to the Roboflow OD ZIP archive"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for class folders"
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    output_path = Path(args.output)

    if not zip_path.exists():
        print(f"Error: ZIP file not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    # Temporary extraction directory
    with tempfile.TemporaryDirectory(prefix="roboflow_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(tmp_path))

        # Find _annotations.csv
        csv_files = list(tmp_path.rglob("_annotations.csv"))
        if not csv_files:
            # Try alternate naming patterns
            csv_files = list(tmp_path.rglob("*annotations*.csv"))
        if not csv_files:
            print("Error: _annotations.csv not found in ZIP archive.", file=sys.stderr)
            sys.exit(1)

        csv_path = csv_files[0]
        print(f"Found annotations: {csv_path}")

        # Parse annotations to get dominant class per image
        image_class_map = parse_annotations(str(csv_path))
        print(f"Parsed {len(image_class_map)} images with annotations.")

        # Collect class counts
        class_counts = defaultdict(int)
        copied_count = 0
        skipped_count = 0

        # Find all image files in the extraction
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
        for img_path in tmp_path.rglob("*"):
            if img_path.suffix.lower() not in image_extensions:
                continue
            if "_annotations" in img_path.name.lower():
                continue

            rel_name = img_path.name

            if rel_name in image_class_map:
                class_name = image_class_map[rel_name]
                dest_dir = output_path / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(img_path), str(dest_dir / rel_name))
                class_counts[class_name] += 1
                copied_count += 1
            else:
                # Try matching without the annotations.csv prefix
                # Some Roboflow zips include subfolder prefix in CSV filenames
                found = False
                for csv_fname, cls_name in image_class_map.items():
                    if rel_name.endswith(csv_fname) or csv_fname.endswith(rel_name):
                        dest_dir = output_path / cls_name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(img_path), str(dest_dir / rel_name))
                        class_counts[cls_name] += 1
                        copied_count += 1
                        found = True
                        break
                if not found:
                    skipped_count += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"Conversion Complete")
    print(f"{'='*50}")
    print(f"Images copied:     {copied_count}")
    print(f"Images skipped:    {skipped_count}")
    print(f"Classes found:     {len(class_counts)}")
    print(f"\nPer-class breakdown:")
    for cls_name in sorted(class_counts.keys()):
        print(f"  {cls_name}: {class_counts[cls_name]} images")
    print(f"\nOutput directory: {output_path.resolve()}")


if __name__ == "__main__":
    main()
