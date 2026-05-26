"""Convert YOLO OD dataset to classification format.

Reads YOLO-format dataset (images/ + labels/ + data.yaml) and converts
to folder-per-class structure suitable for TensorFlow classification.

For each image, the dominant class (by bounding box area) is used as the
image-level label. Images are copied (not cropped) to preserve full context.

Usage:
    python convert_yolo_to_cls.py --input leaf_raw --output leaf_cls
    python convert_yolo_to_cls.py --input grain_raw --output grain_cls
"""

import argparse
import shutil
import yaml
from collections import Counter, defaultdict
from pathlib import Path


def parse_yolo_label(label_path: Path) -> list[tuple[int, float]]:
    """Parse a YOLO label file and return (class_id, bbox_area) pairs.

    YOLO format per line: class_id x_center y_center width height [polygon_pts...]
    We only care about class_id and bbox area (w*h) for the dominant class.
    """
    entries = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            # If it's a bounding box (first 5 values), compute area
            # If it's a polygon (class + 2n points), count points as proxy
            if len(parts) == 5:
                # Standard YOLO bbox: class x_center y_center width height
                w = float(parts[3])
                h = float(parts[4])
                area = w * h
            else:
                # Polygon format: class x1 y1 x2 y2 ... xn yn
                # Area proxy: number of points / 2
                area = (len(parts) - 1) / 2.0
            entries.append((class_id, area))
    return entries


def convert_dataset(input_dir: Path, output_dir: Path):
    """Convert YOLO dataset to classification folder structure."""
    # Load data.yaml for class names
    yaml_path = input_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"ERROR: data.yaml not found in {input_dir}")
        return

    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config["names"]
    # names can be a list or a dict
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in range(len(class_names))]
    elif isinstance(class_names, list):
        pass
    else:
        print(f"ERROR: unexpected class_names format: {type(class_names)}")
        return

    num_classes = data_config.get("nc", len(class_names))
    print(f"Classes ({num_classes}): {class_names}")

    # Process all splits
    total_images = 0
    class_counts = Counter()
    skipped = 0
    multi_label = 0

    for split in ["train", "valid", "test"]:
        images_dir = input_dir / split / "images"
        labels_dir = input_dir / split / "labels"

        if not images_dir.exists():
            print(f"  Skipping {split} — directory not found")
            continue

        image_files = sorted(
            f for f in images_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        print(f"\nProcessing {split}: {len(image_files)} images")

        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists() or label_path.stat().st_size == 0:
                # No label — skip
                skipped += 1
                continue

            # Parse label to find dominant class
            entries = parse_yolo_label(label_path)
            if not entries:
                skipped += 1
                continue

            # Find dominant class by total bbox area
            class_areas = defaultdict(float)
            for cid, area in entries:
                class_areas[cid] += area

            dominant_class = max(class_areas, key=class_areas.get)

            if len(class_areas) > 1:
                multi_label += 1

            # Get class name
            if dominant_class >= len(class_names):
                print(f"  WARNING: class_id {dominant_class} out of range, skipping {img_path.name}")
                skipped += 1
                continue

            cls_name = class_names[dominant_class]

            # Create output directory
            out_cls_dir = output_dir / cls_name
            out_cls_dir.mkdir(parents=True, exist_ok=True)

            # Copy image
            out_path = out_cls_dir / img_path.name
            if not out_path.exists():
                shutil.copy2(img_path, out_path)

            class_counts[cls_name] += 1
            total_images += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total images: {total_images}")
    print(f"Skipped (no label): {skipped}")
    print(f"Multi-label images: {multi_label} (assigned to dominant class)")
    print(f"\nClass distribution:")
    for cls_name in sorted(class_counts.keys()):
        pct = class_counts[cls_name] / total_images * 100
        bar = "█" * int(pct / 2)
        print(f"  {cls_name:<45} {class_counts[cls_name]:>5} ({pct:5.1f}%) {bar}")

    print(f"\nOutput: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO OD dataset to classification format")
    parser.add_argument("--input", required=True, help="Input YOLO dataset directory (contains data.yaml)")
    parser.add_argument("--output", required=True, help="Output classification directory")
    args = parser.parse_args()

    convert_dataset(Path(args.input), Path(args.output))
