"""Upload leaf_cls and grain_cls datasets to Kaggle."""
import os
import json
import shutil
from pathlib import Path

DATA_DIR = Path(r"C:\Users\elkaw\Desktop\CoffeeVision AI\training\data")

TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}


def upload_dataset(token, username, slug, title, data_path):
    """Upload a folder as a Kaggle dataset."""
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
        os.environ.pop(k, None)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # Write metadata inside the data directory
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    meta_path = os.path.join(data_path, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Uploading {username}/{slug} from {data_path}...")
    try:
        api.dataset_create_new(data_path, dir_mode="skip")
        print(f"  SUCCESS!")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        # Clean up metadata file
        if os.path.exists(meta_path):
            os.remove(meta_path)


if __name__ == "__main__":
    # Leaf dataset → julianmedinamonje45
    upload_dataset(
        token=TOKENS["julianmedinamonje45"],
        username="julianmedinamonje45",
        slug="coffevision-leaf-cls",
        title="CoffeeVision Leaf Disease Classification",
        data_path=str(DATA_DIR / "leaf_cls"),
    )

    print()

    # Grain dataset → vann234
    upload_dataset(
        token=TOKENS["vann234"],
        username="vann234",
        slug="coffevision-grain-cls",
        title="CoffeeVision Grain Broca Classification",
        data_path=str(DATA_DIR / "grain_cls"),
    )

    print("\nDone! Datasets should be available at:")
    print("  https://www.kaggle.com/datasets/julianmedinamonje45/coffevision-leaf-cls")
    print("  https://www.kaggle.com/datasets/vann234/coffevision-grain-cls")
