"""Download V4 completed kernels (6 binary models, 3 accounts)."""
import os

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_5e205dcd6f68c3b4ff4c4c121a149bf8",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
    "juanveru": "KGAT_fcb1ba0670cfb603b1a1ab6053eb7961",
}

COMPLETED = [
    ("julianmedinamonje45", "coffeevision-leaf-customcnn-v4", "leaf_custom_cnn"),
    ("julianmedinamonje45", "coffeevision-leaf-resnet50-v4", "leaf_resnet50"),
    ("vann234", "coffeevision-grain-customcnn-v4", "grain_custom_cnn"),
    ("vann234", "coffeevision-grain-resnet50-v4", "grain_resnet50"),
    ("juanveru", "coffeevision-leaf-effnet-v4", "leaf_efficientnetb0"),
    ("juanveru", "coffeevision-grain-effnet-v4", "grain_efficientnetb0"),
]

OUTPUT_DIR = r"C:\Users\elkaw\Desktop\CoffeeVision AI\training\models\v4"

for user, slug, label in COMPLETED:
    print(f"\n  Downloading {label} ({user}/{slug})...")
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKENS[user]
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        dest = os.path.join(OUTPUT_DIR, label)
        os.makedirs(dest, exist_ok=True)
        api.kernels_output(f"{user}/{slug}", dest)
        print(f"    -> Downloaded to {dest}")
        for root, dirs, files in os.walk(dest):
            for f in files:
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, dest)
                print(f"       {rel} ({os.path.getsize(fpath):,} bytes)")
    except Exception as e:
        print(f"    ERR: {e}")

print("\nDone!")
