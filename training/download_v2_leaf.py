"""Download V2 leaf models (completed, but have biased split - for reference only)."""
import os

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
}

COMPLETED = [
    ("julianmedinamonje45", "coffeevision-leaf-cnn-v2", "leaf_custom_cnn_v2"),
    ("julianmedinamonje45", "coffeevision-leaf-resnet50-v2", "leaf_resnet50_v2"),
    ("juanveru", "coffeevision-leaf-efficientnet-v2", "leaf_efficientnetb0_v2"),
]

OUTPUT_DIR = r"C:\Users\elkaw\Desktop\CoffeeVision AI\training\models\v2_leaf_reference"

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
        for f in os.listdir(dest):
            fpath = os.path.join(dest, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                print(f"       {f} ({size:,} bytes)")
            else:
                print(f"       {f}/")
    except Exception as e:
        print(f"    ERR: {e}")

print("\nDone!")
