"""Download V3 completed kernels (grain models with stratified split)."""
import os

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

COMPLETED = [
    ("vann234", "coffeevision-grain-cnn-v3", "grain_custom_cnn"),
    ("vann234", "coffeevision-grain-resnet50-v3", "grain_resnet50"),
    ("juanveru", "coffeevision-grain-efficientnet-v3", "grain_efficientnetb0"),
]

OUTPUT_DIR = r"C:\Users\elkaw\Desktop\CoffeeVision AI\training\models\v3"

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
                print(f"       {f} ({os.path.getsize(fpath):,} bytes)")
            else:
                print(f"       {f}/")
    except Exception as e:
        print(f"    ERR: {e}")

print("\nDone!")
