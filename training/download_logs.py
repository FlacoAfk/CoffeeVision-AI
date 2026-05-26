"""Download error logs from all CoffeeVision kernels."""
import os

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

KERNELS = {
    "julianmedinamonje45": [
        "coffeevision-leaf-resnet50-v1",
        "coffeevision-leaf-cnn-v1",
    ],
    "vann234": [
        "coffeevision-grain-resnet50-v1",
        "coffeevision-grain-cnn-v1",
    ],
    "juanveru": [
        "coffeevision-grain-efficientnet-v1",
        "coffeevision-leaf-efficientnet-v1",
    ],
}

OUTPUT_DIR = r"C:\Users\elkaw\Desktop\CoffeeVision AI\training\error_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for username, token in KAGGLE_TOKENS.items():
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
        os.environ.pop(k, None)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    for slug in KERNELS.get(username, []):
        full_slug = f"{username}/{slug}"
        out_dir = os.path.join(OUTPUT_DIR, slug)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Downloading output for {full_slug}...")
        try:
            api.kernels_output(full_slug, path=out_dir, force=True)
            files = os.listdir(out_dir)
            print(f"  Got {len(files)} files: {files}")
        except Exception as e:
            print(f"  ERROR: {str(e)[:120]}")
