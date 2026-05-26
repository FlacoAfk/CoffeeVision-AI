"""Check status of all CoffeeVision kernels across 3 accounts."""
import os

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

KERNELS = {
    "julianmedinamonje45": [
        "coffevision-leaf-cnn-v1",
        "coffevision-leaf-resnet50-v1",
        "coffevision-leaf-efficientnet-v1",
    ],
    "vann234": [
        "coffevision-grain-cnn-v1",
        "coffevision-grain-resnet50-v1",
        "coffevision-grain-efficientnet-v1",
    ],
    "juanveru": [
        "coffevision-leaf-efficientnet-v1",
        "coffevision-grain-efficientnet-v1",
    ],
}

for username, token in KAGGLE_TOKENS.items():
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
        os.environ.pop(k, None)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"\n{'='*60}")
    print(f"Account: {username}")
    print(f"{'='*60}")

    for slug in KERNELS.get(username, []):
        full_slug = f"{username}/{slug}"
        try:
            status = api.kernels_status(full_slug)
            print(f"  {slug}: {status}")
        except Exception as e:
            print(f"  {slug}: ERROR - {e}")
