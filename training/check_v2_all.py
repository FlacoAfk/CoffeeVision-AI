"""Check status of all 6 V2 kernels using correct auth per account."""
import os

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

KERNELS = [
    ("julianmedinamonje45", "coffeevision-leaf-cnn-v2", "leaf/cnn"),
    ("julianmedinamonje45", "coffeevision-leaf-resnet50-v2", "leaf/resnet50"),
    ("vann234", "coffeevision-grain-cnn-v2", "grain/cnn"),
    ("vann234", "coffeevision-grain-resnet50-v2", "grain/resnet50"),
    ("juanveru", "coffeevision-leaf-efficientnet-v2", "leaf/effnet"),
    ("juanveru", "coffeevision-grain-efficientnet-v2", "grain/effnet"),
]

for user, slug, label in KERNELS:
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKENS[user]
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        s = api.kernels_status(f"{user}/{slug}")
        print(f"  {label:25s} | {s}")
    except Exception as e:
        print(f"  {label:25s} | ERR: {e}")
