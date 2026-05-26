import os

TOKENS = {
    "julianmedinamonje45": "KGAT_5e205dcd6f68c3b4ff4c4c121a149bf8",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
    "juanveru": "KGAT_fcb1ba0670cfb603b1a1ab6053eb7961",
}

KERNELS = [
    ("julianmedinamonje45", "coffeevision-leaf-resnet50-v4", "Leaf ResNet50"),
    ("julianmedinamonje45", "coffeevision-leaf-customcnn-v4", "Leaf Custom CNN"),
    ("vann234", "coffeevision-grain-resnet50-v4", "Grain ResNet50"),
    ("vann234", "coffeevision-grain-customcnn-v4", "Grain Custom CNN"),
    ("juanveru", "coffeevision-leaf-effnet-v4", "Leaf EfficientNet"),
    ("juanveru", "coffeevision-grain-effnet-v4", "Grain EfficientNet"),
]

for user, slug, label in KERNELS:
    token = TOKENS[user]
    os.environ["KAGGLE_API_TOKEN"] = token
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    try:
        status = api.kernels_status(kernel=f"{user}/{slug}")
        s = getattr(status, "status", "?")
        rt = getattr(status, "totalRuntime", "?")
        print(f"{label}: status={s}, runtime={rt}")
    except Exception as e:
        err = str(e)[:120]
        print(f"{label}: {err}")
