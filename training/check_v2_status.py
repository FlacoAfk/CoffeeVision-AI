"""Check status of all 6 V2 training kernels."""
import os, time

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

KERNELS = [
    {"username": "julianmedinamonje45", "slug": "coffeevision-leaf-cnn-v2", "label": "leaf/cnn"},
    {"username": "julianmedinamonje45", "slug": "coffeevision-leaf-resnet50-v2", "label": "leaf/resnet50"},
    {"username": "vann234", "slug": "coffeevision-grain-cnn-v2", "label": "grain/cnn"},
    {"username": "vann234", "slug": "coffeevision-grain-resnet50-v2", "label": "grain/resnet50"},
    {"username": "juanveru", "slug": "coffeevision-leaf-efficientnet-v2", "label": "leaf/efficientnet"},
    {"username": "juanveru", "slug": "coffeevision-grain-efficientnet-v2", "label": "grain/efficientnet"},
]

STATUS_MEANINGS = {
    0: "NOT_STARTED", 1: "RUNNING", 2: "COMPLETE",
    3: "ERROR", 4: "CANCELLED", 5: "QUEUE",
    6: "IN_PROGRESS", 7: "CANCEL_ACK", 8: "PAUSED",
}


def check_all():
    for k in KERNELS:
        token = KAGGLE_TOKENS[k["username"]]
        os.environ["KAGGLE_API_TOKEN"] = token
        for rk in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
            os.environ.pop(rk, None)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        try:
            status = api.kernels_status(user_name=k["username"], kernel_slug=k["slug"])
            s = str(status)
            # Try to extract status code
            code = None
            for name, code_num in STATUS_MEANINGS.items():
                if str(code_num) in s or name in s.upper():
                    code = name
            print(f"  {k['label']:25s} | {k['username']}/{k['slug']:40s} | {s}")
        except Exception as e:
            print(f"  {k['label']:25s} | {k['username']}/{k['slug']:40s} | ERROR: {e}")


if __name__ == "__main__":
    print("=" * 90)
    print("CoffeeVision V2 Kernel Status")
    print("=" * 90)
    check_all()
    print("=" * 90)
