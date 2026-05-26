"""Verify Kaggle API auth and list datasets for both accounts."""
import os
import sys

def check_account(token, label):
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
        os.environ.pop(k, None)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(f"{label}: authenticated OK")

    ds = api.dataset_list(mine=True)
    print(f"  Datasets: {len(ds)}")
    for d in ds[:5]:
        ref = getattr(d, "ref", str(d))
        print(f"    - {ref}")

    # Also list kernels
    kernels = api.kernels_list(mine=True)
    print(f"  Kernels: {len(kernels)}")
    for k in kernels[:5]:
        ref = getattr(k, "ref", str(k))
        print(f"    - {ref}")

check_account("KGAT_026351ca8f39d7e6bf42e0d57172c31e", "julianmedinamonje45")
print()
check_account("KGAT_a2b7c5cc854eea821166afe9cc1fa585", "vann234")
