"""Find actual slugs for all CoffeeVision kernels across 3 accounts."""
import os

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

for username, token in KAGGLE_TOKENS.items():
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
        os.environ.pop(k, None)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    kernels = api.kernels_list(mine=True)
    print(f"\n{username} ({len(kernels)} kernels):")
    for k in kernels:
        ref = getattr(k, "ref", "")
        status = getattr(k, "status", "?")
        # Only show coffevision kernels
        if "coffevision" in ref.lower() or "coffevision" in str(getattr(k, "title", "")).lower():
            print(f"  {ref} | status={status}")
