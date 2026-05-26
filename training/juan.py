import os
os.environ["KAGGLE_API_TOKEN"] = "KGAT_fcb1ba0670cfb603b1a1ab6053eb7961"
os.environ.pop("KAGGLE_USERNAME", None); os.environ.pop("KAGGLE_KEY", None)
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# List all kernels
print("=== juanveru kernels ===")
kernels = api.kernels_list(user="juanveru")
for k in kernels:
    slug = getattr(k, 'ref', '?')
    status = str(getattr(k, 'status', '?'))
    print(f"  {slug}: {status}")
